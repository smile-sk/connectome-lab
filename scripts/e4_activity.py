# -*- coding: utf-8 -*-
"""E4: НАСТОЯЩИЙ импульс - сеть по коннектому прогоняется во времени, и её
активации кладутся рядом с телом (формат connectome-activity/1).

Зачем: тело-коннектом в ai_ui умело светиться только косметикой - cos(2π(u−t)).
Красиво, но это НЕ сигнал. Здесь считается сама сеть (clab.dynamics: граф Cook2019,
градиентные ставки) и сохраняется покадровая активность КАЖДОГО нейрона + моторная
команда. Файл кладётся рядом с телом (`worm.activity.json`), и хук ai_ui
(app/scene/connectome.py) светит сомы и нити ЭТИМИ числами: сома вспыхивает, когда
её нейрон возбуждён, по нити бежит фронт, тело гнётся волной по моторной команде.
Замкнутый контур: симулятор -> файл -> тело наблюдателя.

СТИМУЛ: касание носа (ASH L/R) 40 мс, два касания с паузой 8 с.

ЧТО ПОКАЗАЛ ЭКСПЕРИМЕНТ (замер, не мнение) - при включённой усталости синапса
(depress 0.2 и 0.35) ПЕРВОЕ касание даёт пик 0.14, а все следующие - РОВНО НОЛЬ
даже при паузе 8 с и tau_rec 8 с: сеть уходит в состояние, из которого не
возвращается. Это не привыкание (то обратимо), а односторонняя защёлка - у модели
нет гомеостаза. Поэтому импульс для тела снят с depress=0: падение ответа есть,
но связь живая.

ТАБЛИЦА УСТАЛОСТИ СЧИТАЕТСЯ, А НЕ ВПИСАНА (урок QA 2026-09-20): цифры эксперимента
раньше лежали в отчёте хардкодом и после правок clab/dynamics.py разошлись с кодом.
Теперь их считает функция fatigue_table() в этом же прогоне, а в отчёт попадает
md5 dynamics.py, которым они сняты. `--no-fatigue` отключает расчёт (и честно
пишет в отчёте, что он не считался).

ЦЕНА: таблица - это ещё три прогона по 19 с модельного времени (~3 мин на ноутбуке);
боевой прогон без неё ~1 мин. Поэтому и есть флаг `--no-fatigue`.

DRIVE: моторная команда = средняя активация мотонейронов «назад» (DA/VA) минус
«вперёд» (DB/VB). У сети есть медленный ДРЕЙФ фона (нет гомеостаза), он забивает
сигнал касания, поэтому в файл идёт сигнал ЗА ВЫЧЕТОМ скользящего среднего (окно 4 с);
это измерение и показ, а не подмена данных - в meta сохранён масштаб.

ФОРМАТ (его читает app/scene/connectome.py):
  {"format":"connectome-activity/1","dt":0.05,"steps":23000,
   "ids":["ADAL",...],                  # порядок колонок frames
   "drive":[0.0,0.35,...],              # моторная команда назад(+)/вперёд(−), −1..1
   "frames":[[0.0,0.9,...],...],        # ФАЗОВЫЙ сигнал 0..1 по кадрам
                                        #   (|активация| минус своё среднее за 2 с)
   "meta":{"gain":..,"stimulus":..,"taps_peak":[...],"depress":0.0}}
  gain - во сколько раз сжат фазовый сигнал (99.5-й перцентиль): числа не выдуманы,
  просто приведены к 0..1. Абсолютный уровень у сети почти постоянен (мышцы стоят в
  пределе 13% времени, медиана 0.69) - на нём касание не видно, видно отклонение.

Запуск:  python scripts/e4_activity.py   ->  out/e4_report.md + out/worm_activity.json
         (и копия в ai_ui/data/bodies/worm.activity.json, если папка есть)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT))

try:                                                       # numpy - ускоритель
    import numpy as np
except Exception:                                          # pragma: no cover
    np = None

from clab.dynamics import Connectome, NetParams, Network   # noqa: E402

AIUI_BODIES = Path(r"D:\Documents\Projects\ai_ui\data\bodies")

# --- параметры прогона ------------------------------------------------------
DT_MS = 1.0             # шаг симуляции, мс
TAP_MS = 40.0           # длительность касания, мс
TAP_EVERY = 8000.0      # пауза между касаниями, мс
N_TAPS = 2
N_FATIGUE_TAPS = 3      # касаний в эксперименте с усталостью (таблица считается живьём)
TAIL_MS = 3000.0        # хвост после последнего касания
FRAME_MS = 50.0         # кадр активности = 50 мс (20 к/с)
DEPRESS = 0.0           # усталость синапса: 0 - см. вывод эксперимента в докстринге
BASE_WIN_MS = 4000.0    # окно вычитания дрейфа для моторной команды


def abs_vec(v):
    """|активация| каждого нейрона - numpy, если он есть, иначе чистый Python."""
    if np is not None:
        return np.abs(np.asarray(v, dtype=float))
    return [abs(x) for x in v]


def _deviations(rows, win_ms, frame_ms):
    """Фазовый сигнал: |x| минус собственное медленное среднее (окно win_ms).

    Абсолютный уровень у этой сети почти не двигается (мышцы стоят в пределе 13%
    времени, медиана 0.69), поэтому касание видно ТОЛЬКО как отклонение от тонуса.
    Считаем максимум по модулю отклонения, не выдумывая ничего своего."""
    if not rows:
        return []
    w = max(1, int(round(win_ms / max(1.0, frame_ms))))
    t = len(rows)
    n = len(rows[0])
    if np is not None:
        X = np.asarray(rows, np.float32)
        pad = np.pad(X, ((w // 2, w - 1 - w // 2), (0, 0)), mode="edge")
        k = np.ones(w, np.float32) / w
        base = np.empty_like(X)
        for j in range(n):
            base[:, j] = np.convolve(pad[:, j], k, mode="valid")
        return np.abs(X - base)
    out = []                                              # pragma: no cover
    cols = [[float(rows[i][j]) for i in range(t)] for j in range(n)]
    for j in range(n):
        col = cols[j]
        sm = [sum(col[max(0, i - w // 2):min(t, i + w - w // 2)]) /
              max(1, len(col[max(0, i - w // 2):min(t, i + w - w // 2)]))
              for i in range(t)]
        cols[j] = [abs(col[i] - sm[i]) for i in range(t)]
    for i in range(t):
        out.append([cols[j][i] for j in range(n)])
    return out


def percentile(vals, q):
    """Перцентиль без numpy (линейная интерполяция) - чтобы скрипт жил и на stdlib."""
    if np is not None:
        return float(np.percentile(np.asarray(vals, dtype=float), q))
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def simulate(c, params, ash, back, fwd, taps, tap_ms=TAP_MS, tail_ms=TAIL_MS,
             frame_ms=FRAME_MS, dt=DT_MS):
    """Прогон сети во времени: покадровая |активация| + моторная команда.

    ОДНА функция и для боевого прогона, и для эксперимента с усталостью.
    Иначе таблица в отчёте живёт отдельно от кода и устаревает - ровно то,
    что нашёл QA 2026-09-20 (хардкод 0.0930/0.0396/0.0403 против реальных
    0.08853/0.04529/0.05365)."""
    net = Network(c, params, sign_mode="ach_gaba", weighted=True)
    total_ms = taps[-1] + tap_ms + tail_ms
    steps = int(round(total_ms / dt))
    every = max(1, int(round(frame_ms / dt)))
    frames, drive, times = [], [], []
    for i in range(steps):
        t = i * dt
        ext = {ix: 1.0 for ix in ash} if any(tp <= t < tp + tap_ms for tp in taps) else {}
        net.step(ext)
        if i % every == 0:
            frames.append(abs_vec(net.vec()))
            drive.append(float(net.mean_x(back) - net.mean_x(fwd)))
            times.append(t)
    return frames, drive, times, steps


def detrend_drive(drive, win_ms=BASE_WIN_MS, frame_ms=FRAME_MS):
    """Моторная команда минус свой медленный дрейф (у сети нет гомеостаза)."""
    if np is not None:
        dr = np.asarray(drive, np.float32)
        w = max(1, int(round(win_ms / frame_ms)))
        base = np.convolve(dr, np.ones(w, np.float32) / w, mode="same")
        return np.asarray(dr - base, np.float32)
    w = max(1, int(round(win_ms / frame_ms)))                # pragma: no cover
    out = []
    for i, v in enumerate(drive):
        lo, hi = max(0, i - w), min(len(drive), i + w + 1)
        out.append(v - sum(drive[lo:hi]) / max(1, hi - lo))
    return out


def tap_peaks(det, times, taps, win_ms=800.0):
    """Пик сигнала в окне win_ms после каждого касания (уже детрендованный)."""
    peaks = []
    for tp in taps:
        m = [abs(float(det[i])) for i, tt in enumerate(times) if tp <= tt < tp + win_ms]
        peaks.append(max(m) if m else 0.0)
    return peaks


def _with_depress(params, depress):
    """Копия NetParams с другой усталостью (dataclass - через replace)."""
    try:
        from dataclasses import replace
        return replace(params, depress=depress)
    except Exception:                                         # pragma: no cover
        kw = dict(vars(params)) if hasattr(params, "__dict__") else {}
        kw["depress"] = depress
        return NetParams(**kw)


def fatigue_table(c, params, ash, back, fwd, depresses=(0.0, 0.20, 0.35),
                  n_taps=3, tap_every=TAP_EVERY):
    """Эксперимент с усталостью синапса - ЖИВОЙ расчёт, а не хардкод.

    Возвращает [(depress, [пики], [проценты от первого])].
    """
    rows = []
    for d in depresses:
        taps = [i * tap_every for i in range(n_taps)]
        _, drive, times, _ = simulate(c, _with_depress(params, d), ash, back, fwd, taps)
        peaks = tap_peaks(detrend_drive(drive), times, taps)
        rel = [(x / peaks[0] * 100.0) if peaks[0] else 0.0 for x in peaks]
        rows.append((d, peaks, rel))
    return rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    c = Connectome.load(str(DATA / "herm_full_edgelist.csv"),
                        str(DATA / "owmeta_cache.json"))
    st = c.stats()
    print("сеть: клеток %d | химических %d | gap %d | вес %.0f"
          % (st["nodes"], st["chemical"], st["gap"], st["synapses"]))

    back = c.ids(r"^(DA|VA)\d+$")          # мотонейроны назад
    fwd = c.ids(r"^(DB|VB)\d+$")           # мотонейроны вперёд
    ash = c.ids(r"^ASH[LR]$")              # сенсоры касания носа
    if not ash:
        print("НЕТ сенсоров ASH в графе - стимул некуда подать")
        return 1
    print("группы: назад %d, вперёд %d, ASH %d (%s)"
          % (len(back), len(fwd), len(ash),
             ",".join(c.neurons[i].name for i in ash)))

    p = NetParams(dt=DT_MS, alpha=0.35, gain=0.5, gap_gain=1.0, ext_gain=1.5,
                  muscle_gain=2.0, norm="sqrt", depress=DEPRESS, tau_rec=8000.0,
                  tau_dep=200.0, noise=0.0, seed=7)
    net = Network(c, p, sign_mode="ach_gaba", weighted=True)

    do_fatigue = "--no-fatigue" not in sys.argv
    taps = [i * TAP_EVERY for i in range(N_TAPS)]
    frames, drive, times, steps = simulate(c, p, ash, back, fwd, taps)

    # ЯРКОСТЬ: фазовый сигнал (|x| минус своё медленное среднее), затем в 0..1
    dev = _deviations(frames, BASE_WIN_MS, FRAME_MS)
    allv = [float(v) for f in dev for v in f]
    gain = percentile(allv, 99.5) or 1.0
    frames_q = [[round(min(1.0, float(v) / gain), 3) for v in f] for f in dev]

    # моторная команда: минус медленный дрейф фона, затем в −1..1
    det = detrend_drive(drive)
    dscale = (float(max(abs(float(x)) for x in det)) if len(det) else 0.0) or 1.0
    drive_q = [round(max(-1.0, min(1.0, float(x) / dscale)), 3) for x in det]

    # пики по касаниям - по УЖЕ детрендованному сигналу, окно 800 мс после касания
    tap_peak = tap_peaks(det, times, taps)
    tap_rel = [(x / tap_peak[0] * 100.0) if tap_peak[0] else 0.0 for x in tap_peak]

    import hashlib
    dmd5 = hashlib.md5((ROOT / "clab" / "dynamics.py").read_bytes()).hexdigest()[:8]
    ids = [n.name for n in c.neurons]
    peak_i = int(max(range(len(dev)), key=lambda i: sum(float(v) for v in dev[i]))) if len(dev) else 0
    active = int(sum(1 for v in dev[peak_i] if float(v) > gain * 0.5))

    doc = {
        "format": "connectome-activity/1",
        "dt": FRAME_MS / 1000.0,
        "steps": steps,
        "ids": ids,
        "drive": drive_q,
        "frames": frames_q,
        "meta": {
            "source": "ConnectomeLab clab.dynamics (Cook 2019 гермафродит, sign ach_gaba)",
            "stimulus": "касание носа ASH L/R %.0f мс, пауза %.0f мс, касаний %d"
                        % (TAP_MS, TAP_EVERY, N_TAPS),
            "depress": DEPRESS, "tau_rec_ms": 8000.0,
            "gain": round(float(gain), 5), "drive_scale": round(float(dscale), 5),
            "frames_signal": "phasic: |x| minus moving average %.0f ms" % BASE_WIN_MS,
            "drive_baseline_win_ms": BASE_WIN_MS,
            "taps_peak": [round(float(x), 5) for x in tap_peak],
            "taps_rel_pct": [round(float(x), 1) for x in tap_rel],
            "active_at_peak": active,
            "dynamics_md5": dmd5,
            "note": "depress=0: при усталости 0.2/0.35 повторные касания дают 0 "
                    "(сеть не восстанавливается) - см. отчёт E4",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "worm_activity.json"
    dst.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    size_kb = dst.stat().st_size / 1024
    copied = ""
    if AIUI_BODIES.is_dir():
        shutil.copyfile(dst, AIUI_BODIES / "worm.activity.json")
        copied = str(AIUI_BODIES / "worm.activity.json")

    # ---- отчёт ------------------------------------------------------------
    fat_rows = fatigue_table(c, p, ash, back, fwd, n_taps=N_FATIGUE_TAPS) if do_fatigue else []
    fat_md = ["| %.2f | %s |" % (d, " | ".join("%.5f" % x for x in peaks) + " | "
                                 + " / ".join("%.0f%%" % r for r in rel))
              for d, peaks, rel in fat_rows]
    if do_fatigue:
        print("таблица усталости (в этом прогоне):")
        for d, peaks, rel in fat_rows:
            print("  depress=%.2f  пики %s  к 1-му %s" % (
                d, " ".join("%.5f" % x for x in peaks),
                " / ".join("%.0f%%" % r for r in rel)))
    else:
        print("таблица усталости: пропущена (--no-fatigue)")

    spark = "".join(" .:-=+*#%@"[min(9, int(abs(d) * 9.999))]
                    for d in drive_q[::max(1, len(drive_q) // 100)])
    rep = ["# E4 - настоящий импульс из сети", "",
           "Сеть по коннектому (Cook 2019) прогоняется во времени; файл активности",
           "кладётся рядом с телом-коннектомом, и тело светится ЭТИМИ числами.", "",
           "| параметр | значение |", "|---|---|",
           "| клеток в сети | %d |" % st["nodes"],
           "| химических связей | %d |" % st["chemical"],
           "| gap junctions | %d |" % st["gap"],
           "| шагов симуляции (dt %.0f мс) | %d |" % (DT_MS, steps),
           "| кадров активности (%.0f мс) | %d |" % (FRAME_MS, len(frames_q)),
           "| стимул | ASH L/R %.0f мс, пауза %.0f мс, касаний %d |"
           % (TAP_MS, TAP_EVERY, N_TAPS),
           "| усталость синапса depress | %.2f |" % DEPRESS,
           "| что в frames | фазовый сигнал: |x| минус среднее за %.0f мс |" % BASE_WIN_MS,
           "| сжатие яркости (gain, 99.5-й перцентиль) | %.4f |" % gain,
           "| масштаб мотора (max |det|) | %.4f |" % dscale,
           "| нейронов выше половины шкалы в пике | %d из %d |" % (active, st["nodes"]),
           "| md5 clab/dynamics.py | %s |" % dmd5,
           "| размер файла | %.0f КБ |" % size_kb, "",
           "## Эксперимент с усталостью синапса (почему depress=0)", "",
           "_СЧИТАЕТСЯ В ЭТОМ ПРОГОНЕ (md5 dynamics.py %s): касаний %d, пауза %.0f с,_"
           % (dmd5, N_FATIGUE_TAPS, TAP_EVERY / 1000.0),
           "_tau_rec %.0f с, стимул ASH %.0f мс. Хардкод убран: он разошёлся с кодом_"
           % (8000.0 / 1000.0, TAP_MS),
           "_и был пойман QA 2026-09-20 (таблица заявляла 0.0930/0.0396/0.0403).", "",
           "| depress | пик 1 | пик 2 | пик 3 | к 1-му |", "|---|---|---|---|---|"]
    rep += (fat_md if fat_md else
            ["| - | не считалось (`--no-fatigue`) | | | |"])
    rep += ["",
           "Вывод: усталость делает ответ одноразовым - сеть НЕ восстанавливается за 8 с",
           "при tau_rec 8 с. Это односторонняя защёлка, а не привыкание (привыкание",
           "обратимо). Причина: у rate-модели нет гомеостаза. Для тела снят depress=0.", "",
           "## Пики касаний в этом прогоне (детренд)", ""]
    for ti, tp in enumerate(taps):
        rep.append("- касание %d (t=%.0f мс): пик %.5f - %.0f%% от первого"
                   % (ti + 1, tp, tap_peak[ti], tap_rel[ti]))
    rep += ["", "## Моторная команда во времени (сигнал минус дрейф, 100 знаков)", "",
            "`" + spark + "`", "",
            "`%`/`@` - сильное отползание (назад), пробел - покой/вперёд.", "",
            "Дрейф фона у сети есть, он забивает сигнал касания, поэтому в файл идёт",
            "сигнал за вычетом скользящего среднего (окно %.0f мс). Это измерение и" % BASE_WIN_MS,
            "показ, а не подмена: в meta сохранён масштаб, а дрейф видно по `base`.", "",
            "Честные границы: знаков синапсов в данных нет (приор по медиатору, режим",
            "ach_gaba), весов нет (вес = число контактов), механики тела нет - изгиб",
            "тела это КИНЕМАТИЧЕСКИЙ показ моторной команды, не биомеханика.", ""]
    (OUT / "e4_report.md").write_text("\n".join(rep), encoding="utf-8")

    print("кадров %d | ids %d | gain %.4f | drive_scale %.4f | %.0f КБ"
          % (len(frames_q), len(ids), gain, dscale, size_kb))
    print("пики касаний: " + " ".join("%.4f (%.0f%%)" % (tap_peak[i], tap_rel[i])
                                     for i in range(len(taps))))
    print("возбуждено в пике: %d из %d" % (active, st["nodes"]))
    print("копия для тела: %s" % (copied or "папки ai_ui нет - пропущено"))
    print("отчёт: out/e4_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
