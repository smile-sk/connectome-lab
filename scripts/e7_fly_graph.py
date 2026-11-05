# -*- coding: utf-8 -*-
"""E7: тело МУХИ из реального коннектома + настоящий импульс нашим движком.

Источник (открытый, без логина): Zenodo 18555170, CC-BY-4.0 - производные FlyWire
(самка FAFB, релиз 783): координаты сом, аннотации, коннектом с разбивкой по медиаторам.

  data/fly/coordinates.mat  - coor (138639, 3): координаты сом, нм
  data/fly/annotations.mat  - labels (класс/подкласс/тип/суперкласс) + names (расшифровка)
  data/fly/connectome.mat   - W.{ACH,DA,GABA,GLUT,OCT,SER,TOT}: число синапсов по медиаторам

Почему БЕЗ логина: Codex (codex.flywire.ai) требует аккаунт, а этот дамп - тот же
релиз 783, выложенный открыто под CC-BY. Цитировать: Dorkenwald et al. 2024 (FlyWire)
+ Betzel et al. 2026 (дамп).

Выход:
  out/fly_<популяция>.json          - тело: узлы с РЕАЛЬНЫМИ координатами сом + синапсы
  out/fly_<популяция>.activity.json - импульс: прогон clab.dynamics по этим же связям
  out/e7_report.md                  - отчёт

Популяция выбирается аннотацией: --field super_class --value central (по умолчанию -
центральный комплекс, 2317 нейронов: тот самый кольцевой аттрактор).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clab import graphbuild as gb                     # noqa: E402
from clab.dynamics import Connectome, Edge, NetParams, Network, Neuron   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fly"
OUT = ROOT / "out"
DEFAULT_AIUI = os.environ.get("AIUI_ROOT") or r"D:\Documents\Projects\ai_ui"

# палитра медиаторов - та же, что у червя (e5_body_graph.py): тело читается одинаково
NT_KEYS = {"ACH": "Acetylcholine", "GABA": "GABA", "GLUT": "Glutamate",
           "DA": "Dopamine", "SER": "Serotonin", "OCT": "Octopamine"}
NT_COLORS = {
    "Acetylcholine": (0.36, 0.56, 1.00),
    "Glutamate": (0.38, 0.87, 0.60),
    "GABA": (0.96, 0.74, 0.20),
    "Dopamine": (1.00, 0.60, 0.28),
    "Serotonin": (1.00, 0.42, 0.52),
    "Octopamine": (0.72, 0.50, 0.94),
    "": (0.62, 0.66, 0.72),
}


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def load_fly():
    """Координаты, аннотации и коннектом. Возвращает (coor, labels, names, W)."""
    import scipy.io as sio
    t0 = time.time()
    co = sio.loadmat(str(DATA / "coordinates.mat"), squeeze_me=True, struct_as_record=False)
    an = sio.loadmat(str(DATA / "annotations.mat"), squeeze_me=True, struct_as_record=False)
    lw = sio.loadmat(str(DATA / "connectome.mat"), squeeze_me=True, struct_as_record=False)
    coor = np.asarray(co["coor"], np.float64)
    labels, names = an["labels"], an["names"]
    W = lw["W"]
    log("данные загружены за %.0f с: координат %d, коннектом %s" % (time.time() - t0, len(coor), getattr(W, "TOT").shape))
    return coor, labels, names, W


def name_table(names, field):
    raw = getattr(names, field)
    return [str(x) for x in (raw if hasattr(raw, "__len__") else [raw])]


def select(labels, names, field, value):
    """Индексы нейронов, у которых поле аннотации равно значению (по имени или номеру)."""
    tbl = name_table(names, field)
    if value.isdigit() and int(value) < len(tbl):
        want = int(value)
    elif value in tbl:
        want = tbl.index(value)
    else:
        raise SystemExit("нет значения %r в поле %s. Есть: %s" % (value, field, ", ".join(tbl[:20])))
    idx = np.asarray(getattr(labels, field)).ravel().astype(np.int64)
    sel = np.where(idx == want)[0]
    return sel, tbl[want], tbl


def rows(mat):
    return np.asarray(mat.sum(axis=1)).ravel()


def build(args):
    coor, labels, names, W = load_fly()
    sel, label, tbl = select(labels, names, args.field, args.value)
    log("популяция %s=%s: %d нейронов" % (args.field, label, len(sel)))

    tot = W.TOT.tocsc()
    if args.top and len(sel) > args.top:
        # оставить самые «тяжёлые» по числу синапсов (вход+выход), чтобы тело читалось
        deg = rows(tot)[sel] + np.asarray(tot.sum(axis=0)).ravel()[sel]
        sel = sel[np.argsort(-deg)[:args.top]]
        sel = np.sort(sel)
        log("урезано до топ-%d по числу синапсов" % args.top)

    sub = tot[sel][:, sel].tocoo()
    keep = sub.data >= args.min_w
    ea, eb, ew = sub.row[keep], sub.col[keep], sub.data[keep]
    log("связей внутри популяции: %d (из них синапсов %d, порог веса %g)"
        % (len(ea), int(ew.sum()), args.min_w))
    if len(ea) == 0:
        raise SystemExit("связей нет - снизьте порог --min-w")

    # медиатор нейрона - по ИСХОДЯЩИМ синапсам (нейрон выделяет то, что выпускает)
    nt_tot = {}
    for code in NT_KEYS:
        nt_tot[code] = rows(getattr(W, code).tocsc())[sel]
    stack = np.vstack([nt_tot[c] for c in NT_KEYS])
    best = np.argmax(stack, axis=0)
    best_val = stack[best, np.arange(len(sel))]
    nt_name = np.array([NT_KEYS[list(NT_KEYS)[b]] if v > 0 else "" for b, v in zip(best, best_val)])

    # координаты: реальные (нм) -> [-1,1] с сохранением пропорций
    pos = gb.fit(coor[sel], span=2.0)
    cls_idx = np.asarray(getattr(labels, args.field)).ravel()[sel]
    total = (rows(tot)[sel] + np.asarray(tot.sum(axis=0)).ravel()[sel])
    nodes = [gb.node(int(sel[i]), pos[i], w=max(1.0, float(total[i])),
                     color=NT_COLORS.get(nt_name[i], NT_COLORS[""]),
                     nt=nt_name[i], cls=tbl[int(cls_idx[i])] if 0 <= cls_idx[i] < len(tbl) else "")
             for i in range(len(sel))]
    meta = {
        "source": "FlyWire FAFB v783 (Zenodo 18555170, CC-BY-4.0) - %s=%s" % (args.field, label),
        "kind": "fly",
        "taxon": "Drosophila melanogaster (самка, взрослая)",
        "neurons_total": int(len(coor)),
        "population": label, "population_size": int(len(sel)),
        "edges": int(len(ea)), "synapses": int(ew.sum()), "min_w": args.min_w,
        "coords": "реальные координаты сом, нм -> [-1,1] (пропорции сохранены)",
        "edge_meaning": "стрелка пресинаптический -> постсинаптический, вес = число синапсов",
        "nt_palette": {k: gb.hexcolor(v) for k, v in NT_COLORS.items() if k},
        "citation": "Dorkenwald et al., Nature 2024 (FlyWire); дамп: Betzel et al. 2026",
    }

    # ------- настоящий импульс: наш движок E3 по ЭТИМ связям -------
    c = Connectome()
    c.neurons = [Neuron(str(int(sel[i])), str(int(sel[i])), nt_name[i], (nt_name[i],), False)
                 for i in range(len(sel))]
    c.index = {n.name: i for i, n in enumerate(c.neurons)}
    c.chem = [Edge(int(a), int(b), float(w)) for a, b, w in zip(ea, eb, ew)]

    # стимул - самый сильный ВХОД в популяцию ИЗВНЕ (данные, а не вкус)
    outside = np.setdiff1d(np.arange(tot.shape[0]), sel, assume_unique=False)
    if len(outside) and args.inputs:
        inc = np.asarray(tot[outside][:, sel].sum(axis=0)).ravel()
        stim = int(np.argmax(inc))
        log("стимул: узел %d (входов извне %d) - сильнейший входной порт" % (sel[stim], int(inc[stim])))
    else:
        stim = int(np.argmax([n.w for n in nodes]))
        log("стимул: самый тяжёлый узел %d (внешних входов не считали)" % sel[stim])

    def run(ext_node, frame_ms=None):
        """Прогон сети. ext_node=None - КОНТРОЛЬ без стимула (сеть обязана молчать)."""
        frame_ms = float(frame_ms or args.frame_ms)
        net = Network(c, NetParams(dt=args.dt, depress=0.0, seed=7),
                      sign_mode="ach_gaba", weighted=True)
        steps = int(round(args.total / args.dt))
        every = max(1, int(round(frame_ms / args.dt)))
        rows = []
        for i in range(steps):
            t = i * args.dt
            tap = ext_node is not None and args.tap <= t < args.tap + args.tap_ms
            net.step({ext_node: 1.0} if tap else {})
            if i % every == 0:
                rows.append(np.abs(np.asarray(net.vec(), float)))
        arr = np.asarray(rows, np.float32)
        return arr, np.asarray(arr.mean(axis=1), np.float32)

    t0 = time.time()
    F, D = run(stim)
    F0, _D0 = run(None)                       # КОНТРОЛЬ: без касания
    log("прогон: %d кадров (%.0f мс) + контроль, %.0f с" % (F.shape[0], args.total, time.time() - t0))

    # Сеть без гомеостаза: после касания она защёлкивается на плато и второго
    # касания не покажет (тот же вывод, что дал E4 у червя). Поэтому крупный план -
    # ПЕРВЫЕ миллисекунды: видно, как возбуждение расходится от стимула.
    mx = float(F.max()) or 1.0
    n_neur = F.shape[1]
    final = F[-1]
    thr = 0.25 * (float(F.max()) or 1.0)
    peaked = F.max(axis=0) > thr                      # какие нейроны вообще сработали
    ign = np.where(peaked, np.argmax(F > thr, axis=0), -1)

    if args.mode == "wave":
        # Сеть возбуждается ЦЕЛИКОМ за считанные миллисекунды (проверено: между
        # соседними кадрами 2 мс число сработавших прыгает с 0 до сотен). Честный
        # выход - показать НЕ реальное время, а ПОРЯДОК срабатывания: фронт идёт
        # по сети в том порядке, в каком нейроны действительно вспыхнули.
        order = [int(i) for i in np.argsort(ign) if ign[i] >= 0]
        W, pre = int(args.wave_frames), int(args.pre_frames)
        span = max(1, W - pre)
        frames, drive = [], []
        for k in range(W):
            row = [0.0] * n_neur
            if k >= pre and order:
                hi = int(round((k - pre + 1) * len(order) / span))
                for i in order[:hi]:          # накопление: сработал - горит до конца
                    row[i] = 1.0
            frames.append(row)
            drive.append(sum(1 for i in order[:int(round(max(0, k - pre) * len(order) / span))]) / float(max(1, len(order))))
        shown = "порядок срабатывания, время растянуто (реальное - единицы мс)"
    else:
        frames = [[min(1.0, float(v) / mx) for v in row] for row in F]
        dmax = float(np.abs(D).max()) or 1.0
        drive = [float(v) / dmax for v in D]
        shown = "реальное время, кадр %g мс модели" % args.frame_ms

    ignited = peaked
    first = np.argmax(F > thr, axis=0).astype(np.float32)
    first = np.where(ignited, first, np.nan)
    med = float(np.nanmedian(first)) * args.frame_ms if np.any(ignited) else 0.0
    log("сработало %.1f%% нейронов, средина срабатывания %.0f мс; контроль без стимула: макс %.4f"
        % (100.0 * float(ignited.mean()), med, float(F0.max())))

    act_meta = {
        "source": "clab.dynamics по коннектому FlyWire v783 (%s=%s)" % (args.field, label),
        "kind": "fly",
        "stimulus": "узел %d (сильнейший внешний вход), %g мс с %g мс" % (sel[stim], args.tap_ms, args.tap),
        "sign_mode": "ach_gaba (ACh возбуждение, GABA торможение) - по данным, не по вкусу",
        "depress": 0.0, "frame_ms": args.frame_ms, "dt_ms": args.dt,
        "shown_ms": args.total, "mode": args.mode,
        "slow_motion": "кадр показа = %.2f с (dt активности)" % args.dt_shown,
        "shown": shown,
        "control_no_stim_max": round(float(F0.max()), 6),
        "ignited_pct": round(100.0 * float(ignited.mean()), 2),
        "ignition_median_ms": round(med, 1),
        "drive": "средняя активация популяции (общий тонус) - тело тянется, пока сеть возбуждена",
        "why_one_tap": "сеть без гомеостаза защёлкивается на плато: второе касание ничего не меняет",
    }
    return nodes, [{"a": int(a), "b": int(b), "w": float(w)} for a, b, w in zip(ea, eb, ew)], \
        {"dt": args.dt_shown, "ids": [n["id"] for n in nodes], "frames": frames,
         "drive": drive, "meta": act_meta}, meta


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="тело мухи из коннектома FlyWire")
    ap.add_argument("--field", default="super_class", help="поле аннотации (super_class/class/sub_class/cell_type)")
    ap.add_argument("--value", default="central", help="значение поля")
    ap.add_argument("--min-w", type=float, default=3.0, help="минимальный вес связи (синапсов)")
    ap.add_argument("--top", type=int, default=0, help="оставить топ-N по числу синапсов (0 = все)")
    ap.add_argument("--inputs", action="store_true", default=True, help="искать сильнейший внешний вход")
    ap.add_argument("--no-inputs", dest="inputs", action="store_false")
    ap.add_argument("--tap", type=float, default=200.0, help="касание, мс")
    ap.add_argument("--tap-ms", type=float, default=60.0, help="длительность стимула, мс")
    ap.add_argument("--total", type=float, default=1000.0, help="длина прогона, мс (крупный план волны)")
    ap.add_argument("--frame-ms", type=float, default=20.0, help="шаг кадра модели, мс")
    ap.add_argument("--dt", type=float, default=1.0, help="шаг модели, мс")
    ap.add_argument("--mode", default="wave", choices=("wave", "raw"),
                    help="wave = показ ПОРЯДКА срабатывания (время растянуто), raw = реальное время")
    ap.add_argument("--wave-frames", type=int, default=240, help="кадров в показе волны")
    ap.add_argument("--pre-frames", type=int, default=40, help="тёмных кадров до касания")
    ap.add_argument("--slow", dest="dt_shown", type=float, default=0.15,
                    help="сколько секунд показа на кадр (замедление: 20 мс модели -> 0.15 с)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--copy", action="store_true", help="скопировать тело в ai_ui/data/bodies")
    args = ap.parse_args()

    if not (DATA / "connectome.mat").is_file():
        log("нет данных: %s (сначала скачать Zenodo 18555170)" % DATA)
        return 1

    nodes, edges, act, meta = build(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.value.replace("/", "_")
    g = gb.write_graph(out / ("fly_%s.json" % tag), nodes, edges, meta)
    a = gb.write_activity(out / ("fly_%s.activity.json" % tag), act["dt"], act["ids"],
                          act["frames"], act["drive"], act["meta"])
    log("тело:  узлов %d, рёбер %d, %.0f КБ" % (g["nodes"], g["edges"], g["bytes"] / 1024))
    log("импульс: кадров %d, цикл %.1f с, %.0f КБ" % (a["frames"], a["period"], a["bytes"] / 1024))

    if args.copy:
        dest = Path(DEFAULT_AIUI) / "data" / "bodies"
        dest.mkdir(parents=True, exist_ok=True)
        for frm, to in ((out / ("fly_%s.json" % tag), dest / ("fly_%s.json" % tag)),
                        (out / ("fly_%s.activity.json" % tag), dest / ("fly_%s.activity.json" % tag))):
            to.write_text(frm.read_text(encoding="utf-8"), encoding="utf-8", newline="")
        log("скопировано в %s" % dest)

    nts = Counter(n.get("nt", "") for n in nodes)
    hubs = sorted(nodes, key=lambda n: -n["w"])[:8]
    rep = ["# E7 - тело мухи из реального коннектома", "",
           "- источник: %s" % meta["source"],
           "- цитирование: %s" % meta["citation"],
           "- открыто, без логина: Zenodo 18555170 (CC-BY-4.0)", "",
           "## Цифры", "",
           "| что | значение |", "|---|---|",
           "| нейронов в популяции | %d |" % len(nodes),
           "| связей | %d |" % len(edges),
           "| синапсов | %d |" % meta["synapses"],
           "| размер тела | %.0f КБ |" % (g["bytes"] / 1024),
           "| импульс | %d кадров, цикл %.1f с |" % (a["frames"], a["period"]),
           "| сработало нейронов | %.1f%% (средина %.0f мс) |" % (act["meta"]["ignited_pct"],
                                                                 act["meta"]["ignition_median_ms"]),
           "| КОНТРОЛЬ без стимула | макс активации %.6f |" % act["meta"]["control_no_stim_max"], "",
           "Один стимул, а не серия: сеть без гомеостаза защёлкивается на плато - %s."
           % act["meta"]["why_one_tap"], ""
           "## Медиаторы (по исходящим синапсам)", ""]
    rep += ["- %s: %d" % (k or "не определён", v) for k, v in nts.most_common()]
    rep += ["", "## Хабы (по числу синапсов)", ""]
    rep += ["- %s - синапсов %.0f, медиатор %s, класс %s" % (n["id"], n["w"], n["nt"] or "?", n["cls"])
            for n in hubs]
    (out / "e7_report.md").write_text("\n".join(rep), encoding="utf-8", newline="")
    log("отчёт: %s" % (out / "e7_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
