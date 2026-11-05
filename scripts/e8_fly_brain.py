# -*- coding: utf-8 -*-
"""E8 - тело-мозг: ВЕСЬ мозг дрозофилы реальными сомами (ФОРМА, а не горсть).

Зачем: тело из центрального комплекса (E7) честное, но бесформенное - 2317
нейронов в маленьком объёме читаются сгустком. Нужна ФОРМА: узнаваемый мозг.
У FlyWire сомы лежат корковой мантией ВОКРУГ нейропиля, поэтому выборка всех
нейронов даёт силуэт мозга (зрительные доли + центральный мозг), а не шар.

Что делаем:
  * узлы - равномерная (seed) выборка 20 000 из 138 639 сом, координаты РЕАЛЬНЫЕ;
  * рёбра - только ЛОКАЛЬНЫЕ связи: порог по числу синапсов + обрезка по
    расстоянию. Дальние связи тянутся хордами через весь объём и дают «спагетти»:
    читается не структура, а случайные линии. Оставляем близкие - это проводка;
  * медиатор - по ИСХОДЯЩИМ синапсам (данные, а не приор), палитра как у червя;
  * файла активности НЕТ и не будет: симулировать 138k нейронов мы не берёмся.
    Импульс этому телу дают ДЕЙСТВИЯ агента (зоны в рендерере, pulse=0) - то
    есть тело светится тем, что я делаю: инструмент, ответ, ошибка, размышление.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from clab import graphbuild as gb                      # noqa: E402
from e7_fly_graph import load_fly, rows, NT_COLORS, NT_KEYS   # noqa: E402

AIUI = Path(r"D:\Documents\Projects\ai_ui")
OUT = ROOT / "out"

N_NODES = 20000        # узлы: выборка сом (лимит читателя - 50 000)
MIN_W = 5              # связь засчитываем от 5 синапсов (шум отсекаем)
DIST_Q = 0.55          # оставляем ближние 55% связей по длине - локальная проводка
MAX_EDGES = 120000     # лимит читателя - 400 000, держим запас
SEED = 20260920


def main() -> int:
    t0 = time.time()
    coor, labels, names, W = load_fly()
    n_all = len(coor)
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(n_all, size=min(N_NODES, n_all), replace=False))
    print("узлы: %d из %d сом (равномерная выборка, seed %d)" % (len(sel), n_all, SEED))

    tot = W.TOT.tocsc()
    total = rows(tot) + np.asarray(tot.sum(axis=0)).ravel()

    sub = tot[sel][:, sel].tocoo()
    keep = sub.data >= MIN_W
    ea, eb, ew = sub.row[keep], sub.col[keep], sub.data[keep]
    print("связей внутри выборки (w>=%d): %d" % (MIN_W, len(ea)))

    pos_nm = np.asarray(coor, np.float64)[sel]
    dist = np.linalg.norm(pos_nm[ea] - pos_nm[eb], axis=1)
    lim = float(np.quantile(dist, DIST_Q))
    keep = dist <= lim
    ea, eb, ew = ea[keep], eb[keep], ew[keep]
    print("после обрезки дальних (<= %.0f нм, квантиль %.2f): %d рёбер"
          % (lim, DIST_Q, len(ea)))

    if len(ea) > MAX_EDGES:
        idx = np.argsort(-ew)[:MAX_EDGES]
        idx.sort()
        ea, eb, ew = ea[idx], eb[idx], ew[idx]
        print("урезано по силе до %d рёбер" % len(ea))

    keys = list(NT_KEYS)
    stack = np.vstack([rows(getattr(W, c).tocsc())[sel] for c in keys])
    best = np.argmax(stack, axis=0)
    best_val = stack[best, np.arange(len(sel))]
    nt_name = np.array([keys[b] if v > 0 else "" for b, v in zip(best, best_val)])

    pos = np.asarray(gb.fit(pos_nm, span=2.0), np.float64)
    ptp = (pos.max(axis=0) - pos.min(axis=0))
    print("габарит тела (а.е.): %.2f x %.2f x %.2f" % tuple(ptp))

    nodes = [gb.node(int(sel[i]), pos[i], w=max(1.0, float(total[sel[i]])),
                     color=NT_COLORS.get(nt_name[i], NT_COLORS[""]), nt=str(nt_name[i]))
             for i in range(len(sel))]
    edges = [{"a": int(a), "b": int(b), "w": float(w)} for a, b, w in zip(ea, eb, ew)]
    nt_count = {c: int((nt_name == c).sum()) for c in keys if (nt_name == c).sum()}
    meta = {
        "source": "FlyWire FAFB v783 (Zenodo 18555170, CC-BY-4.0) - ВЕСЬ мозг (выборка сом)",
        "kind": "fly-brain",
        "taxon": "Drosophila melanogaster (самка, взрослая)",
        "neurons_total": int(n_all),
        "nodes": int(len(sel)), "edges": int(len(ea)),
        "synapses": int(ew.sum()), "min_w": MIN_W,
        "edge_filter": "локальные связи: синапсов >= %d и длина <= %.0f нм (квантиль %.2f)"
                       % (MIN_W, lim, DIST_Q),
        "coords": "реальные координаты сом, нм -> [-1,1] (пропорции сохранены)",
        "medial": {},
        "nt_count": nt_count,
        "impulse": "ДЕЙСТВИЯ агента (зоны рендерера, форма надевается с ?pulse=0) - "
                   "файла активности нет: 138k нейронов не симулируем",
        "citation": "Dorkenwald et al., Nature 2024 (FlyWire); дамп: Betzel et al. 2026",
    }
    info = gb.write_graph(OUT / "fly_brain.json", nodes, edges, meta)
    print("записан %s (%.1f МБ)" % (info.get("path"), info.get("bytes", 0) / 1e6))

    dst = AIUI / "data" / "bodies" / "fly_brain.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "fly_brain.json", dst)
    print("скопирован в %s" % dst)
    print("готово за %.0f с" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
