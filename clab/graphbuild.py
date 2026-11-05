# -*- coding: utf-8 -*-
"""clab.graphbuild - сборка «тел из данных» из ЛЮБОГО источника.

Два формата, которые читает хук ai_ui (form='graph:<файл>', app/scene/connectome.py):

  connectome-graph/1      тело:     узлы с координатами + рёбра
      {"format","meta","nodes":[{"id","x","y","z","w","c"}],"edges":[{"a","b","w"}]}
  connectome-activity/1   импульс:  покадровая активность узлов во времени
      {"format","dt","steps","ids":[...],"drive":[...],"frames":[[...]],"meta":{}}

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Глаз (ai_ui) умеет читать ровно эти два формата, и ему
всё равно, откуда данные: коннектом червя, шина сигналов, заметки агента, лог
инструментов. Поэтому вся «универсальность» живёт здесь: новый источник - это
новый адаптер в scripts/, а форматы, запись, раскладки и проверки не трогаются.

Модуль зависит только от стандартной библиотеки.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FORMAT_GRAPH = "connectome-graph/1"
FORMAT_ACTIVITY = "connectome-activity/1"

TAU = math.tau
GOLDEN = (1.0 + 5.0 ** 0.5) / 2.0

# Пределы читателя (app/scene/connectome.py): MAX_BYTES/MAX_NODES/MAX_EDGES/MAX_FRAMES
LIMIT_NODES = 50000
LIMIT_EDGES = 400000
LIMIT_FRAMES = 20000
LIMIT_BYTES = 64 * 1024 * 1024


def seed_of(name) -> float:
    """Детерминированный угол 0..1 из имени.

    Встроенный hash() у строк солится при каждом запуске процесса - раскладка
    «поехала» бы от запуска к запуску, и тело меняло форму само по себе.
    """
    h = hashlib.md5(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big") / float(1 << 48)


def fit(points, span: float = 2.0):
    """Нормировать точки в куб со стороной span (центр в 0), сохранив пропорции."""
    pts = [tuple(float(v) for v in p[:3]) for p in points]
    if not pts:
        return []
    mins = [min(p[i] for p in pts) for i in range(3)]
    maxs = [max(p[i] for p in pts) for i in range(3)]
    ctr = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    ext = max(maxs[i] - mins[i] for i in range(3)) or 1.0
    k = span / ext
    return [[(p[i] - ctr[i]) * k for i in range(3)] for p in pts]


def norm01(values):
    """Список -> 0..1 по максимуму (нулевой максимум = нули, без деления на ноль)."""
    vals = [float(v) for v in values]
    mx = max(vals) if vals else 0.0
    if mx <= 0:
        return [0.0 for _ in vals]
    return [v / mx for v in vals]


def layout_atom(names, weights=None, hub=None, radius: float = 0.9, pull: float = 0.55):
    """«Социальный атом»: хаб в центре, остальные по сфере, тяжёлые ближе к центру.

    Сфера заполняется по золотой спирали (равномерно по площади), поворот -
    сдвиг от md5 имени. Возвращает список (name, x, y, z).
    """
    names = list(names)
    if not names:
        return []
    w = {str(n): float((weights or {}).get(n, 1.0)) for n in names}
    mx = max(w.values()) or 1.0
    if hub not in w:
        hub = max(w, key=w.get)
    rest = sorted(n for n in names if n != hub)
    out = []
    if hub is not None:
        out.append((hub, 0.0, 0.0, 0.0))
    n = len(rest)
    for i, name in enumerate(rest):
        y = 1.0 - 2.0 * (i + 0.5) / n
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        ang = TAU * (i / GOLDEN + seed_of(name))
        rr = radius * (1.0 - pull * (w[name] / mx))
        out.append((name, rr * ring * math.cos(ang), rr * y, rr * ring * math.sin(ang)))
    return out


def layout_spiral(n: int, turns: float = 6.0, r0: float = 0.25, r1: float = 0.95,
                  height: float = 1.9, jitter: float = 0.0):
    """Спираль по порядку (0 = самый старый). Время читается как ось вверх.

    Для лент, у которых нет координат, но есть порядок: записи памяти, события,
    история вызовов. Форма честно называется спиралью в meta - это раскладка,
    а не анатомия.
    """
    pts = []
    n = int(n)
    for i in range(n):
        t = i / max(1, n - 1)
        ang = TAU * turns * t
        rr = r0 + (r1 - r0) * t
        x, y, z = rr * math.cos(ang), (t - 0.5) * height, rr * math.sin(ang)
        if jitter > 0.0:
            # Без дрожания точки ложатся на КРИВУЮ: при сотнях записей шаг вдоль
            # витка меньше радиуса узла, и лента превращается в сплошной шнур.
            # Смещение детерминировано (md5), поэтому форма повторяема.
            s1, s2 = seed_of("j%d" % i), seed_of("k%d" % i)
            off = float(jitter) * (0.30 + 0.70 * s2)
            oang = TAU * s1
            x += off * math.cos(oang)
            z += off * math.sin(oang)
            y += float(jitter) * 0.5 * (s1 - 0.5)
        pts.append((x, y, z))
    return pts


def write_graph(path, nodes, edges, meta=None) -> dict:
    """Записать тело connectome-graph/1. nodes: [{"id","x","y","z","w","c"}], edges по индексам."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(nodes)
    if n > LIMIT_NODES:
        raise ValueError("узлов %d > предела %d" % (n, LIMIT_NODES))
    clean, dropped = [], 0
    for e in edges:
        a, b = int(e["a"]), int(e["b"])
        if 0 <= a < n and 0 <= b < n and a != b:
            clean.append({"a": a, "b": b, "w": round(float(e.get("w", 1.0)), 4)})
        else:
            dropped += 1
    if len(clean) > LIMIT_EDGES:
        raise ValueError("рёбер %d > предела %d" % (len(clean), LIMIT_EDGES))
    out_meta = dict(meta or {})
    out_meta.setdefault("source", "clab.graphbuild")
    out_meta.update({"nodes": n, "edges": len(clean), "dropped_edges": dropped})
    doc = {"format": FORMAT_GRAPH, "meta": out_meta, "nodes": nodes, "edges": clean}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8", newline="")
    size = path.stat().st_size
    if size > LIMIT_BYTES:
        raise ValueError("файл тела %.1f МБ > предела" % (size / 1e6))
    return {"path": str(path), "nodes": n, "edges": len(clean), "bytes": size}


def write_activity(path, dt, ids, frames, drive=None, meta=None) -> dict:
    """Записать импульс connectome-activity/1.

    frames - список кадров, каждый кадр список значений 0..1 по порядку ids
    (короткие кадры добиваются нулями: читатель ожидает полную строку).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in ids]
    if not ids or not frames:
        raise ValueError("активность: нужны ids и frames")
    if len(frames) > LIMIT_FRAMES:
        raise ValueError("кадров %d > предела %d" % (len(frames), LIMIT_FRAMES))
    n = len(ids)
    rows = []
    for row in frames:
        vals = list(row) if isinstance(row, (list, tuple)) else [row]
        out = [min(1.0, max(0.0, float(v))) for v in vals[:n]]
        out += [0.0] * (n - len(out))
        rows.append([round(v, 4) for v in out])
    drv = [min(1.0, max(-1.0, float(v))) for v in (drive or [])][:len(rows)]
    drv += [0.0] * (len(rows) - len(drv))
    doc = {
        "format": FORMAT_ACTIVITY,
        "dt": float(dt) if dt and dt > 0 else 0.05,
        "steps": len(rows),
        "ids": ids,
        "drive": [round(v, 4) for v in drv],
        "frames": rows,
        "meta": dict(meta or {}),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8", newline="")
    size = path.stat().st_size
    if size > LIMIT_BYTES:
        raise ValueError("файл активности %.1f МБ > предела" % (size / 1e6))
    return {"path": str(path), "frames": len(rows), "ids": n, "bytes": size,
            "period": round(len(rows) * doc["dt"], 2)}


def node(cid, xyz, w=1.0, color=(0.62, 0.66, 0.72), **extra) -> dict:
    """Узел тела: id, координаты, вес, цвет (0..1). Лишние поля идут в JSON как есть."""
    x, y, z = (float(v) for v in xyz[:3])
    r, g, b = (float(v) for v in color[:3])
    out = {"id": str(cid), "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
           "w": round(float(w), 4), "c": [round(r, 4), round(g, 4), round(b, 4)]}
    out.update(extra)
    return out


def hexcolor(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(min(1.0, max(0.0, v)) * 255)) for v in rgb[:3])
