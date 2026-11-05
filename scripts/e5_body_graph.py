# -*- coding: utf-8 -*-
"""E5: тело из данных - реальный коннектом как файл-тело (формат connectome-graph/1).

Источники (все локальные, уже скачаны E1):
  data/anatlas_neuron_positions.txt - РЕАЛЬНЫЕ 3D-координаты сом (WormNeuroAtlas / Anatlas)
  data/herm_full_edgelist.csv       - связи (Source,Target,Weight,Type)
  data/owmeta_cache.json            - паспорта клеток (медиатор - для цвета)

Выход:
  out/connectome_whole.json - весь червь: нейроны с координатами + все хим. связи
  out/connectome_ava.json   - окрестность AVA (2 хопа) - компактное тело
  out/e5_report.md          - отчёт живым прогоном

Формат (его читает хук ai_ui form='graph:<файл>'):
  {"format":"connectome-graph/1",
   "meta":{...},
   "nodes":[{"id":str,"x":f,"y":f,"z":f,"w":f,"c":[r,g,b],"nt":str}, ...],
   "edges":[{"a":int,"b":int,"w":f}, ...]}      # a/b - индексы в nodes
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

# цвета по медиатору (приглушённые, чтобы «нить» читалась на тёмном фоне)
NT_COLORS = {
    "Acetylcholine": (0.36, 0.56, 1.00),
    "Glutamate": (0.38, 0.87, 0.60),
    "GABA": (0.96, 0.74, 0.20),
    "Dopamine": (1.00, 0.60, 0.28),
    "Serotonin": (1.00, 0.42, 0.52),
    "Octopamine": (0.72, 0.50, 0.94),
    "Tyramine": (0.60, 0.72, 0.98),
}
NT_DEFAULT = (0.62, 0.66, 0.72)


def read_positions(path: Path):
    """anatlas_neuron_positions.txt: строка '#...' - имена, далее тройки X Y Z."""
    txt = io.open(path, encoding="utf-8", errors="replace").read().split("\n")
    names, rows = [], []
    for line in txt:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            names = s.lstrip("#").split()
            continue
        parts = [p for p in s.replace(",", "\t").split("\t") if p.strip()]
        if len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    n = min(len(names), len(rows))
    return {names[i]: rows[i] for i in range(n)}


def read_edges(path: Path):
    """herm_full_edgelist.csv -> (список (src,dst,w) химических связей, число electrical)."""
    chem, elec = [], 0
    with io.open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            key = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            s, t = key.get("source", ""), key.get("target", "")
            if not s or not t:
                continue
            try:
                w = float(key.get("weight", 1) or 1)
            except ValueError:
                w = 1.0
            if "electrical" in key.get("type", "").lower():
                elec += 1
                continue
            chem.append((s, t, w))
    return chem, elec


def read_nt(path: Path):
    """owmeta_cache.json -> {клетка: медиатор} (структура: [uri,[класс],[рец],[медиатор],..])."""
    out = {}
    try:
        raw = json.loads(io.open(path, encoding="utf-8").read())
    except Exception:
        return out
    if not isinstance(raw, dict):
        return out
    table = raw
    if "neuron_info" in raw or "muscle_info" in raw:   # формат owmeta-дампа
        table = {}
        for key in ("neuron_info", "muscle_info"):
            if isinstance(raw.get(key), dict):
                table.update(raw[key])
    for cell, val in table.items():
        if isinstance(val, list) and len(val) > 3:
            nt = val[3]
            if isinstance(nt, list):
                nt = nt[0] if nt else ""
            nt = str(nt or "").split(":")[-1].strip()
            if nt:
                out[cell] = nt
    return out


def build(nodes_pos, chem, nt_map, keep, hops=None, center=None, min_w=1.0, approx=None):
    """Собрать граф: keep - множество имён клеток; hops - окрестность center."""
    if hops is not None:
        adj = {}
        for s, t, w in chem:
            if w < min_w:
                continue
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
        keep = {center} if center else set()
        front = set(keep)
        for _ in range(int(hops)):
            nxt = set()
            for c in front:
                nxt |= adj.get(c, set())
            keep |= nxt
            front = nxt
    names = [n for n in nodes_pos if n in keep]
    names.sort()
    idx = {n: i for i, n in enumerate(names)}
    inw = {n: 0.0 for n in names}
    outw = {n: 0.0 for n in names}
    edges = []
    for s, t, w in chem:
        if s in idx and t in idx and w >= min_w:
            edges.append({"a": idx[s], "b": idx[t], "w": float(w)})
            inw[t] += w
            outw[s] += w
    xs = [nodes_pos[n][0] for n in names]
    ys = [nodes_pos[n][1] for n in names]
    zs = [nodes_pos[n][2] for n in names]
    ctr = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
    k = 2.0 / span                                     # -> тело в [-1,1]
    nodes = []
    for n in names:
        x, y, z = nodes_pos[n]
        deg = inw[n] + outw[n]
        nt = nt_map.get(n, "")
        c = NT_COLORS.get(nt, NT_DEFAULT)
        node = {
            "id": n,
            "x": round((x - ctr[0]) * k, 6),
            "y": round((y - ctr[1]) * k, 6),
            "z": round((z - ctr[2]) * k, 6),
            "w": round(deg, 4),
            "c": [round(v, 4) for v in c],
            "nt": nt,
        }
        if approx and n in approx:
            node["approx"] = 1          # координата приблизительная (мышца: центр партнёров)
        nodes.append(node)
    meta = {
        "source": "Anatlas soma positions (WormNeuroAtlas) + Cook2019 herm_full_edgelist",
        "nodes": len(nodes), "edges": len(edges),
        "chemical_weight": round(sum(e["w"] for e in edges), 1),
        "unit": "normalized [-1,1] (real relative geometry)",
        "center": center or "", "hops": hops, "min_w": min_w,
        "nt_palette": {k2: "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in c2)
                       for k2, c2 in NT_COLORS.items()},
    }
    return {"format": "connectome-graph/1", "meta": meta, "nodes": nodes, "edges": edges}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pos = read_positions(DATA / "anatlas_neuron_positions.txt")
    chem, elec = read_edges(DATA / "herm_full_edgelist.csv")
    nt_map = read_nt(DATA / "owmeta_cache.json")
    print("позиции сом: %d | хим. связей: %d | electrical строк: %d | медиаторов: %d"
          % (len(pos), len(chem), elec, len(nt_map)))

    in_pos = set(pos)
    touched = {s for s, _, _ in chem} | {t for _, t, _ in chem}
    print("в связях: %d | из них с координатами: %d | без координат (мышцы/органы): %d"
          % (len(touched), len(touched & in_pos), len(touched - in_pos)))

    OUT.mkdir(parents=True, exist_ok=True)
    whole = build(pos, chem, nt_map, keep=in_pos, min_w=1.0)
    (OUT / "connectome_whole.json").write_text(
        json.dumps(whole, ensure_ascii=False), encoding="utf-8")
    print("whole: nodes=%d edges=%d weight=%.0f"
          % (whole["meta"]["nodes"], whole["meta"]["edges"], whole["meta"]["chemical_weight"]))

    # мышцы/органы: координат сом у них нет - ставим в центр их химических партнёров
    rough = {}
    for n in sorted(touched - in_pos):
        ps = [pos[s] for s, tt, _ in chem if tt == n and s in pos]
        ps += [pos[tt] for s, tt, _ in chem if s == n and tt in pos]
        if ps:
            rough[n] = [sum(c[i] for c in ps) / len(ps) for i in range(3)]
    full = build({**pos, **rough}, chem, nt_map, keep=touched, min_w=1.0, approx=set(rough))
    (OUT / "connectome_full.json").write_text(
        json.dumps(full, ensure_ascii=False), encoding="utf-8")
    print("full: nodes=%d (из них приблизительных %d) edges=%d weight=%.0f"
          % (full["meta"]["nodes"], len(rough), full["meta"]["edges"],
             full["meta"]["chemical_weight"]))

    ava = build(pos, chem, nt_map, keep=None, hops=2, center="AVAL", min_w=3.0)
    (OUT / "connectome_ava.json").write_text(
        json.dumps(ava, ensure_ascii=False), encoding="utf-8")
    print("ava(2 хопа, w>=3): nodes=%d edges=%d"
          % (ava["meta"]["nodes"], ava["meta"]["edges"]))

    # отчёт
    hub = sorted(whole["nodes"], key=lambda n: -n["w"])[:8]
    rep = ["# E5 - тело из данных (реальный коннектом)", "",
           "Формат: `connectome-graph/1` - узлы с РЕАЛЬНЫМИ координатами сом + рёбра.",
           "",
           "- источники координат: Anatlas soma positions (пакет WormNeuroAtlas, `anatlas_neuron_positions.txt`)",
           "- связи: `herm_full_edgelist.csv` (Cook 2019 гермафродит), медиаторы: `owmeta_cache.json`",
           "- позиции нормированы в [-1,1] при сохранении реальной геометрии (не пружинная раскладка!)",
           "",
           "## Цифры прогона", "",
           "| файл | узлы | рёбра | вес |", "|---|---|---|---|",
           "| connectome_whole.json | %d | %d | %.0f |" % (
               whole["meta"]["nodes"], whole["meta"]["edges"], whole["meta"]["chemical_weight"]),
           "| connectome_ava.json (2 хопа AVA, w>=3) | %d | %d | %.0f |" % (
               ava["meta"]["nodes"], ava["meta"]["edges"],
               sum(e["w"] for e in ava["edges"])),
           "",
           "## Хабы по суммарному весу связей", ""]
    for n in hub:
        rep.append("- %s - вес %.0f, медиатор %s" % (n["id"], n["w"], n["nt"] or "?"))
    rep += ["", "Цвет узла кодирует медиатор (палитра в meta.nt_palette).", ""]
    (OUT / "e5_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("отчёт: out/e5_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
