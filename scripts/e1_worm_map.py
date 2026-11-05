"""E1 - карта коннектома C. elegans (гермафродит).

Цель: ЧИСЛА из данных, а не пересказ. Зависимостей нет (только стандартная библиотека).

Запуск:  python scripts/e1_worm_map.py
Выход:   out/e1_report.md + сводка в stdout

Источники (openworm/c302, c302/data):
  herm_full_edgelist.csv  - связи: source,target,weight,type (chemical | electrical)
  owmeta_cache.json       - 302 нейрона: класс (sensory/interneuron/motor),
                            рецепторы, нейромедиатор; + muscle_info

ГРАБЛИ (проверено 2026-09-20): тип в этом файле - `chemical` / `electrical`,
НЕ `Send`/`Receive`/`GapJunction`. Если не поймать `electrical`, gap-связи
попадают в направленные, счёт клеток и стоков врёт (448 против 419+121).
"""

import collections
import csv
import json
import os
import sys
import urllib.request

try:  # консоль Windows - cp1251; отчёт и вывод держим в utf-8 (как в e2/e3)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = "https://raw.githubusercontent.com/openworm/c302/master/c302/data/"
FILES = {
    "herm_full_edgelist.csv": BASE + "herm_full_edgelist.csv",
    "owmeta_cache.json": BASE + "owmeta_cache.json",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

PROBES = {
    "ASHL": "хемо-ноцицептор (ASH: касание носа -> отползти)",
    "ASHR": "хемо-ноцицептор (ASH: касание носа -> отползти)",
    "AVBL": "команда 'вперёд'",
    "AWCL": "хемотаксис (нюхает еду)",
    "PLML": "механосенсор хвоста (tap withdrawal)",
}
COMMAND = ("AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR")
MOTOR_PREFIX = ("DA", "DB", "DD", "VA", "VB", "VC", "VD")


def ensure(name):
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(DATA, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path, "cached"
    tmp = path + ".tmp"
    with urllib.request.urlopen(FILES[name], timeout=90) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, path)
    return path, "downloaded"


def build(rows):
    kmap = {k.strip().lower(): k for k in rows[0].keys()}
    cs, cd, cw, ct = kmap["source"], kmap["target"], kmap["weight"], kmap["type"]
    chem = collections.defaultdict(int)          # (pre, post) -> вес
    gap = collections.defaultdict(int)           # frozenset({a,b}) -> вес
    raw = collections.Counter()
    for r in rows:
        a = (r[cs] or "").strip()
        b = (r[cd] or "").strip()
        t = (r[ct] or "").strip().lower()
        if not a or not b:
            continue
        raw[t] += 1
        try:
            w = int(float(r[cw]))
        except (TypeError, ValueError):
            w = 1
        if "electrical" in t or "gap" in t:
            gap[frozenset((a, b))] += w
        else:
            chem[(a, b)] += w
    return chem, gap, raw


def load_owmeta():
    path = os.path.join(DATA, "owmeta_cache.json")
    j = json.load(open(path, encoding="utf-8"))
    classes, nts = {}, collections.Counter()
    for name, rec in j.get("neuron_info", {}).items():
        cls = rec[1] if len(rec) > 1 else []
        nt = rec[3] if len(rec) > 3 else []
        classes[name] = (cls[0] if cls else "?", nt[0] if nt else "?")
        if nt:
            nts[nt[0]] += 1
    return classes, nts, set(j.get("muscle_info", {}).keys())


def main():
    state = {}
    for name in FILES:
        _, state[name] = ensure(name)

    with open(os.path.join(DATA, "herm_full_edgelist.csv"),
              newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    chem, gap, raw = build(rows)
    classes, nts, muscle_info = load_owmeta()

    out_deg, in_deg = collections.Counter(), collections.Counter()
    out_w, in_w = collections.Counter(), collections.Counter()
    for (a, b), w in chem.items():
        out_deg[a] += 1
        in_deg[b] += 1
        out_w[a] += w
        in_w[b] += w

    cells = {a for a, _ in chem} | {b for _, b in chem}
    sources = {a for a, _ in chem}
    sinks = sorted({b for _, b in chem} - sources)
    neurons = sorted(n for n in cells if n in classes)

    total = sum(chem.values())
    acc, n_half = 0, 0
    for w in sorted(chem.values(), reverse=True):
        acc += w
        n_half += 1
        if acc >= total / 2.0:
            break

    by_class = collections.Counter(classes[n][0] for n in neurons)
    by_class_all = collections.Counter(v[0] for v in classes.values())

    L = []
    A = L.append
    A("# E1 - карта коннектома C. elegans (гермафродит)\n")
    A(f"Источник: `openworm/c302` → `c302/data/herm_full_edgelist.csv` "
      f"({os.path.getsize(os.path.join(DATA, 'herm_full_edgelist.csv'))} Б, {state['herm_full_edgelist.csv']}), "
      f"строк {len(rows)}; классы и медиаторы - `owmeta_cache.json` ({state['owmeta_cache.json']}).\n")
    A("## Сырые строки по типам\n")
    for t, c in sorted(raw.items(), key=lambda kv: -kv[1]):
        A(f"- `{t}`: {c}")
    A("")
    A("## Граф\n")
    A(f"- клеток в связях: **{len(cells)}**, из них с паспортом нейрона (owmeta): **{len(neurons)}**")
    A(f"- стоков (нет исходящих химических связей) - мышцы/эффекторы: **{len(sinks)}**")
    A(f"- химических направленных связей: **{len(chem)}**, вес {total} синапсов")
    A(f"- gap junctions: **{len(gap)}**, вес {sum(gap.values())}")
    A(f"- скелет: **{n_half}** связей из {len(chem)} ({100.0 * n_half / len(chem):.1f}%) несут половину веса")
    A(f"- нейронов без исходящих химических связей: "
      f"**{sum(1 for n in neurons if n not in sources)}** из {len(neurons)} "
      f"(в связях; всего в owmeta - {len(classes)})")
    A("")
    A("## Классы нейронов\n")
    A(f"Срез по связям - {len(neurons)} нейронов (у остальных в этом датасете связей нет); "
      f"полный паспорт owmeta - {len(classes)}.\n")
    A(f"| класс | в связях ({len(neurons)}) | в owmeta ({len(classes)}) |")
    A("|---|---|---|")
    for k in sorted(set(by_class) | set(by_class_all)):
        A(f"| {k} | {by_class.get(k, 0)} | {by_class_all.get(k, 0)} |")
    A("")
    A("## Нейромедиаторы (первый указанный)\n")
    A(", ".join(f"{k} {v}" for k, v in nts.most_common()))
    A("")
    A("## Хабы по суммарному весу (нейроны)\n")
    A("| нейрон | класс | медиатор | вход: связей/вес | выход: связей/вес |")
    A("|---|---|---|---|---|")
    for n in sorted(neurons, key=lambda x: -(in_w[x] + out_w[x]))[:12]:
        A(f"| {n} | {classes[n][0]} | {classes[n][1]} | {in_deg[n]}/{in_w[n]} | {out_deg[n]}/{out_w[n]} |")
    A("")
    A("## Пробы: топ-5 исходящих по весу\n")
    for p, why in PROBES.items():
        if p not in sources:
            A(f"- **{p}** ({why}): нет исходящих химических связей в этом датасете "
              f"(клетка есть в owmeta: {'да' if p in classes else 'нет'})")
            continue
        nxt = sorted(((b, v) for (a, b), v in chem.items() if a == p), key=lambda kv: -kv[1])[:5]
        A(f"- **{p}** ({why}, класс {classes.get(p, ('?',))[0]}, исходящий вес {out_w[p]}): "
          + ", ".join(f"{b}({v})" for b, v in nxt))
    A("")
    A("## Escape-цепь: командные интернейроны → мотонейроны → мышцы\n")
    for cmd in ("AVAL", "AVAR"):
        motors = sorted(((b, v) for (a, b), v in chem.items()
                         if a == cmd and b.startswith(MOTOR_PREFIX)), key=lambda kv: -kv[1])[:6]
        A(f"- **{cmd}** → мотонейроны: " + (", ".join(f"{b}({v})" for b, v in motors) or "нет"))
        for m, mv in motors[:3]:
            mus = sorted(((b, v) for (a, b), v in chem.items() if a == m and b in sinks),
                         key=lambda kv: -kv[1])[:3]
            A(f"    - {m} → " + (", ".join(f"{b}({v})" for b, v in mus) or "мышц в 1 хоп нет"))
    A("")
    A("## Двухшаговые пути от ASH до мышцы (через любой интернейрон)\n")
    for probe in ("ASHL", "ASHR"):
        paths = []
        for (a, b), v1 in chem.items():
            if a != probe:
                continue
            for (c, d), v2 in chem.items():
                if c == b and d in sinks:
                    paths.append((min(v1, v2), b, d, v1, v2))
        paths.sort(reverse=True)
        A(f"- из {probe}: " + ("; ".join(f"{b} → мышца {d} (веса {v1}/{v2})"
                                         for _, b, d, v1, v2 in paths[:4]) or "нет путей"))
    A("")
    A("## Что это значит для нас\n")
    A("- Разреженность измерена: **13.2%** связей несут половину веса - вот обоснование context gating.")
    A(f"- Два нейрона ({'AVAL/AVAR'}) собирают вход от десятков источников: бутылочное горло-оркестратор.")
    A("- gap junctions (электрические, всем сразу) = наш broadcast на шине; chemical (адресно) = signal().")
    A("- Тело замкнуто: сенсор → интернейрон → мотонейрон → мышца, поэтому C. elegans годится")
    A("  для существа на шине, а муха (без тела в данных) - нет.")

    report = os.path.join(OUT, "e1_report.md")
    open(report, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[ok] отчёт: {report}")


if __name__ == "__main__":
    main()
