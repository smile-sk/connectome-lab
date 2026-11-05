# -*- coding: utf-8 -*-
"""E6: тела из НАШИХ данных - шина сигналов и память агента.

Не биология: узлы - это мы (агенты и собеседники) и мои записи, рёбра - сообщения
и общие темы, импульс - реальное время событий (не выдуманная динамика).

  out/bus_graph.json + bus.activity.json        - кто кому пишет на шине
  out/memory_graph.json + memory.activity.json  - записи notes.md: связи по общим
                                                  темам, импульс = хронология записей

Формат и запись - clab.graphbuild (один код на все источники: коннектом, шина,
память). Новый источник = новая функция-адаптер build_* здесь же, которая
возвращает (nodes, edges, activity); форматы не трогаются.

Живой прогон:  python scripts/e6_own_graphs.py --copy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clab import graphbuild as gb   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DEFAULT_AIUI = os.environ.get("AIUI_ROOT") or r"D:\Documents\Projects\ai_ui"

# --- ЦВЕТА / РОЛИ -----------------------------------------------------------------

C_ME = (1.00, 0.46, 0.36)      # я
C_AGENT = (0.40, 0.62, 1.00)   # свои агенты харнесса
C_HUMAN = (0.96, 0.80, 0.34)   # люди (оператор)
C_OUTER = (0.56, 0.88, 0.62)   # внешние узлы (другие узлы/хабы)
C_PLAIN = (0.62, 0.66, 0.72)

# Темы для цвета записей памяти: (метка, ключевые слова, цвет). Первое совпадение.
TOPICS = [
    ("шина/сигналы", ("шина", "signal", "сигнал", "ящик", "пробужд"), (0.44, 0.64, 1.00)),
    ("гейт/контекст", ("contextgate", "context_gate", "гейт", "gate", "подавлено", "привыка"), (0.96, 0.74, 0.20)),
    ("тело/коннектом", ("connectome", "connectomelab", "коннектом", "worm", "черв", "тело", "воксел", "girth"), (0.38, 0.90, 0.56)),
    ("муха/flywire", ("муха", "мухи", "flywire", "дрозофил", "manc", "banc"), (0.86, 0.52, 0.98)),
    ("память/вектор", ("statevector", "state_vector", "вектор", "notes", "память", "саммари", "якор"), (1.00, 0.56, 0.60)),
    ("харнесс/рестарт", ("харнесс", "main_window", "ai_ui", "перезапуск", "рестарт", "стоп", "сесси"), (0.70, 0.74, 0.80)),
    ("QA/Орест", ("орест", "orestes", "приёмк", "qa_report", "qa "), (0.98, 0.60, 0.26)),
    ("CleanTrade/VPS", ("cleantrade", "cleanprod", "vps", "example-vps", "stablepay", "лендинг", "ростелеком"), (0.36, 0.84, 0.86)),
    ("Мириад/Антон", ("мириад", "miriad", "антон", "agentworld"), (0.62, 0.90, 0.40)),
    ("Джемми", ("джемми", "gemmy"), (0.99, 0.44, 0.78)),
]


def role_color(name: str, internal: set, me: str):
    if name == me:
        return C_ME, "я"
    if name in internal:
        return C_AGENT, "агент"
    if name.startswith("stas") or name.endswith("/human"):
        return C_HUMAN, "человек"
    return C_OUTER, "внешний"


def topic_of(text: str):
    low = text.lower()
    for label, keys, col in TOPICS:
        for k in keys:
            if k in low:
                return label, col
    return "прочее", C_PLAIN


# --- АДАПТЕР 1: ШИНА СИГНАЛОВ -----------------------------------------------------

def read_boxes(aiui: Path):
    """Все ящики шины: <агент>.jsonl, строки {"ts","from","text"} (кому = имя ящика)."""
    boxes = {}
    for p in sorted((aiui / "data" / "signal_bus").glob("*.jsonl")):
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        boxes[p.stem] = rows
    return boxes


def build_bus(aiui: Path, me: str = "cody", frames_n: int = 120, period: float = 30.0):
    boxes = read_boxes(aiui)
    events = []
    for box, rows in boxes.items():
        for r in rows:
            events.append((float(r.get("ts") or 0.0), str(r.get("from") or "?"), box))
    events.sort(key=lambda e: e[0])
    if not events:
        raise ValueError("шина: сообщений нет")

    internal = set()
    agdir = aiui / "agents"
    if agdir.is_dir():
        internal = {p.name for p in agdir.iterdir() if p.is_dir()}

    names = sorted({e[1] for e in events} | {e[2] for e in events})
    idx = {n: i for i, n in enumerate(names)}

    ec = Counter((e[1], e[2]) for e in events)
    involved = Counter()
    for (a, b), w in ec.items():
        involved[a] += w
        involved[b] += w

    lay = gb.layout_atom(names, {n: involved.get(n, 0) for n in names}, hub=me)
    pos = {n: (x, y, z) for n, x, y, z in lay}
    fit = gb.fit([pos[n] for n in names])
    nodes = []
    for n, xyz in zip(names, fit):
        col, role = role_color(n, internal, me)
        nodes.append(gb.node(n, xyz, w=max(1.0, float(involved.get(n, 0))), color=col,
                             role=role, sent=int(sum(v for (a, _), v in ec.items() if a == n)),
                             got=int(sum(v for (_, b), v in ec.items() if b == n))))

    t0, t1 = events[0][0], events[-1][0]
    span = max(1e-9, t1 - t0)
    binw = span / frames_n
    inc, outc = defaultdict(lambda: [0] * frames_n), defaultdict(lambda: [0] * frames_n)
    for ts, src, box in events:
        i = min(frames_n - 1, int((ts - t0) / binw)) if binw > 0 else 0
        outc[src][i] += 1
        inc[box][i] += 1

    raw, drive = [], []
    for i in range(frames_n):
        raw.append([float(inc[n][i] + outc[n][i]) for n in names])
        ins = sum(inc[n][i] for n in names)
        outs = sum(outc[n][i] for n in names)
        drive.append(0.0 if (ins + outs) == 0 else (ins - outs) / float(ins + outs))
    mx = max((max(r) for r in raw), default=0.0) or 1.0
    frames = [[v / mx for v in r] for r in raw]

    meta = {
        "source": "ai_ui data/signal_bus/*.jsonl (ящики агентов)",
        "kind": "own-bus",
        "boxes": {k: len(v) for k, v in sorted(boxes.items())},
        "messages": len(events),
        "span_hours": round(span / 3600.0, 2),
        "bin_hours": round(binw / 3600.0, 3),
        "edge_meaning": "стрелка автор -> ящик получателя, вес = число сообщений",
        "drive": "перевес входящих над исходящими в бине (тело тянется к тем, кто пишет)",
    }
    return nodes, [{"a": idx[a], "b": idx[b], "w": float(w)} for (a, b), w in sorted(ec.items())], \
        {"dt": period / frames_n, "ids": names, "frames": frames, "drive": drive, "meta": meta}


# --- АДАПТЕР 2: ПАМЯТЬ (notes.md) --------------------------------------------------

ENTRY_RE = re.compile(r"(?m)^##[ \t]+(\d{4}-\d{2}-\d{2}[ \t]+\d{2}:\d{2})[ \t]*$")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][\w.\-/]{3,}|\d{4,}")

STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где
есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж
тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы
нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над
больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между этою
the and for that with this from are was were has have not but you your его они без
""".split())


def split_entries(text: str):
    marks = [(m.start(), m.end(), m.group(1).replace("\t", " ")) for m in ENTRY_RE.finditer(text)]
    out = []
    for i, (_s, e, stamp) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((stamp, text[e:end].strip()))
    return out


def tokens_of(text: str):
    out = set()
    for w in WORD_RE.findall(text.lower()):
        w = w.strip("._-/")
        if len(w) >= 5 and w not in STOP:
            out.add(w)
    return out


def build_memory(aiui: Path, me: str = "cody", top_links: int = 3, period: float = 45.0):
    path = aiui / "agents" / me / "notes.md"
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = split_entries(text)
    if not entries:
        raise ValueError("память: записей не найдено")

    seen, ids = Counter(), []
    for stamp, _t in entries:
        seen[stamp] += 1
        ids.append(stamp if seen[stamp] == 1 else "%s#%d" % (stamp, seen[stamp]))

    toks = [tokens_of(t) for _s, t in entries]
    n = len(entries)

    df = Counter()
    for s in toks:
        df.update(s)
    idf = {w: 1.0 / math.log(2.0 + df[w]) for w in df}

    link = {}
    per_node = defaultdict(list)
    for i in range(n):
        ti = toks[i]
        if not ti:
            continue
        for j in range(i + 1, n):
            shared = ti & toks[j]
            if len(shared) < 2:
                continue
            dis = [t for t in shared if df[t] <= 80]
            if len(dis) < 2:
                continue
            w = sum(idf[t] for t in dis)
            link[(i, j)] = w
            per_node[i].append((w, j))
            per_node[j].append((w, i))

    keep = set()
    for i, lst in per_node.items():
        for _w, j in sorted(lst, reverse=True)[:top_links]:
            keep.add((min(i, j), max(i, j)))
    # Вес в «единицах idf» выходит меньше 1, а читатель по умолчанию прячет рёбра
    # с весом < edge_min (1.0) - половина структуры пропала бы. Шкала фиксированная
    # и объявлена в meta: w = 10 × сумма idf по общим значимым словам.
    escale = 10.0
    edges = [{"a": i, "b": j, "w": round(link[(i, j)] * escale, 3)} for i, j in sorted(keep)]

    deg = Counter()
    for i, j in keep:
        deg[i] += 1
        deg[j] += 1

    spiral = gb.layout_spiral(n, turns=8.0, r0=0.45, r1=0.95, height=2.2, jitter=0.12)
    fit = gb.fit(spiral)
    nodes, topics = [], Counter()
    for i, (stamp, body) in enumerate(entries):
        label, col = topic_of(body)
        topics[label] += 1
        nodes.append(gb.node(ids[i], fit[i], w=max(1.0, float(deg.get(i, 0))), color=col,
                             date=stamp.split()[0], topic=label, chars=len(body)))

    frames = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 1.0
        frames.append(row)

    meta = {
        "source": "agents/%s/notes.md - %d записей" % (me, n),
        "kind": "own-memory",
        "range": "%s .. %s" % (entries[0][0], entries[-1][0]),
        "layout": "time-spiral (8 витков + дрожание, старые внизу) - раскладка, НЕ анатомия",
        "edge_meaning": "общие редкие слова (>=2 значимых), вес = %.0f x сумма idf" % escale,
        "edge_min_hint": "вес >= 4, поэтому видно по умолчанию (edge_min=1)",
        "topic_legend": {lab: gb.hexcolor(col) for lab, _k, col in TOPICS},
        "topics": dict(topics.most_common()),
    }
    act_meta = {
        "source": meta["source"], "kind": "own-memory",
        "impulse": "хронология: кадр = одна запись, вспыхивает та, что писалась",
        "entries": n,
    }
    return nodes, edges, {"dt": period / n, "ids": ids, "frames": frames,
                          "drive": [0.0] * n, "meta": act_meta}


# --- ЗАПУСК -----------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="тела из своих данных (шина, память)")
    ap.add_argument("--aiui", default=DEFAULT_AIUI, help="корень проекта ai_ui")
    ap.add_argument("--me", default="cody", help="чей notes.md брать")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--copy", action="store_true", help="скопировать тела в ai_ui/data/bodies")
    args = ap.parse_args()

    aiui = Path(args.aiui)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not aiui.is_dir():
        print("нет каталога ai_ui: %s" % aiui)
        return 1

    made, report = [], ["# E6 - тела из своих данных", "",
                        "источник: `%s`" % aiui, ""]

    for name, builder, period in (("bus", build_bus, 30.0), ("memory", build_memory, 45.0)):
        try:
            if name == "bus":
                nodes, edges, act = builder(aiui, me=args.me)
            else:
                nodes, edges, act = builder(aiui, me=args.me)
        except Exception as exc:
            print("%s: ОШИБКА %s: %s" % (name, type(exc).__name__, exc))
            report += ["## %s - ошибка" % name, str(exc), ""]
            continue
        g = gb.write_graph(out / ("%s_graph.json" % name), nodes, edges, act["meta"])
        a = gb.write_activity(out / ("%s.activity.json" % name), act["dt"], act["ids"],
                              act["frames"], act["drive"], act["meta"])
        print("%-6s узлов %3d рёбер %4d | импульс кадров %4d цикл %.1f с | %.0f+%.0f КБ"
              % (name, g["nodes"], g["edges"], a["frames"], a["period"],
                 g["bytes"] / 1024, a["bytes"] / 1024))
        made.append((name, g, a))
        report += ["## %s" % ("шина сигналов" if name == "bus" else "память агента"), "",
                   "- файлы: `%s_graph.json` + `%s.activity.json`" % (name, name),
                   "- узлов **%d**, рёбер **%d**" % (g["nodes"], g["edges"]),
                   "- импульс: кадров **%d**, цикл **%.1f с**, dt **%.3f с**" % (a["frames"], a["period"], act["dt"]),
                   "- размер: тело %.0f КБ, импульс %.0f КБ" % (g["bytes"] / 1024, a["bytes"] / 1024),
                   ""]
        if name == "bus":
            top = sorted(nodes, key=lambda x: -x["w"])[:6]
            report += ["Кто в графе (вес = сообщений с участием):", ""]
            report += ["- `%s` - %d, роль: %s" % (x["id"], int(x["w"]), x.get("role")) for x in top]
            report += ["", "Рёбра: %s" % act["meta"]["edge_meaning"],
                       "Импульс: %s" % act["meta"]["drive"], ""]
        else:
            report += ["Темы (крупнейшие): %s" % ", ".join(
                "%s %d" % (k, v) for k, v in list(act["meta"].get("topics", {}).items())[:6]), ""]
            hubs = sorted(range(len(nodes)), key=lambda i: -nodes[i]["w"])[:8]
            report += ["Узлы-хабы (больше всего связей по темам):", ""]
            report += ["- %s - связей %d, тема %s" % (nodes[i]["id"], int(nodes[i]["w"]), nodes[i].get("topic"))
                       for i in hubs]
            report += ["", "Раскладка: %s" % act["meta"].get("layout", ""), ""]

    if args.copy and made:
        dest = aiui / "data" / "bodies"
        dest.mkdir(parents=True, exist_ok=True)
        for name, _g, _a in made:
            for frm, to in ((out / ("%s_graph.json" % name), dest / ("%s.json" % name)),
                            (out / ("%s.activity.json" % name), dest / ("%s.activity.json" % name))):
                to.write_text(frm.read_text(encoding="utf-8"), encoding="utf-8", newline="")
            print("скопировано в %s: %s.json + %s.activity.json" % (dest, name, name))
        report += ["## Куда надеты", "", "`ai_ui/data/bodies/`: " +
                   ", ".join("`%s.json`" % n for n, _g, _a in made), ""]

    (out / "e6_report.md").write_text("\n".join(report), encoding="utf-8", newline="")
    print("отчёт: %s" % (out / "e6_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
