# -*- coding: utf-8 -*-
"""Регрессия E6/E7: тела из СВОИХ данных (шина, память) и тело МУХИ (FlyWire).

Что защищаем (каждая проверка - конкретный риск):
  * детерминизм раскладки: seed_of берётся из md5, а не из hash() - иначе форма
    тела меняется от запуска к запуску (проверяется в ДВУХ процессах с разной
    PYTHONHASHSEED - это единственный честный способ поймать соль hash());
  * вес рёбер памяти >= 1: читатель по умолчанию прячет рёбра с весом < edge_min,
    и половина структуры исчезала бы молча;
  * write_graph/write_activity: битые индексы и петли отбрасываются со счётчиком,
    короткие кадры добиваются, значения обрезаются по пределам;
  * сшивка активности с телом: у файла импульса ids обязаны совпадать с узлами -
    иначе тело мигает «не теми» нейронами и никто этого не замечает;
  * у импульса мухи в meta есть КОНТРОЛЬ без стимула (сеть обязана молчать).

Запуск (без pytest, только stdlib):
    python tests/test_own_graphs.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from clab import graphbuild as gb                                  # noqa: E402
import e6_own_graphs as e6                                         # noqa: E402

AIUI = os.environ.get("AIUI_ROOT") or r"D:\Documents\Projects\ai_ui"
BODIES = os.path.join(AIUI, "data", "bodies")
OUT = os.path.join(ROOT, "out")
TMP = os.path.join(ROOT, "out", "_test")


class TestGraphbuild(unittest.TestCase):
    def test_seed_is_not_python_hash(self):
        """Одна и та же раскладка в разных процессах: hash() здесь недопустим."""
        code = ("import sys; sys.path.insert(0, %r); from clab.graphbuild import layout_atom, seed_of;"
                "print('%%s|%%s' %% ([round(v, 6) for v in layout_atom(['b','a','c'])[1][1:]], round(seed_of('x'), 9)))"
                % ROOT)
        outs = []
        for salt in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=salt)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
            outs.append(r.stdout.decode("utf-8", "replace").strip())
        self.assertEqual(outs[0], outs[1], "раскладка зависит от PYTHONHASHSEED - форма тела поедет")

    def test_fit_keeps_proportions(self):
        pts = gb.fit([(0, 0, 0), (1, 0, 0), (0, 0.5, 0)], span=2.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.assertAlmostEqual(max(xs) - min(xs), 2.0, places=6)
        self.assertAlmostEqual((max(xs) - min(xs)) / (max(ys) - min(ys)), 2.0, places=6)
        # fit центрует ГАБАРИТ (мин+макс = 0), а не среднее - так пропорции честнее
        self.assertAlmostEqual(max(xs) + min(xs), 0.0, places=6)

    def test_spiral_jitter_breaks_the_line(self):
        """Без дрожания узлы ложатся на кривую и слипаются в шнур."""
        line = gb.layout_spiral(40, turns=2.0)
        jit = gb.layout_spiral(40, turns=2.0, jitter=0.2)
        self.assertEqual([tuple(round(v, 9) for v in p) for p in line],
                         [tuple(round(v, 9) for v in p) for p in gb.layout_spiral(40, turns=2.0)],
                         "раскладка должна быть повторяемой")
        self.assertNotEqual([tuple(round(v, 6) for v in p) for p in line],
                            [tuple(round(v, 6) for v in p) for p in jit])

    def test_write_graph_drops_bad_edges(self):
        os.makedirs(TMP, exist_ok=True)
        nodes = [gb.node("a", (0, 0, 0)), gb.node("b", (1, 1, 1))]
        res = gb.write_graph(os.path.join(TMP, "t.json"), nodes,
                             [{"a": 0, "b": 1, "w": 2}, {"a": 0, "b": 0, "w": 1}, {"a": 0, "b": 9, "w": 1}])
        doc = json.loads(open(res["path"], encoding="utf-8").read())
        self.assertEqual(doc["format"], "connectome-graph/1")
        self.assertEqual(len(doc["edges"]), 1)
        self.assertEqual(doc["meta"]["dropped_edges"], 2)
        self.assertEqual(doc["meta"]["nodes"], 2)

    def test_write_graph_refuses_over_limit(self):
        os.makedirs(TMP, exist_ok=True)
        with self.assertRaises(ValueError):
            gb.write_graph(os.path.join(TMP, "big.json"),
                           [gb.node(str(i), (0, 0, 0)) for i in range(gb.LIMIT_NODES + 1)], [])

    def test_write_activity_pads_and_clamps(self):
        os.makedirs(TMP, exist_ok=True)
        res = gb.write_activity(os.path.join(TMP, "a.json"), 0.1, ["x", "y", "z"],
                                [[0.5], [2.0, -1.0, 0.3]], drive=[5.0])
        doc = json.loads(open(res["path"], encoding="utf-8").read())
        self.assertEqual(doc["format"], "connectome-activity/1")
        self.assertEqual(doc["frames"], [[0.5, 0.0, 0.0], [1.0, 0.0, 0.3]])
        self.assertEqual(doc["drive"], [1.0, 0.0])
        with self.assertRaises(ValueError):
            gb.write_activity(os.path.join(TMP, "b.json"), 0.1, [], [])


@unittest.skipUnless(os.path.isdir(os.path.join(AIUI, "data", "signal_bus")), "нет ai_ui")
class TestOwnGraphs(unittest.TestCase):
    """Свои тела: шина сигналов и память. Считается на живых данных, быстро."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.nodes, cls.edges, cls.act = e6.build_bus(Path(AIUI))
        cls.mnodes, cls.medges, cls.mact = e6.build_memory(Path(AIUI))

    def test_bus_edges_match_boxes(self):
        msgs = 0
        for name in os.listdir(os.path.join(AIUI, "data", "signal_bus")):
            if name.endswith(".jsonl"):
                with open(os.path.join(AIUI, "data", "signal_bus", name), encoding="utf-8") as fh:
                    msgs += sum(1 for l in fh if l.strip())
        self.assertEqual(self.act["meta"]["messages"], msgs, "шина посчитана не по всем ящикам")
        self.assertTrue(all(int(n["w"]) >= int(n.get("got", 0)) for n in self.nodes))
        self.assertEqual(len(self.nodes), len({n["id"] for n in self.nodes}))
        for e in self.edges:
            self.assertTrue(0 <= e["a"] < len(self.nodes) and 0 <= e["b"] < len(self.nodes))

    def test_bus_activity_ids_match_nodes(self):
        self.assertEqual(self.act["ids"], [n["id"] for n in self.nodes])
        self.assertEqual(len(self.act["frames"]), len(self.act["drive"]))
        self.assertTrue(all(-1.0 <= d <= 1.0 for d in self.act["drive"]))
        self.assertTrue(any(any(v > 0 for v in row) for row in self.act["frames"]),
                        "импульс шины пуст - сообщений не видно")

    def test_memory_weights_visible_by_default(self):
        """Вес < 1 читатель прячет по умолчанию (edge_min=1) - структура исчезла бы."""
        self.assertTrue(self.medges, "у памяти не собралось ни одной связи")
        self.assertGreaterEqual(min(e["w"] for e in self.medges), 1.0)
        self.assertGreater(len(self.mnodes), 200)
        self.assertEqual(len({n["id"] for n in self.mnodes}), len(self.mnodes), "дубли id записей")

    def test_memory_activity_is_chronology(self):
        self.assertEqual(self.mact["ids"], [n["id"] for n in self.mnodes])
        lit = [sum(1 for v in row if v > 0) for row in self.mact["frames"]]
        self.assertEqual(set(lit), {1}, "в хронологии должен вспыхивать ровно один узел за кадр")


class TestBodiesOnDisk(unittest.TestCase):
    """Уже собранные тела в ai_ui/data/bodies: формат, ids, пределы, сшивка."""

    NAMES = ("bus", "memory", "fly_central")

    def _load(self, name):
        g = os.path.join(BODIES, "%s.json" % name)
        a = os.path.join(BODIES, "%s.activity.json" % name)
        if not (os.path.isfile(g) and os.path.isfile(a)):
            self.skipTest("тела %s нет на диске" % name)
        return (json.loads(open(g, encoding="utf-8").read()),
                json.loads(open(a, encoding="utf-8").read()))

    def test_format_and_limits(self):
        for name in self.NAMES:
            g, a = self._load(name)
            self.assertEqual(g["format"], "connectome-graph/1")
            self.assertEqual(a["format"], "connectome-activity/1")
            self.assertLessEqual(len(g["nodes"]), gb.LIMIT_NODES)
            self.assertLessEqual(len(g["edges"]), gb.LIMIT_EDGES)
            self.assertLessEqual(len(a["frames"]), gb.LIMIT_FRAMES)
            self.assertLess(os.path.getsize(os.path.join(BODIES, "%s.json" % name)), gb.LIMIT_BYTES)
            for e in g["edges"]:
                self.assertTrue(0 <= e["a"] < len(g["nodes"]) and 0 <= e["b"] < len(g["nodes"]))

    def test_activity_ids_stitched_to_nodes(self):
        """Главный молчаливый риск: ids импульса не совпали с узлами."""
        for name in self.NAMES:
            g, a = self._load(name)
            self.assertEqual(a["ids"], [n["id"] for n in g["nodes"]],
                             "%s: импульс не сшит с телом" % name)
            self.assertEqual(len(a["frames"][0]), len(g["nodes"]))

    def test_fly_has_honest_control(self):
        """Импульс мухи обязан нести контроль без стимула: иначе это не импульс."""
        g, a = self._load("fly_central")
        self.assertIn("control_no_stim_max", a["meta"])
        self.assertEqual(a["meta"]["control_no_stim_max"], 0.0,
                         "контроль показал возбуждение без стимула - сеть самовозбуждается")
        self.assertGreater(a["meta"]["ignited_pct"], 1.0)
        self.assertIn("v783", g["meta"]["source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
