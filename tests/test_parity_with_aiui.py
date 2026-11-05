"""Паритет двух копий гейта: clab/gate.py  <->  ai_ui/app/agents/context_gate.py.

Зачем: копии уже один раз разъехались (ключ привыкания, дефолт numeric_kinds).
Тест сравнивает ПОВЕДЕНИЕ, а не текст, поэтому формулировки комментариев ему не мешают.

Тест ОПЦИОНАЛЬНЫЙ: если ai_ui нет на диске - SKIP, а не падение; лаборатория не
зависит от харнесса. Путь задаётся переменной окружения AIUI_PATH.

Запуск: python tests/test_parity_with_aiui.py
"""

import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDIR = os.environ.get("AIUI_PATH", r"D:\Documents\Projects\ai_ui")
sys.path.insert(0, ROOT)
sys.path.insert(0, AIDIR)

from clab import gate as CLAB                                        # noqa: E402

try:
    from app.agents import context_gate as AIUI                      # noqa: E402
except Exception as exc:                                             # pragma: no cover
    AIUI = None
    SKIP_REASON = "ai_ui не найден (%s): %s" % (AIDIR, exc)

SCENARIOS = {
    "шум со счётчиками": ["heartbeat seq=%d ok" % i for i in range(30)],
    "данные с разными числами": ["row %d: value=%d" % (i, i * 7919) for i in range(40)],
    "эскалация таймаутов": ["run_python timeout after %ds" % t for t in
                            (1, 2, 3, 4, 5, 6, 120, 300, 600, 900, 1200, 1800)],
    "буквальные повторы": ["error: disk full"] * 5,
    "лог с метками времени": ["2026-09-20 12:00:%02d INFO starting tool call" % (i % 60)
                              for i in range(30)],
    "смешанный лог": (["error: connection refused to 127.0.0.1:800%d" % p for p in range(3)]
                      + ["run_python timeout after 120s", "info: done"] * 4),
    "пусто": [],
    "одна длинная строка": ["x" * 5000],
}
SIG_LINES = ["error: timeout 120s", "error: timeout 300s", "heartbeat seq=7 ok",
             "INFO done", "row 3: value=7919", "row 4: value=23757",
             "warning: retry 2 of 5", "a" * 300]


class TestParityWithAiUi(unittest.TestCase):
    def setUp(self):
        if AIUI is None:
            self.skipTest(SKIP_REASON)

    def test_defaults_are_equal(self):
        """Одинаковые дефолты: иначе копии ведут себя по-разному без спроса."""
        ps = inspect.signature(CLAB.ContextGate.__init__).parameters
        pa = inspect.signature(AIUI.ContextGate.__init__).parameters
        self.assertEqual(sorted(ps), sorted(pa), "разный набор параметров")
        for name in ps:
            if name == "self":
                continue
            self.assertEqual(ps[name].default, pa[name].default,
                             "разный дефолт параметра %s" % name)

    def test_signature_matches(self):
        for line in SIG_LINES:
            for keep in (False, True):
                self.assertEqual(CLAB.signature(line, keep_numbers=keep),
                                 AIUI.signature(line, keep_numbers=keep),
                                 "подпись разошлась: %r (keep=%s)" % (line[:40], keep))

    def test_gate_behaviour_matches(self):
        for title, lines in SCENARIOS.items():
            gc = CLAB.ContextGate(max_chars=2000, max_lines=12, head_lines=1,
                                  keep_kinds=("error",), numeric_kinds=("error", "warn", "info"))
            ga = AIUI.ContextGate(max_chars=2000, max_lines=12, head_lines=1,
                                  keep_kinds=("error",), numeric_kinds=("error", "warn", "info"))
            b, c = gc.gate(lines), ga.gate(lines)
            for field in ("text", "kept", "suppressed", "chars_in", "chars_out",
                          "hab_drops", "repeats", "dropped_kinds", "escalations"):
                self.assertEqual(getattr(b, field), getattr(c, field),
                                 "%s: поле %s разошлось" % (title, field))
            self.assertEqual(CLAB.apply_policy(b), AIUI.apply_policy(c),
                             "%s: решение «резать или нет» разошлось" % title)

    def test_guess_kind_matches(self):
        for line in SIG_LINES:
            self.assertEqual(CLAB.guess_kind(line), AIUI.guess_kind(line), line[:40])


if __name__ == "__main__":
    unittest.main(verbosity=2)
