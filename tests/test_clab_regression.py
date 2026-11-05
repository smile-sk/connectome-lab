# -*- coding: utf-8 -*-
"""Регрессионные тесты по отчёту Ореста (2026-09-20).

Каждый тест закрывает конкретную находку из
`agents/cody/qa_report_orestes_connectomelab_2026-09-20.md`:

  №1 (🟠) e1_worm_map.py падал на русской консоли (cp1251) - UnicodeEncodeError
  №2 (🟡) gate схлопывал разные ошибки, отличающиеся только числом (эскалация терялась)
  №3 (🟡) label «302» вместо 234 (классы в связях против полного owmeta)
  №4 (🟡) response_of() брал floor «info» для любого ключа → занижал отклик ошибки

Плюс защита достигнутого: редкий повтор не гасится, ошибка не гасится никогда,
идентичные строки по-прежнему схлопываются, однострочный гигант режется по line_cap.

Запуск (без pytest, только stdlib):
    python tests/test_clab_regression.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:  # консоль Windows - cp1251
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from clab import ContextGate, Habituation, apply_policy      # noqa: E402
from clab.gate import signature                                # noqa: E402

OUT = os.path.join(ROOT, "out")
SCRIPTS = os.path.join(ROOT, "scripts")

# Ровно тот набор, на котором Орест показал схлопывание: 3 порта + 2 таймаута.
OREST_LINES = [
    "2026-09-20 11:00:01 INFO starting tool call",
    "connection refused when connecting to 127.0.0.1:8000",
    "connection refused when connecting to 127.0.0.1:8001",
    "connection refused when connecting to 127.0.0.1:8002",
    "run_python timeout after 120s",
    "run_python timeout after 300s",
]
ERRS = OREST_LINES[1:]


class TestE1Console(unittest.TestCase):
    """Находка №1 - падение e1 на cp1251 + находка №3 - знаменатели в отчёте."""

    @classmethod
    def setUpClass(cls):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
        cls.r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "e1_worm_map.py")],
                               cwd=ROOT, capture_output=True, env=env, timeout=300)
        with io.open(os.path.join(OUT, "e1_report.md"), encoding="utf-8") as f:
            cls.report = f.read()

    def test_no_crash_on_windows_console(self):
        """rc=0 и вывод - валидный utf-8 со стрелкой (до фикса был трейсбек, rc=1)."""
        self.assertEqual(self.r.returncode, 0,
                         "e1 упал: " + self.r.stderr.decode("cp1251", errors="replace")[:400])
        text = self.r.stdout.decode("utf-8")           # UnicodeDecodeError = фикс сломан
        self.assertIn("→", text)

    def test_class_labels_are_honest(self):
        """Классы: срез по связям (234) и полный owmeta (302) - разными числами."""
        self.assertIn("| sensory | 64 | 66 |", self.report)
        self.assertIn("| motor | 51 | 115 |", self.report)
        self.assertIn("| interneuron | 119 | 119 |", self.report)
        self.assertIn("всего в owmeta - 302", self.report)
        self.assertNotIn("из 302", self.report.replace("всего в owmeta - 302", ""))


class TestGateEscalation(unittest.TestCase):
    """Находка №2 - эскалация ошибок больше не теряется."""

    def test_signature_numbers_matter(self):
        """Суть фикса в одной проверке: без чисел - 2 подписи, с числами - 5."""
        old = {signature(l) for l in ERRS}
        new = {signature(l, keep_numbers=True) for l in ERRS}
        self.assertEqual(len(old), 2, "старое поведение: 3 порта и 2 таймаута = 2 подписи")
        self.assertEqual(len(new), 5, "новое: каждый порт и каждый таймаут различимы")

    def test_all_errors_survive_gate(self):
        """Ровно репро Ореста: до фикса в промпт проходило 2 строки, теперь - все."""
        g = ContextGate(max_lines=8, max_chars=1200)
        b = g.gate(list(OREST_LINES))
        for needle in (":8000", ":8001", ":8002", "120s", "300s"):
            self.assertIn(needle, b.text, f"потеряна строка с {needle}")
        self.assertEqual(b.suppressed, 0)

    def test_escalation_reported(self):
        """Гейт не только сохраняет, но и сообщает: «эскалаций N»."""
        b = ContextGate(max_lines=8).gate(list(OREST_LINES))
        self.assertEqual(len(b.escalations), 2, "эскалации: порты и таймауты")
        self.assertEqual(b.summary()["escalations"], 2)

    def test_identical_lines_still_collapse(self):
        """Обратная сторона: буквально одинаковые строки по-прежнему схлопываются."""
        dup = ["error: disk full"] * 2
        b = ContextGate(max_lines=8).gate(dup)
        self.assertEqual(b.kept_lines.count("error: disk full"), 1)
        self.assertEqual(sum(b.repeats.values()), 1)
        self.assertIn("[подавлено 1 строк", b.text)

    def test_long_single_line_is_capped(self):
        """Однострочный гигант (лог одной строкой) не съедает бюджет целиком."""
        long_line = "error: " + "x" * 5000
        b = ContextGate(max_lines=8, line_cap=400).gate([long_line])
        self.assertLess(len(b.kept_lines[0]), 600)
        self.assertIn("срезано", b.kept_lines[0])

    def test_empty_input(self):
        b = ContextGate().gate([])
        self.assertEqual((b.text, b.kept, b.suppressed), ("", 0, 0))


class TestResponseOf(unittest.TestCase):
    """Находка №4 - floor по типу события, а не всегда «info»."""

    def test_error_floor_not_info_floor(self):
        """Ровно замер Ореста: одно событие (n_eff=1.0) → 0.7885, а не 0.6108.

        Считается так: floor + (1-floor)*exp(-0.55*1). При floor=0.50 → 0.7885,
        при ошибочном floor=0.08 (info) → 0.6108. Разница в 0.18 «тишины» -
        именно та, на которую ошибка казалась почти притихшей.
        """
        h = Habituation(tau_rec=120.0)
        h.observe("error:boom", kind="error", ts=0.0)
        self.assertAlmostEqual(h.response_of("error:boom"), 0.7885, places=3)
        self.assertNotAlmostEqual(h.response_of("error:boom"), 0.6108, places=3)

    def test_info_key_quieter_than_error_key(self):
        """Тот же n_eff, но info-ключ: честные 0.6108 против 0.7885 у ошибки."""
        # Два независимых потока: в одном потоке новая ошибка сбросила бы
        # привыкание соседнего ключа (dishabituation) и сравнение было бы нечестным.
        h_info, h_err = Habituation(tau_rec=120.0), Habituation(tau_rec=120.0)
        h_info.observe("info:noise", kind="info", ts=0.0)
        h_err.observe("error:boom", kind="error", ts=0.0)
        self.assertAlmostEqual(h_info.response_of("info:noise"), 0.6108, places=3)
        self.assertGreater(h_err.response_of("error:boom"), h_info.response_of("info:noise"))

    def test_dishabituation_wakes_neighbours(self):
        """Побочный эффект, о котором предупредил Орест: новый error будит шум.

        Это не баг, а биология - но пусть будет зафиксировано числом, чтобы
        при переносе в харнесс помнить про «на один ход открылся шлюз шума».
        """
        h = Habituation(tau_rec=120.0)
        for _ in range(20):
            h.observe("info:noise", kind="info", ts=0.0)
        quiet = h.response_of("info:noise")
        h.observe("error:boom", kind="error", ts=1.0)
        self.assertGreater(h.response_of("info:noise"), quiet)

    def test_unknown_key_is_zero(self):
        self.assertEqual(Habituation().response_of("нет-такого"), 0.0)


class TestHabituationStillHolds(unittest.TestCase):
    """Свойства, подтверждённые Орестом живьём, - не должны отвалиться."""

    def test_rare_repeat_passes(self):
        h = Habituation(tau_rec=120.0)
        votes = [h.observe("k", kind="info", ts=150.0 * i).emit for i in range(5)]
        self.assertEqual(votes, [True] * 5)

    def test_error_never_suppressed(self):
        h = Habituation(tau_rec=120.0, now=lambda: 0.0)
        votes = [h.observe("e", kind="error").emit for _ in range(50)]
        self.assertTrue(all(votes))

    def test_burst_flapping_is_damped(self):
        h = Habituation(tau_rec=120.0)
        votes = [h.observe("w", kind="warn", ts=2.0 * i).emit for i in range(20)]
        self.assertLess(sum(votes), 10, "флаппинг warn должен приглушаться")


class TestGateKeyScope(unittest.TestCase):
    """Ключ гейта: числа значимы везде, кроме шума (сведено с ai_ui 20.09.2026)."""

    def test_data_rows_are_not_treated_as_repeats(self):
        """40 строк расчёта с разными числами - 40 стимулов, а не повторы."""
        rows = ["row %d: value=%d" % (i, (i * 7919) % 9973) for i in range(40)]
        b = ContextGate(max_lines=8, max_chars=4000).gate(rows)
        self.assertEqual(sum(b.repeats.values()), 0, "разные числа - не повтор")
        self.assertEqual(b.hab_drops, 0, "привыкание не должно убрать данные")
        self.assertFalse(apply_policy(b), "уникальные данные нельзя отдавать урезанными")

    def test_noise_counters_still_collapse(self):
        """Шум с меняющимися счётчиками по-прежнему один и тот же стимул."""
        noise = ["heartbeat seq=%d ok" % i for i in range(30)]
        b = ContextGate(max_lines=8, max_chars=4000).gate(noise)
        self.assertGreater(sum(b.repeats.values()), 20, "счётчики шума - один стимул")
        self.assertTrue(apply_policy(b), "шум урезать можно и нужно")

    def test_escalation_survives_habituation_key(self):
        """Эскалация не должна гаснуть привыканием: 120s и 300s - разные стимулы."""
        lines = ["run_python timeout after %ds" % t_ for t_ in
                 (1, 2, 3, 4, 5, 6, 120, 300, 600, 900, 1200, 1800)]
        b = ContextGate(max_lines=8, max_chars=4000).gate(lines)
        self.assertEqual(sum(b.repeats.values()), 0)
        self.assertEqual(len(b.escalations), 1, "одна подпись - 12 числовых вариантов")
        self.assertEqual(max(b.escalations.values()), 12)

    def test_repetitive_log_still_worth_gating(self):
        """Обратная сторона: настоящий лог с повторами урезать стоит."""
        log = ["2026-09-20 12:00:0%d INFO starting tool call" % (i % 10) for i in range(30)]
        b = ContextGate(max_lines=8, max_chars=4000).gate(log)
        self.assertTrue(apply_policy(b), "лог из повторов - законная цель гейта")
        self.assertLess(b.chars_out, b.chars_in)


if __name__ == "__main__":
    unittest.main(verbosity=2)
