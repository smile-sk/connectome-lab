# -*- coding: utf-8 -*-
"""Защита от возврата хардкода в E4 и от устаревания файла активности.

Зачем: QA 2026-09-20 поймал, что таблица «Эксперимент с усталостью» в e4_activity.py
и out/e4_report.md была ВПИСАНА ЧИСЛАМИ и разошлась с кодом после правок dynamics.py
(заявляла 0.0930/0.0396/0.0403 против воспроизводимых 0.08853/0.04529/0.05365).
Теперь таблица считается в прогоне, а тест следит:
  1) расчёт на месте, хардкода строк таблицы нет;
  2) файл активности для тела не устарел относительно clab/dynamics.py.

Только stdlib, сеть НЕ прогоняется (прогон дорогой: таблица ~3 мин).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "e4_activity.py"
DYN = ROOT / "clab" / "dynamics.py"
REP = ROOT / "out" / "e4_report.md"
ACT = ROOT / "out" / "worm_activity.json"

try:                                                       # кириллица в консоли
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover
    pass

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print("  OK   %s" % name)
    else:
        fail += 1
        print("  FAIL %s %s" % (name, detail))


def main():
    t = SRC.read_text(encoding="utf-8")

    check("есть simulate()", "def simulate(" in t)
    check("есть detrend_drive()", "def detrend_drive(" in t)
    check("есть fatigue_table()", "def fatigue_table(" in t)
    check("main вызывает fatigue_table()", "fatigue_table(c, p, ash, back, fwd" in t)
    check("есть флаг --no-fatigue", '"--no-fatigue"' in t)
    check("N_FATIGUE_TAPS = 3", re.search(r"N_FATIGUE_TAPS\s*=\s*3", t) is not None)

    hc = [ln.strip() for ln in t.splitlines() if re.match(r'^"[|]\s*0\.\d\d\s*[|]', ln.strip())]
    check("нет вписанных строк таблицы усталости", not hc, str(hc[:2]))

    if REP.is_file():
        r = REP.read_text(encoding="utf-8", errors="replace")
        check("отчёт помечен md5 dynamics.py", "md5 clab/dynamics.py" in r)
        check("в таблице отчёта нет старых чисел", not re.search(r"\| 0\.00 \| 0\.0930 \|", r))
    else:
        print("  ..   отчёта нет - пропускаю")

    if ACT.is_file() and DYN.is_file():
        stamp = json.loads(ACT.read_text(encoding="utf-8"))["meta"].get("dynamics_md5", "")
        cur = hashlib.md5(DYN.read_bytes()).hexdigest()[:8]
        check("worm_activity.json снят ТЕКУЩИМ dynamics.py", stamp == cur,
              "в файле %s, сейчас %s -> перепрогони scripts/e4_activity.py" % (stamp or "нет", cur))
    else:
        print("  ..   файла активности нет - пропускаю")

    print("\nE4 guard: PASS %d / FAIL %d" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
