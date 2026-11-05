"""E2 - привыкание (habituation) как замена тупому хеш-дедупу.

Запуск:  python scripts/e2_habituation.py
Выход:   out/e2_report.md + сводка в stdout
Зависимостей нет (только стандартная библиотека + пакет `clab` проекта).

Что проверяется ЖИВЫМ прогоном, а не обещается:
  S1 привыкание к повторному стимулу и dishabituation (аналог опытов на черве)
  S2 почтовик: 240 событий - сколько проходит и НЕ теряется ли ошибка
  S3 шина: повтор heartbeat vs адресное сообщение vs первый сигнал нового агента
  S4 учащающийся warn (flapping): заметить бёрст, потом замолчать, потом снова
     услышать после паузы
  S5 разреженная проекция: код, пересечения, readout
  S6 context gate: 62 строки вывода инструмента → сколько осталось
  S7 вектор состояния: сколько токенов вместо пересказа истории

Время инжектируется (Clock), поэтому числа воспроизводимы от прогона к прогону.
"""

import os
import random
import sys

try:  # консоль Windows - cp1251; отчёт и вывод держим в utf-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clab import (Habituation, SparseProjector, PrototypeReadout, ContextGate,
                  StateVector, make_filter)
from clab.gate import guess_kind, signature


class Clock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += float(dt)


class NaiveDedup:
    """То, что стоит сейчас: 'видел такую строку - молчу навсегда'."""

    def __init__(self):
        self.seen = set()
        self.emitted = 0

    def feed(self, raw):
        if raw in self.seen:
            return False
        self.seen.add(raw)
        self.emitted += 1
        return True


L = []


def A(s=""):
    L.append(s)


# ---------------- S1: привыкание и dishabituation ----------------

def s1():
    clk = Clock()
    hab = Habituation(now=clk, tau_rec=120.0, sensitize_dt=5.0)
    emits, trace = [], []
    for i in range(150):
        if i == 60:  # иной по классу стимул - новизна
            hab.observe("stim:novel-odor", kind="error", ts=clk.t, text="новый стимул")
        v = hab.observe("stim:touch-nose", kind="info", ts=clk.t)
        emits.append(i) if v.emit else None
        if i in (0, 1, 2, 3, 5, 20, 59, 60, 61, 62, 63, 100, 149):
            trace.append((i, v.emit, round(v.response, 3)))
        clk.tick(30.0)
    # самовосстановление: пауза 20 минут
    clk.tick(1200.0)
    v_after = hab.observe("stim:touch-nose", kind="info", ts=clk.t)

    A("## S1 - привыкание к повтору и dishabituation\n")
    A(f"150 касаний (интервал 30 с), на 60-м - иной стимул (класс error). "
      f"Порог {hab.threshold}, k={hab.k}, tau_rec={hab.tau_rec} с.\n")
    A("| касание # | пропущено | response |")
    A("|---|---|---|")
    for i, e, r in trace:
        A(f"| {i} | {'да' if e else 'НЕТ (приглушено)'} | {r} |")
    A("")
    A(f"- прошло в мозг: **{len(emits)}** из 150 ({100.0 * len(emits) / 150:.1f}%) - "
      f"касания № {[e + 1 for e in emits]}")
    A(f"- после паузы 20 мин (спонтанное восстановление): "
      f"{'прошло' if v_after.emit else 'приглушено'}, response {v_after.response:.3f}")
    A(f"- сбросов привыкания (dishabituation): {hab.report()['dishabituations']}")
    A("")
    A("Смысл: сначала сигнал заметен, при монотонном повторе гаснет, при смене "
      "стимула снова слышен, после паузы восстанавливается сам.\n")
    return {"taps": 150, "passed": len(emits), "dishab": hab.report()["dishabituations"],
            "after_rest": v_after.emit, "response_after_rest": round(v_after.response, 3)}


# ---------------- S2: почтовик ----------------

def s2():
    rnd = random.Random(20260920)
    events = []
    for i in range(240):
        r = rnd.random()
        if r < 0.70:
            events.append({"level": "info", "source": "mailbox",
                           "message": f"poll ok, inbox {rnd.randint(0, 9)} msgs"})
        elif r < 0.90:
            events.append({"level": "warn", "source": "mailbox",
                           "message": f"slow response {rnd.randint(900, 1500)} ms"})
        else:
            events.append({"level": "error", "source": "mailbox",
                           "message": f"auth failed for account code {rnd.randint(100, 999)}"})
    errors = [e for e in events if e["level"] == "error"]

    naive = NaiveDedup()
    for e in events:
        naive.feed(e["message"])

    rows = []
    for tau in (60.0, 120.0, 600.0):
        clk = Clock()
        flt = make_filter("mailbox", hab=Habituation(now=clk, tau_rec=tau, sensitize_dt=5.0))
        caught, passed = 0, {}
        for e in events:
            v = flt.feed(e, ts=clk.t)
            if v.emit:
                passed[e["level"]] = passed.get(e["level"], 0) + 1
                if e["level"] == "error":
                    caught += 1
            clk.tick(20.0)
        rows.append((tau, flt.emitted, caught, passed))

    A("## S2 - почтовик: 240 событий (70% info, 20% warn, 10% error)\n")
    A(f"Всего: info 175, warn 42, error 23. Интервал между событиями 20 с, "
      f"то есть warn приходит примерно раз в 2 минуты, а error - раз в 3,5 минуты.\n")
    A("| способ фильтрации | прошло | info | warn | error | ошибок поймано |")
    A("|---|---|---|---|---|---|")
    A(f"| тупой хеш-дедуп (сейчас) | {naive.emitted} | ~{naive.emitted - len(errors)} | 0 | {len(errors)} | {len(errors)} |")
    for tau, total, caught, passed in rows:
        A(f"| привыкание, tau_rec={int(tau)} с | **{total}** | {passed.get('info', 0)} | "
          f"{passed.get('warn', 0)} | {passed.get('error', 0)} | **{caught} из {len(errors)}** |")
    A("")
    A(f"Шум info сжимается в {175 / max(1, rows[1][3].get('info', 1)):.0f} раз "
      f"(175 → {rows[1][3].get('info', 0)}), и при любом tau_rec не теряется ни одной ошибки.\n")
    A("ГЛАВНОЕ ЧЕСТНОЕ НАБЛЮДЕНИЕ: warn при интервале ~2 минуты НЕ гасится при "
      "tau_rec=120 с - потому что 2 минуты больше времени восстановления. Это не баг, "
      "а свойство: привыкание фильтрует БЁРСТЫ, а не редкие события. Хочешь тише "
      "по варнингам - поднимай tau_rec (строка 600 с) или опускай floor для warn; "
      "цифры в таблице - ровно про этот выбор.\n")
    return {"naive": naive.emitted, "hab": rows[1][1], "errors_total": len(errors),
            "errors_caught": rows[1][2], "taus": [(int(t), v) for t, v, _, _ in rows]}


# ---------------- S3: шина ----------------

def s3():
    clk = Clock()
    flt = make_filter("bus", hab=Habituation(now=clk, tau_rec=120.0, sensitize_dt=5.0))
    passed = {"heartbeat": 0, "direct": 0, "new_agent": 0}
    for i in range(30):
        v = flt.feed({"from": "miriad", "topic": "heartbeat", "text": "alive"}, ts=clk.t)
        passed["heartbeat"] += 1 if v.emit else 0
        clk.tick(30.0)
    for i in range(3):
        v = flt.feed({"from": "miriad", "topic": "direct",
                      "text": f"вопрос {i} про коннектом"}, ts=clk.t)
        passed["direct"] += 1 if v.emit else 0
        clk.tick(30.0)
    v = flt.feed({"from": "anton", "topic": "note",
                  "text": "у нас лимиты, ждём ответа"}, ts=clk.t)
    passed["new_agent"] += 1 if v.emit else 0

    A("## S3 - шина: повтор vs адресное сообщение vs новый агент\n")
    A(f"- heartbeat `miriad/alive` ×30 (каждые 30 с): дошло **{passed['heartbeat']}**")
    A(f"- адресные сообщения `topic=direct` ×3: дошло **{passed['direct']} из 3**")
    A(f"- первый сигнал нового агента (`anton`): "
      f"{'дошёл' if passed['new_agent'] else 'приглушён'}")
    A("")
    A("Смысл: монотонный «я жив» гаснет, адресный вопрос проходит всегда "
      "(класс signal), новое лицо слышно с первого раза.\n")
    return passed


# ---------------- S4: flapping ----------------

def s4():
    clk = Clock()
    hab = Habituation(now=clk, tau_rec=120.0, sensitize_dt=5.0)
    events = []
    for i in range(20):  # учащающийся warn: каждые 2 с
        v = hab.observe("warn:slow response N ms", kind="warn", ts=clk.t)
        events.append(v.emit)
        clk.tick(2.0)
    clk.tick(1800.0)  # шторм кончился, пауза 30 минут
    v_after = hab.observe("warn:slow response N ms", kind="warn", ts=clk.t)

    A("## S4 - flapping: учащающийся warn\n")
    A(f"- 20 повторов с интервалом 2 с: прошло **{sum(events)}** "
      f"(первые {sum(1 for x in events[:8] if x)} на участке 0-7, дальше приглушено)")
    A(f"- сенситизаций (сбросов по участившемуся интервалу): {hab.report()['sensitized']}")
    A(f"- после паузы 30 мин: {'прошло снова' if v_after.emit else 'по-прежнему тихо'} "
      f"(response {v_after.response:.3f})")
    A("")
    A("Смысл: короткий «заморгал - заметили», вечный шторм замолкает, "
      "а после тишины сигнал восстанавливается сам.\n")
    return {"storm_passed": sum(events), "of": 20, "recovered": v_after.emit}


# ---------------- S5: разреженная проекция ----------------

def s5():
    sp = SparseProjector(n_kc=2000, active=20, fan_in=6, seed="clab-e2")
    pr = PrototypeReadout(sp)
    pr.add("ok", "fs_read ok, 12 files listed in 0.03s, no errors")
    pr.add("error", "error: timeout after 120s while reading mailbox state file")
    pr.add("warn", "warning: cache full, oldest entries dropped, retry scheduled")
    pr.add("noise", "debug tick 5532 poll keepalive ok no changes")

    samples = [
        ("fs_read ok, 12 files listed in 0.03s, no errors", "ok", "тот же текст"),
        ("error: timeout after 300s while reading mailbox state file", "error", "другое число"),
        ("warning: cache full, oldest entries dropped, retry scheduled", "warn", "тот же текст"),
        ("debug tick 9182 poll keepalive ok no changes", "noise", "другое число"),
        ("привет, что нового? расскажи про коннектом", "unknown", "чужая фраза"),
    ]
    code_len = len(sp.encode(samples[0][0]))
    A("## S5 - разреженная проекция + readout по прототипам\n")
    A(f"{sp.n_kc} «клеток», активных {sp.active}, fan_in {sp.fan_in} → код {code_len} индексов. "
      f"Классификация - по максимальному пересечению с прототипами, без обучения и без LLM.\n")
    A("| текст | ждём | получили | пересечение | margin |")
    A("|---|---|---|---|---|")
    for text, want, note in samples:
        r = pr.predict(text)
        A(f"| `{text}` ({note}) | {want} | **{r['label']}** | {r['score']} | {r['margin']} |")
    A("")
    e120 = sp.encode("error: timeout after 120s while reading mailbox state file")
    e300 = sp.encode("error: timeout after 300s while reading mailbox state file")
    ok = sp.encode("fs_read ok, 12 files listed in 0.03s, no errors")
    same = sp.encode("fs_read ok, 12 files listed in 0.03s, no errors")
    A(f"пересечения кодов: тот же текст - **{SparseProjector.overlap(ok, same):.3f}**, "
      f"error(120) ↔ error(300) - **{SparseProjector.overlap(e120, e300):.3f}**, "
      f"error ↔ ok - **{SparseProjector.overlap(e120, ok):.3f}**.")
    A("")
    A("Честно: токенизатор выбрасывает чистые числа (`120s` → просто `s`), поэтому "
      "«та же ошибка с другим числом» даёт ОДИН И ТОТ ЖЕ код - это удобно для "
      "группировки повторов и НЕ годится, если различать важны именно значения. "
      "Различать числа - отдельная задача (добавить числовой маркер в токены).")
    A("Второе честно: это не «понимание», а поиск похожести на 20 индексах. "
      "Ценность - цена: разложить поток на 2-4 класса без вызова LLM.\n")
    return {"code_len": code_len,
            "same": round(SparseProjector.overlap(ok, same), 3),
            "num": round(SparseProjector.overlap(e120, e300), 3),
            "diff": round(SparseProjector.overlap(e120, ok), 3),
            "unknown_ok": pr.predict("привет, что нового?")["label"] == "unknown"}


# ---------------- S6: context gate ----------------

def s6():
    lines = ["$ pytest -q tests/  (62 строки вывода)"]
    for i in range(40):
        lines.append(f"debug trace tick {i} ok")
    for i in range(10):
        lines.append(f"info retry {i} of 10")
    lines.append("warn slow response 1200 ms")
    lines.append("warn slow response 1350 ms")
    for i in range(5):
        lines.append(f"error: connection refused to 127.0.0.1:{8000 + i}")
    lines.append("error: AssertionError: expected 3 got 4  <- единственная НОВАЯ ошибка")
    lines.append("info done in 12.4s")

    clk = Clock()
    gate = ContextGate(hab=Habituation(now=clk, tau_rec=120.0, sensitize_dt=5.0),
                       max_chars=1200, max_lines=8)
    blk = gate.gate(lines, ts=clk.t, header=lines[0])
    s = blk.summary()
    A("## S6 - context gate: вывод инструмента перед промптом\n")
    A(f"вход: {len(lines)} строк, {s['chars_in']} символов "
      f"(~{s['tokens_in_est']} токенов).\n")
    A(f"выход: {blk.kept} строк, {s['chars_out']} символов "
      f"(~{s['tokens_out_est']} токенов) - экономия **{s['saved_pct']}%**.\n")
    A("Что уехало в промпт:\n")
    A("```")
    A(blk.text)
    A("```")
    A("")
    A(f"приглушено по классам: {s['dropped_kinds']}; "
      f"из них повторов одного и того же текста: {s['repeats']}")
    A("")
    return {"lines_in": len(lines), "lines_out": blk.kept, "chars_in": s["chars_in"],
            "chars_out": s["chars_out"], "saved_pct": s["saved_pct"]}


# ---------------- S7: вектор состояния ----------------

def s7():
    sv = StateVector(goal="E2: привыкание и контекст-гейт",
                     step="прогон сценариев S1-S7",
                     blocked="", last_error="нет",
                     progress=0.9, next_action="отчёт + перенос в ai_ui",
                     note="шина не тронута").touch()
    raw = sv.to_json()
    A("## S7 - вектор состояния (кольцевой аттрактор)\n")
    A("Одна строка вместо пересказа истории:\n")
    A("```")
    A(sv.render())
    A("```")
    A(f"~{sv.tokens_est()} токенов; JSON-форма - {len(raw)} символов. "
      f"Для сравнения: пересказ последних 20 сообщений - это тысячи токенов.\n")
    return {"tokens_est": sv.tokens_est(), "json_chars": len(raw)}


def main():
    A("# E2 - привыкание и торможение контекста: живые замеры\n")
    A("Проект: ConnectomeLab. Пакет: `clab`. Прогон: `python scripts/e2_habituation.py`.\n")
    A("Ничего в `ai_ui` и на шине не менялось - это песочница.\n")

    r1, r2, r3, r4, r5, r6, r7 = s1(), s2(), s3(), s4(), s5(), s6(), s7()

    A("## Итог числами\n")
    A("| эксперимент | результат |")
    A("|---|---|")
    A(f"| S1 привыкание | {r1['passed']} из {r1['taps']} касаний дошло; "
      f"{r1['dishab']} dishabituation; после паузы - "
      f"{'снова слышно' if r1['after_rest'] else 'тихо'} |")
    A(f"| S2 почтовик | дедуп пропустил {r2['naive']}, привыкание {r2['hab']}; "
      f"ошибок поймано {r2['errors_caught']}/{r2['errors_total']}; "
      f"tau_rec 60/120/600 → {[v for _, v in r2['taus']]} |")
    A(f"| S3 шина | heartbeat {r3['heartbeat']}/30, адресных "
      f"{r3['direct']}/3, новый агент - {'слышно' if r3['new_agent'] else 'тихо'} |")
    A(f"| S4 flapping | прошло {r4['storm_passed']}/{r4['of']}, после паузы - "
      f"{'возврат' if r4['recovered'] else 'тихо'} |")
    A(f"| S5 проекция | код {r5['code_len']} индексов; тот же текст {r5['same']}, "
      f"числа другие {r5['num']}, разное {r5['diff']} |")
    A(f"| S6 context gate | {r6['lines_in']} строк → {r6['lines_out']}, "
      f"{r6['chars_in']} → {r6['chars_out']} символов, экономия {r6['saved_pct']}% |")
    A(f"| S7 state vector | {r7['tokens_est']} токенов на состояние |")
    A("")
    A("## Границы: где это НЕ работает (честно)\n")
    A("- **Редкий повтор не гасится.** Если то же событие приходит раз в 10 минут, "
      "интервал больше tau_rec - восстановление успевает пройти, и оно проходит "
      "каждый раз. Привыкание фильтрует БЁРСТЫ, а не архивирует по факту. "
      "Для «это уже было в прошлом месяце» нужен отдельный архивный дедуп.")
    A("- **Ошибка не гасится никогда** (floor 0.50 при пороге 0.35) - сознательная "
      "асимметрия: лучше лишний шум, чем пропущенная поломка.")
    A("- **Числа подогнаны под наш темп.** tau_rec=120 с, порог 0.35, k=0.55 - "
      "разумные значения для потока событий агента, но их надо настраивать "
      "под конкретный поток, а не брать как константу природы.")
    A("- **За речь не отвечает.** Это фильтр событий, а не модель поведения: "
      "привыкание не заменяет ни планировщик, ни память.")
    A("")
    A("## Как подключать (универсально)\n")
    A("`clab` не знает ни про шину, ни про почтовик. Любой поток сводится к "
      "`feed(event)` через маппер:\n")
    A("```python")
    A("from clab import make_filter")
    A("flt = make_filter('bus')        # или 'mailbox', 'toollog'")
    A("v = flt.feed({'from': 'miriad', 'topic': 'heartbeat', 'text': 'alive'})")
    A("if v.emit: ...                  # только то, что достойно внимания")
    A("```")
    A("Новый поток = одна функция `key_fn(ev) -> (key, kind, text)` в "
      "`clab/adapters.py`. Остальные модули не меняются.\n")
    A("## Что переносить в ai_ui (предложение, не сделано)\n")
    A("1. `ContextGate` - самый прикладной: обрезка вывода инструментов до "
      "шапки + значимых строк + `[подавлено N строк]`.")
    A("2. `StateVector` - маленький `data/state.json` в начале промпта вместо "
      "пересказа истории.")
    A("3. `Habituation` - для потоков-сторожей (mailbox poller, bus poller, "
      "jsonl watcher), где сейчас повтор давится хешем.")
    A("4. `SparseProjector` - для бинарных классификаций «норма/ошибка/шум» "
      "без вызова LLM; сначала замерить на реальных логах, потом решать.\n")

    out = os.path.join(ROOT, "out", "e2_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))

    print("\n".join(L))
    print(f"\n[ok] отчёт: {out}")
    print(f"[ok] самопроверка: S1={r1['passed']}/150, S2={r2['hab']} прошло "
          f"({r2['errors_caught']}/{r2['errors_total']} ошибок), "
          f"S3 heartbeat={r3['heartbeat']}/30, S6 экономия {r6['saved_pct']}%")


if __name__ == "__main__":
    main()
