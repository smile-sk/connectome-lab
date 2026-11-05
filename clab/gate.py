"""ContextGate - латеральное торможение: вырезание шума из промпта.

Биология: тормозные ГАМК-нейроны глушат соседей при срабатывании сильного
импульса - так мозг держит контраст и не тонет в фоне. В коннектоме червя это
видно числами: 13.2% связей несут половину веса, остальное - подложка.

Инженерия: вывод инструмента (100 строк логов) не должен целиком уезжать в
историю. Гейт делает три вещи:
  1. нормализует строку в подпись (hex/uuid → #, номера → N) - чтобы «одна и
     та же» ошибка с разными числами считалась повтором, а не новостью.
     НО для error/warn числа в подписи сохраняются: «timeout after 120s» и
     «timeout after 300s» - разные сообщения, эскалация должна быть видна
     (QA-находка №2, Орест, 2026-09-20);
  2. спрашивает привыкание (Habituation) - пропускать или приглушить;
  3. добирает остаток по значимости в бюджет символов, остальное физически
     заменяет строкой `[подавлено N строк]`.

Смысл: в промпт уезжает 1-2 самые значимые строки и шапка, а не стек целиком.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .habituation import Habituation

_ERR = re.compile(r"(?i)\b(error|exception|traceback|failed|failure|fatal|denied|refused|timeout|critical)\b")
_WARN = re.compile(r"(?i)\b(warn|warning|deprecated|retry|slow|stale|skipped)\b")
_NOISE = re.compile(r"(?i)\b(debug|trace|verbose|heartbeat|keepalive|poll|ping)\b")
_HEX = re.compile(r"\b[0-9a-fA-F]{7,}\b")
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")

KIND_WEIGHT = {"error": 3.0, "signal": 2.0, "warn": 1.5, "info": 1.0, "noise": 0.4}


def guess_kind(line: str) -> str:
    if _ERR.search(line):
        return "error"
    if _WARN.search(line):
        return "warn"
    if _NOISE.search(line):
        return "noise"
    return "info"


def signature(line: str, max_len: int = 140, keep_numbers: bool = False) -> str:
    """Подпись строки: «тот же тип сообщения».

    keep_numbers=True (для error/warn) оставляет числа, поэтому «timeout after
    120s» и «timeout after 300s» - РАЗНЫЕ подписи, и эскалация не теряется как
    повтор. Хексы/длинные id всё равно сворачиваются в «#», иначе меняющийся
    request-id делал бы каждую строку уникальной.
    """
    s = _HEX.sub("#", line.strip())
    if not keep_numbers:
        s = _NUM.sub("N", s)
    s = _WS.sub(" ", s)
    return s[:max_len].lower()


@dataclass
class GatedBlock:
    text: str
    kept: int = 0
    suppressed: int = 0
    chars_in: int = 0
    chars_out: int = 0
    kept_lines: list = field(default_factory=list)
    dropped_kinds: dict = field(default_factory=dict)
    repeats: dict = field(default_factory=dict)
    escalations: dict = field(default_factory=dict)  # подпись → сколько разных числовых вариантов

    @property
    def tokens_in_est(self) -> int:
        return self.chars_in // 4

    @property
    def tokens_out_est(self) -> int:
        return self.chars_out // 4

    @property
    def hab_drops(self) -> int:
        """Сколько строк убрано ПРИВЫКАНИЕМ/дедупом (а не бюджетом)."""
        return sum(int(v) for v in self.dropped_kinds.values())

    def summary(self) -> dict:
        return {
            "kept": self.kept, "suppressed": self.suppressed,
            "chars_in": self.chars_in, "chars_out": self.chars_out,
            "tokens_in_est": self.tokens_in_est, "tokens_out_est": self.tokens_out_est,
            "saved_pct": round(100.0 * (1 - self.chars_out / max(1, self.chars_in)), 1),
            "dropped_kinds": dict(self.dropped_kinds),
            "repeats": sum(self.repeats.values()),
            "escalations": len(self.escalations),
            "hab_drops": self.hab_drops,
        }


class ContextGate:
    """Фильтр перед формированием промпта: оставить суть, приглушить фон."""

    def __init__(self, hab: Habituation | None = None, max_chars: int = 1200,
                 max_lines: int = 8, head_lines: int = 1, keep_kinds=("error",),
                 numeric_kinds=("error", "warn", "info"), line_cap: int = 400):
        self.hab = hab or Habituation()
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.head_lines = head_lines
        self.keep_kinds = tuple(keep_kinds)
        self.numeric_kinds = tuple(numeric_kinds)   # для них числа значимы: всё, кроме шума
        self.line_cap = int(line_cap)               # однострочный гигант не съест бюджет целиком

    def gate(self, lines, ts: float | None = None, header: str | None = None) -> GatedBlock:
        lines = [str(x) for x in lines]
        chars_in = sum(len(x) + 1 for x in lines)
        if not lines:
            return GatedBlock(text="", chars_in=0, chars_out=0)
        if header is None:
            header = lines[0] if self.head_lines else None

        cand = []
        dropped = {}
        seen_sig = {}
        repeats = {}
        variants = {}
        for i, line in enumerate(lines):
            kind = guess_kind(line)
            base = signature(line)                                    # «тот же тип сообщения»
            exact = signature(line, keep_numbers=kind in self.numeric_kinds)
            variants.setdefault(base, set()).add(exact)
            dup = seen_sig.setdefault(exact, i) != i                  # повтор ВНУТРИ блока
            # Ключ привыкания. Шум (heartbeat/poll/retry с меняющимися счётчиками)
            # Гасим по ОБЕЗЛИЧЕННОЙ подписи - там числа не значимы. Всё прочее - по
            # ТОЧНОЙ: «row 0: value=0» и «row 1: value=7919» - РАЗНЫЕ стимулы, иначе
            # 40 строк расчёта схлопнулись бы в 4 (находка живого теста харнесса,
            # 20.09.2026: 36 из 40 строк данных было подавлено как «повтор»).
            key_sig = base if kind == "noise" else exact
            key = f"{kind}:{key_sig}"
            v = self.hab.observe(key, kind=kind, ts=ts, text=line)
            forced = (kind in self.keep_kinds or i < self.head_lines) and not dup
            if dup:
                repeats[base] = repeats.get(base, 0) + 1
                dropped[kind] = dropped.get(kind, 0) + 1
            elif not forced and not v.emit:
                dropped[kind] = dropped.get(kind, 0) + 1
            shown = line if len(line) <= self.line_cap else \
                line[:self.line_cap] + f"…[{len(line) - self.line_cap} симв. срезано]"
            score = v.response * KIND_WEIGHT.get(kind, 1.0) * (1.0 + 0.15 * min(len(line), 200) / 200.0)
            cand.append({"i": i, "line": shown, "kind": kind, "forced": forced,
                         "score": score, "verdict": v, "dup": dup})

        chosen, budget, used = set(), self.max_chars, 0
        for c in sorted(cand, key=lambda c: -c["score"]):
            if c["dup"]:          # тот же самый текст внутри блока - второй раз не нужен
                continue
            if len(chosen) >= self.max_lines:
                break
            if c["forced"]:
                chosen.add(c["i"])
                used += len(c["line"]) + 1
                continue
            if used + len(c["line"]) + 1 > budget:
                continue
            chosen.add(c["i"])
            used += len(c["line"]) + 1

        kept_lines = [c["line"] for c in cand if c["i"] in chosen]
        body = list(kept_lines)
        if header:
            hdr = header if len(header) <= self.line_cap else header[:self.line_cap] + "…"
            if hdr not in body:
                body = [hdr] + body
        suppressed = len(lines) - len(body)
        escalations = {b: len(v) for b, v in variants.items() if len(v) > 1}
        if suppressed > 0:
            extra = f", повторов {sum(repeats.values())}" if repeats else ""
            if escalations:
                extra += f", эскалаций {len(escalations)}"
            body.append(f"[подавлено {suppressed} строк{extra}]")
        text = "\n".join(body)
        return GatedBlock(text=text, kept=len(body), suppressed=suppressed,
                          chars_in=chars_in, chars_out=len(text),
                          kept_lines=kept_lines, dropped_kinds=dropped,
                          repeats=repeats, escalations=escalations)


# ──────────────────────────────────────────────────────────────────────────
# Слой политики - перенесён из ai_ui/app/agents/context_gate.py 20.09.2026,
# чтобы две копии не разъезжались: решает, СТОИТ ли применять урезанный текст.
# ──────────────────────────────────────────────────────────────────────────

# Поля результата инструмента, в которых лежит «многострочный текст».
TEXT_KEYS: tuple = ("stdout", "stderr", "content", "text", "output", "result",
                    "note", "log", "logs", "lines", "detail")


def apply_policy(block: GatedBlock, min_lines: int = 8,
                 min_share: float = 0.4) -> bool:
    """Стоит ли ПРИМЕНЯТЬ урезанный текст (а не отдать вывод целиком).

    Ключевая проверка харнесса: экономия должна быть сделана ПОВТОРАМИ, а не
    бюджетом. Если строки уникальны (расчёт, данные, файл) - гейт не трогает
    ничего: 50 строк результата дороже, чем 50 строк лога.
    """
    total = max(1, block.kept + block.suppressed)
    need = max(3, int(min_share * total))
    return total >= min_lines and block.hab_drops >= need


def gate_result(gate: ContextGate, result, min_chars: int = 1000,
                keys: tuple = TEXT_KEYS):
    """Прогнать текстовые поля результата инструмента через гейт.

    Возвращает (результат, статистика по полям). Статистика пуста - гейт не
    сработал (короткий вывод или нет повторов), и результат отдан как есть.
    Служебные поля (ok / error / exit_code) не трогаются НИКОГДА: именно они
    решают, «получилось» или нет, - их приглушать нельзя.
    """
    if not isinstance(result, dict):
        return result, {}
    out = None
    stats: dict = {}
    for key in keys:
        val = result.get(key)
        if not isinstance(val, str) or len(val) < int(min_chars):
            continue
        block = gate.gate(val.splitlines())
        if not apply_policy(block):
            continue
        if out is None:
            out = dict(result)      # копия: исходный результат не мутируем
        out[key] = block.text
        stats[key] = block.summary()
    return (out if out is not None else result), stats
