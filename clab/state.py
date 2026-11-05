"""StateVector - компактный вектор состояния вместо пересказа истории.

Биология: кольцевой аттрактор центрального комплекса мухи (~3000 нейронов,
E-PG клетки) держит текущий угол полёта и обновляет его дельта-импульсами,
не пересчитывая всю картину мира. Аналог для агента - маленький JSON:
цель, шаг, блокер, последняя ошибка, прогресс.

Зачем: дрейф контекста и «круги ходами» - это когда агент пересказывает себе
историю вместо того, чтобы держать состояние. Вектор состояния стоит десятки
токенов и не плывёт.

Универсально: подходит любому агенту/процессу - не только мне.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field


def _cut(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


@dataclass
class StateVector:
    goal: str = ""
    step: str = ""
    blocked: str = ""
    last_error: str = ""
    progress: float = 0.0
    next_action: str = ""
    note: str = ""
    ts: float = 0.0
    limits: dict = field(default_factory=lambda: {"goal": 80, "step": 60,
                                                  "blocked": 60, "last_error": 80,
                                                  "next_action": 60, "note": 60})

    def touch(self) -> "StateVector":
        self.ts = time.time()
        return self

    def render(self) -> str:
        """Одна строка для промпта - десятки токенов вместо истории."""
        lim = self.limits
        parts = [
            f"[цель] {_cut(self.goal, lim['goal']) or '-'}",
            f"[шаг] {_cut(self.step, lim['step']) or '-'}",
            f"[блок] {_cut(self.blocked, lim['blocked']) or 'нет'}",
            f"[ошибка] {_cut(self.last_error, lim['last_error']) or 'нет'}",
            f"[прогресс] {int(round(100 * max(0.0, min(1.0, self.progress))))}%",
        ]
        if self.next_action:
            parts.append(f"[далее] {_cut(self.next_action, lim['next_action'])}")
        if self.note:
            parts.append(f"[заметка] {_cut(self.note, lim['note'])}")
        return " | ".join(parts)

    def tokens_est(self) -> int:
        return int(math.ceil(len(self.render()) / 4.0))

    def to_json(self) -> str:
        d = asdict(self)
        d["ts"] = self.ts
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "StateVector":
        d = json.loads(raw)
        lim = d.pop("limits", None)
        sv = cls(**d)
        if lim:
            sv.limits.update(lim)
        return sv
