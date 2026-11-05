"""Habituation - привыкание к повтору стимула (C. elegans → агентный харнесс).

Биология, на которой стоит модуль (проверяемые факты, не метафора):
1. повтор одного и того же стимула гасит ответ, но НЕ до нуля: у червя ответ
   падает до плато - habituation;
2. без стимула ответ восстанавливается сам - spontaneous recovery;
3. сильный/иной стимул сбрасывает привыкание ко всем прежним - dishabituation
   (в модели: только новый по подписи стимул и не чаще dh_cooldown секунд);
4. важный сигнал (реальная поломка) обязан остаться заметным - это floor по типу.

Практика: замена «тупого» хеш-дедупа повторов на кривую затухания.
Ключевая разница: дедуп говорит «это уже было» навсегда, привыкание говорит
«это было недавно и часто - сейчас тихо, а через tau_rec снова слышно».

Модель (одна формула, без обучения):
    n_eff(t) = n_eff * exp(-dt / tau_rec) + weight     # восстановление + повтор
    response = floor(kind) + (1 - floor(kind)) * exp(-k * n_eff)
    emit     = response >= threshold

Сенситизация (обратный процесс): если интервал между повторами резко упал
(dt < sensitize_dt) - это уже не тот же стимул, а учащающийся: сбрасываем
привыкание (dishabituation) и пропускаем. Но только burst_cap раз подряд,
иначе вечный шторм никогда не отфильтруется.

Зависимостей нет. Время инжектируется (`now=`), поэтому поведение
детерминировано и воспроизводимо в отчётах.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

# Плато, до которого может упасть ответ, по типу события.
DEFAULT_FLOOR = {
    "error": 0.50,   # поломка - выше порога, не гасится никогда
    "signal": 0.40,  # адресное сообщение другого агента - тоже
    "warn": 0.30,    # предупреждение - гасится при настойчивом повторе
    "info": 0.08,    # фоновая информация - гасится быстро
    "noise": 0.00,   # шум - гасится сразу
}


@dataclass
class Entry:
    """Состояние привыкания по одному ключу стимула."""
    n_eff: float = 0.0        # накопленная «усталость» с учётом восстановления
    last_ts: float = 0.0
    seen: int = 0
    emitted: int = 0
    suppressed: int = 0
    burst: int = 0            # сколько раз подряд сработала сенситизация
    last_dt: float | None = None
    last_text: str = ""
    last_kind: str = "info"   # тип последнего события по ключу - floor берём по нему


@dataclass
class Verdict:
    """Решение по одному событию: пропускать или приглушить."""
    emit: bool
    response: float
    n_eff: float
    reason: str
    kind: str = "info"
    key: str = ""

    def as_dict(self) -> dict:
        return {
            "emit": self.emit, "response": round(self.response, 4),
            "n_eff": round(self.n_eff, 4), "reason": self.reason,
            "kind": self.kind, "key": self.key,
        }


class Habituation:
    """Привыкание по ключам стимулов. Один экземпляр = один поток событий."""

    def __init__(
        self,
        k: float = 0.55,
        tau_rec: float = 120.0,
        threshold: float = 0.35,
        floor: dict | None = None,
        dishabituation: float = 0.20,
        dh_kinds: tuple = ("error",),
        dh_cooldown: float = 120.0,
        never_suppress: tuple = ("error",),
        sensitize_dt: float = 5.0,
        burst_cap: int = 3,
        now=None,
    ):
        self.k = k
        self.tau_rec = tau_rec
        self.threshold = threshold
        self.floor = dict(DEFAULT_FLOOR)
        if floor:
            self.floor.update(floor)
        self.dishabituation = dishabituation
        self.dh_kinds = tuple(dh_kinds)
        self.dh_cooldown = dh_cooldown
        self._dh_last = None
        self.never_suppress = tuple(never_suppress)
        self.sensitize_dt = sensitize_dt
        self.burst_cap = burst_cap
        self._now = now or time.time
        self._entries: dict[str, Entry] = {}
        self.stats = {"events": 0, "emitted": 0, "suppressed": 0,
                      "dishabituations": 0, "sensitized": 0}

    # ---------- наблюдение ----------

    def observe(self, key: str, kind: str = "info", ts: float | None = None,
                weight: float = 1.0, text: str = "") -> Verdict:
        now = self._now() if ts is None else float(ts)
        e = self._entries.get(key)
        novel = e is None
        if novel:
            e = Entry(last_ts=now)
            self._entries[key] = e
        else:
            dt = max(0.0, now - e.last_ts)
            e.n_eff *= math.exp(-dt / self.tau_rec)   # spontaneous recovery
            e.last_dt = dt
            if dt >= self.sensitize_dt:
                e.burst = 0                          # пауза - бёрст закрыт
            e.last_ts = now

        # dishabituation: только НОВЫЙ стимул класса dh_kinds и не чаще cooldown.
        # Замер E2 (два прогона): сначала тут стояло «любой error сбрасывает» -
        # 42 warn из 42 прошли насквозь, поток не успокаивался. Потом «error не чаще
        # cooldown» - тоже мимо: ошибки шли реже 120 с, и каждая снова всё
        # расшевеливали (91 из 240 на выходе). Работает только НОВИЗНА: сбрасывает
        # первый в жизни такой стимул, а его повторы - нет.
        if kind in self.dh_kinds and novel and (self._dh_last is None
                                               or (now - self._dh_last) >= self.dh_cooldown):
            self._dishabituate(exclude=key if novel else None)
            self._dh_last = now

        reason = "novel" if novel else "repeat"

        # сенситизация: тот же ключ, но интервал резко упал → учащающийся процесс
        if (e.last_dt is not None and e.last_dt < self.sensitize_dt
                and e.burst < self.burst_cap):
            e.burst += 1
            self.stats["sensitized"] += 1
            e.n_eff *= self.dishabituation
            reason = "sensitized"

        e.n_eff += max(0.0, weight)
        e.seen += 1
        e.last_text = text[:200]
        e.last_kind = kind

        fl = self.floor.get(kind, 0.05)
        response = fl + (1.0 - fl) * math.exp(-self.k * e.n_eff)
        emit = response >= self.threshold
        if kind in self.never_suppress:
            emit = True

        if emit:
            e.emitted += 1
            self.stats["emitted"] += 1
        else:
            e.suppressed += 1
            self.stats["suppressed"] += 1
        self.stats["events"] += 1
        return Verdict(emit=emit, response=response, n_eff=e.n_eff,
                       reason=reason, kind=kind, key=key)

    # ---------- служебное ----------

    def _dishabituate(self, exclude: str | None = None) -> None:
        """Сброс привыкания: сильный/новый стимул возвращает чувствительность всем."""
        for k, e in self._entries.items():
            if k == exclude:
                continue
            e.n_eff *= self.dishabituation
        self.stats["dishabituations"] += 1

    def response_of(self, key: str) -> float:
        """Текущий отклик ключа. floor берётся по типу события ЭТОГО ключа.

        Раньше floor был жёстко «info», и для error-ключа отклик занижался
        (0.6108 вместо честных 0.7885) - ошибка выглядела почти притихшей.
        QA-находка №4, Орест, 2026-09-20.
        """
        e = self._entries.get(key)
        if e is None:
            return 0.0
        fl = self.floor.get(e.last_kind, 0.05)
        return fl + (1.0 - fl) * math.exp(-self.k * e.n_eff)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def top(self, n: int = 5, by: str = "emitted") -> list[tuple[str, Entry]]:
        key = (lambda kv: -kv[1].emitted) if by == "emitted" else (lambda kv: -kv[1].seen)
        return sorted(self._entries.items(), key=key)[:n]

    def report(self) -> dict:
        r = dict(self.stats)
        r["keys"] = len(self._entries)
        ev = max(1, self.stats["events"])
        r["emit_ratio"] = round(self.stats["emitted"] / ev, 4)
        return r
