"""adapters - единый вход для ЛЮБОГО потока событий.

Смысл файла: библиотека не должна знать, откуда приходят события. Шина,
почтовик (mailbox_manager), логи инструментов, уведомления UI, сторожа -
всё это превращается в один и тот же вызов feed(event) → Verdict.

Добавить новый поток = добавить функцию-маппер в ADAPTERS (одна строка).
Ничего в остальных модулях менять не нужно - это и есть универсализация.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .habituation import Habituation, Verdict
from .gate import signature


# ---------- мапперы: событие → (key, kind, text) ----------

def bus_key(ev: dict):
    """Сигнал с шины AIUI-BUS: {from, topic, text}."""
    src = str(ev.get("from") or ev.get("agent") or "?")
    topic = str(ev.get("topic") or ev.get("kind") or "note")
    text = str(ev.get("text") or ev.get("message") or "")
    kind = "signal" if topic in ("direct", "ask", "reply", "task") else "info"
    return f"bus:{src}:{topic}:{signature(text)[:60]}", kind, text


def mailbox_key(ev: dict):
    """Событие почтовика: {level, source, message} (браузерные логи, письма, статусы)."""
    level = str(ev.get("level") or "info").lower()
    src = str(ev.get("source") or ev.get("url") or "?")
    text = str(ev.get("message") or ev.get("text") or "")
    if level in ("fatal", "critical", "severe"):
        level = "error"
    kind = level if level in ("error", "warn", "info", "noise") else "info"
    return f"box:{src}:{kind}:{signature(text)[:60]}", kind, text


def toollog_key(ev: dict):
    """Вызов инструмента: {tool, args, ok}."""
    tool = str(ev.get("tool") or "?")
    args = signature(str(ev.get("args") or ""))[:40]
    ok = bool(ev.get("ok", True))
    kind = "info" if ok else "error"
    text = f"{tool}({args}) -> {'ok' if ok else ev.get('error', 'fail')}"
    return f"tool:{tool}:{kind}:{args}", kind, text


DEFAULT_ADAPTERS = {"bus": bus_key, "mailbox": mailbox_key, "toollog": toollog_key}


# ---------- фильтр потока ----------

@dataclass
class StreamFilter:
    """Один фильтр на один поток. Внутри - привыкание; статистика - снаружи."""

    name: str = "stream"
    hab: Habituation = field(default_factory=Habituation)
    key_fn: object = None
    seen: int = 0
    emitted: int = 0
    by_kind_emitted: dict = field(default_factory=dict)
    by_kind_total: dict = field(default_factory=dict)

    def feed(self, event: dict, ts: float | None = None) -> Verdict:
        fn = self.key_fn or (lambda e: (f"{self.name}:{signature(str(e))[:60]}", "info", str(e)))
        key, kind, text = fn(event)
        v = self.hab.observe(key, kind=kind, ts=ts, text=text)
        self.seen += 1
        self.by_kind_total[kind] = self.by_kind_total.get(kind, 0) + 1
        if v.emit:
            self.emitted += 1
            self.by_kind_emitted[kind] = self.by_kind_emitted.get(kind, 0) + 1
        return v

    def feed_many(self, events, dt: float | None = None, t0: float = 0.0):
        """Пакет событий; dt - интервал между ними (детерминированные прогоны)."""
        t = t0
        out = []
        for e in events:
            out.append(self.feed(e, ts=t))
            if dt:
                t += dt
        return out

    def report(self) -> dict:
        r = dict(self.hab.report())
        r.update({"stream": self.name, "seen": self.seen, "emitted": self.emitted,
                  "kept_pct": round(100.0 * self.emitted / max(1, self.seen), 1),
                  "by_kind_total": dict(self.by_kind_total),
                  "by_kind_emitted": dict(self.by_kind_emitted)})
        return r


def make_filter(name: str = "bus", **kwargs) -> StreamFilter:
    """Собрать фильтр по имени потока: 'bus', 'mailbox', 'toollog' или свой маппер."""
    key_fn = DEFAULT_ADAPTERS.get(name, kwargs.pop("key_fn", None))
    return StreamFilter(name=name, key_fn=key_fn, **kwargs)
