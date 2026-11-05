"""SparseProjector - разреженная проекция (грибовидные тела → readout).

Биология: ~50 000 входных волокон проецируются на ~2 000 клеток Кеньона
случайными разреженными связями, а те сжимаются в ~20 управляющих нейронов
(MBON), выдающих «опасно/полезно». Это НЕ обучение - это фиксированная
случайная проекция + чтение. Ровно то, что нужно харнессу: дешёвая
классификация потока без вызова LLM.

Инженерия: устойчивый хеш (hashlib.blake2b), а не hash() - встроенный хеш
в CPython солится на процесс, и код был бы разным между запусками.

Применение: превратить вывод инструмента или поток логов в разреженный код
(20 индексов из 2000) и прочитать его 4 классами: ok / error / noise / attention.
Дальше по коду сравниваются Jaccard-пересечением: «это тот же тип проблемы?»
"""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[a-zA-Zа-яА-Я_][a-zA-Zа-яА-Я0-9_\-]{1,}")


def tokens(text: str, limit: int = 400) -> list[str]:
    """Грубая токенизация без зависимостей: слова + числовые маркеры."""
    t = _WORD.findall(text.lower())
    if len(t) > limit:
        t = t[:limit]
    return t


class SparseProjector:
    """Разреженный код как замена эмбеддеру, когда эмбеддер избыточен."""

    def __init__(self, n_kc: int = 2000, active: int = 20, fan_in: int = 6,
                 seed: str = "clab"):
        self.n_kc = int(n_kc)
        self.active = max(1, int(active))
        self.fan_in = max(1, int(fan_in))
        self.seed = seed

    # ---------- проекция ----------

    def _idx(self, token: str, j: int) -> int:
        raw = f"{self.seed}|{token}|{j}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") % self.n_kc

    def encode(self, text_or_tokens) -> list[int]:
        """Текст (или список токенов) → отсортированный разреженный код индексов."""
        toks = text_or_tokens if isinstance(text_or_tokens, (list, tuple)) else tokens(text_or_tokens)
        acc: dict[int, float] = {}
        for t in toks:
            for j in range(self.fan_in):
                i = self._idx(t, j)
                acc[i] = acc.get(i, 0.0) + 1.0
        if not acc:
            return []
        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:self.active]
        return sorted(i for i, _ in top)

    # ---------- readout (MBON) ----------

    def readout(self, code: list[int], labels=("ok", "error", "noise", "attention"),
                seed: str | None = None) -> dict:
        """Фиксированные случайные веса → класс + отрыв. Без обучения."""
        seed = seed or self.seed
        scores = {}
        for c, lab in enumerate(labels):
            s = 0.0
            for i in code:
                h = hashlib.blake2b(f"{seed}|mbon|{c}|{i}".encode("utf-8"), digest_size=2).digest()
                s += 1.0 if (h[0] & 1) else -1.0
            scores[lab] = s / max(1.0, len(code) ** 0.5)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_v = ordered[0]
        second_v = ordered[1][1] if len(ordered) > 1 else 0.0
        return {"label": best, "score": round(best_v, 4),
                "margin": round(best_v - second_v, 4), "scores": scores}

    # ---------- сравнение ----------

    @staticmethod
    def overlap(a, b) -> float:
        """Jaccard по разреженным кодам: 1.0 - тот же код, 0.0 - ничего общего."""
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / float(len(sa | sb))


class PrototypeReadout:
    """Readout без обучения: ближайший прототип по пересечению разреженных кодов.

    Честно: это не «понимание» и не классификатор - это поиск похожести.
    Смысл в цене: разложить поток на 2-4 известных класса (норма/ошибка/шум)
    можно без вызова LLM и без эмбеддера - кодами из 20 индексов.

    Пример:
        pr = PrototypeReadout()
        pr.add("ok", "fs_read ok, 12 files")
        pr.add("error", "error: timeout after 120s")
        pr.predict("error: timeout after 300s")  -> {'label': 'error', ...}
    """

    def __init__(self, projector: SparseProjector | None = None, threshold: float = 0.12):
        self.sp = projector or SparseProjector()
        self.threshold = threshold
        self.protos: dict[str, list[list[int]]] = {}

    def add(self, label: str, text: str) -> list[int]:
        code = self.sp.encode(text)
        self.protos.setdefault(label, []).append(code)
        return code

    def scores(self, text_or_code) -> dict:
        code = text_or_code if isinstance(text_or_code, list) else self.sp.encode(text_or_code)
        out = {}
        for lab, codes in self.protos.items():
            out[lab] = max(SparseProjector.overlap(code, c) for c in codes)
        return out

    def predict(self, text_or_code) -> dict:
        sc = self.scores(text_or_code)
        if not sc:
            return {"label": "?", "score": 0.0, "margin": 0.0, "scores": {}}
        ordered = sorted(sc.items(), key=lambda kv: -kv[1])
        best, bv = ordered[0]
        sv = ordered[1][1] if len(ordered) > 1 else 0.0
        label = best if bv >= self.threshold else "unknown"
        return {"label": label, "score": round(bv, 3), "margin": round(bv - sv, 3),
                "scores": {k: round(v, 3) for k, v in sc.items()}}
