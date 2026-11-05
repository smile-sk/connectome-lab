"""dynamics.py - сеть, собранная из РЕАЛЬНОГО коннектома C. elegans, во времени.

Зачем: E1/E2 работали со статикой (граф, потоки событий). Здесь другой вопрос -
если по этой топологии пустить сигнал, получится ли поведение, которое биологи уже
описали (носовое касание → отползти), и получится ли ПРИВЫКАНИЕ на уровне самой
сети, а не внешнего фильтра.

ПОЧЕМУ МОДЕЛЬ СТАВОК (rate), А НЕ СПАЙКОВ - это биология, а не упрощение ради кода:
нейроны C. elegans работают ГРАДИЕНТНЫМИ потенциалами, спайков у них нет. Сначала
я написала LIF и получила честный нуль: импульс от ASH до AVA не доходил (одна связь
из 87 входов AVAL - это 1/660 веса, до порога не дотянуть). Ставка ближе к реальности.

ЧТО ЕСТЬ: граф как в данных Cook 2019 (4681 химических + 1359 gap); знак синапса из
режимов sign_mode; активация x ∈ [-1,1], x ← (1−α)x + α·tanh(вход); нормировка входа
'sqrt'/'sum'/'none'; use-dependent depression (субстрат привыкания); мышцы - пассивные
интеграторы; группы чтения: назад DA*/VA*, вперёд DB*/VB*.

ЧЕГО НЕТ - ЧЕСТНО: весов синапсов (в данных только ЧИСЛО контактов, оно и берётся за
вес); знаков синапсов (их в датасете нет - беру грубый приор по медиатору, от него
зависит СИЛА, но не направление рефлекса, см. E3); градиентной динамики как у живой
клетки, нейропептидов, механики тела. Числа сравнивать МЕЖДУ СОБОЙ, не с вольтметром.

Два движка, один результат: numpy-путь (быстрый) и pure-python (эталон), сверяются
в e3_dynamics.py. Если numpy нет - работает медленный, ничего не падает.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass

try:                                    # numpy - ускоритель, не требование
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

# --- знаки синапсов: в данных их НЕТ, отсюда три честных приора --------------------
SIGN_MODES = {
    # «по учебнику медиатора»: ACh возбуждает, Glu и GABA тормозят
    "transmitter": {"Acetylcholine": 1.0, "Glutamate": -1.0, "GABA": -1.0,
                    "Dopamine": 0.25, "Serotonin": 0.25, "Octopamine": 0.25, "Tyramine": 0.25},
    # «глутамат возбуждает» (как в escape-цепи ASH→AVA), тормозит только GABA - приор по умолчанию
    "ach_gaba": {"Acetylcholine": 1.0, "Glutamate": 1.0, "GABA": -1.0,
                 "Dopamine": 0.25, "Serotonin": 0.25, "Octopamine": 0.25, "Tyramine": 0.25},
    # «торможения нет вовсе» - проверка, держится ли картина без знаков
    "excit_all": {"Acetylcholine": 1.0, "Glutamate": 1.0, "GABA": 1.0,
                  "Dopamine": 1.0, "Serotonin": 1.0, "Octopamine": 1.0, "Tyramine": 1.0},
}
UNKNOWN_SIGN = 0.5

_BW = re.compile(r"^([dv])BWM([LR])(\d+)$")   # dBWML10 - мышца тела, дорсально-левая №10
_PAD = re.compile(r"^([A-Za-z]+)(\d+)$")      # DA01 (граф) ↔ DA1 (owmeta)

BACKWARD_MOTOR = re.compile(r"^(DA|VA|AS)\d+$")
FORWARD_MOTOR = re.compile(r"^(DB|VB)\d+$")


def norm_name(name: str) -> str:
    """DA01 → DA1, dBWML10 → MDL10 (имя мышцы в owmeta), ASHL → ASHL."""
    name = (name or "").strip()
    m = _BW.match(name)
    if m:
        return f"M{'D' if m.group(1) == 'd' else 'V'}{m.group(2)}{int(m.group(3)):02d}"
    m = _PAD.match(name)
    return m.group(1) + str(int(m.group(2))) if m else name


@dataclass
class Neuron:
    name: str                 # как в графе (DA01, ASHL)
    key: str                  # нормализованное (DA1)
    klass: str = "unknown"    # sensory / interneuron / motor / muscle / other
    transmitters: tuple = ()
    is_muscle: bool = False

    @property
    def role(self) -> str:
        if self.is_muscle:
            return "muscle"
        if BACKWARD_MOTOR.match(self.name):
            return "motor_back"
        if FORWARD_MOTOR.match(self.name):
            return "motor_fwd"
        return self.klass


@dataclass
class Edge:
    src: int
    dst: int
    weight: float


class Connectome:
    """Граф + паспорта клеток. Только чтение данных, ноль симуляции."""

    def __init__(self):
        self.neurons: list[Neuron] = []
        self.index: dict[str, int] = {}
        self.chem: list[Edge] = []
        self.gap: list[Edge] = []
        self.passthrough: list[str] = []

    @classmethod
    def load(cls, edgelist: str, owmeta: str) -> "Connectome":
        self = cls()
        ow = json.load(open(owmeta, encoding="utf-8"))
        n_info, m_info = ow.get("neuron_info", {}), ow.get("muscle_info", {})
        rows = []
        with open(edgelist, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
        names, seen = [], set()
        for r in rows:
            for n in (r["Source"], r["Target"]):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        for n in names:
            key = norm_name(n)
            info, mus = n_info.get(key), m_info.get(key)
            if info is not None:
                kl = info[1] or []
                self.neurons.append(Neuron(n, key, kl[0] if kl else "unknown",
                                           tuple(info[3] or ()), False))
            elif mus is not None:
                self.neurons.append(Neuron(n, key, "muscle", (), True))
            else:
                self.passthrough.append(n)
                self.neurons.append(Neuron(n, key, "other", (), False))
        self.index = {ne.name: i for i, ne in enumerate(self.neurons)}
        chem, gap = {}, {}
        for r in rows:
            a, b = self.index[r["Source"]], self.index[r["Target"]]
            w = float(r["Weight"] or 1)
            if r["Type"] == "chemical":
                chem[(a, b)] = chem.get((a, b), 0.0) + w
            else:
                k = (a, b) if a <= b else (b, a)
                gap[k] = gap.get(k, 0.0) + w
        self.chem = [Edge(a, b, w) for (a, b), w in chem.items()]
        self.gap = [Edge(a, b, w) for (a, b), w in gap.items()]
        return self

    def ids(self, pattern: str) -> list[int]:
        rx = re.compile(pattern)
        return [i for i, n in enumerate(self.neurons) if rx.match(n.name)]

    def group(self, pattern: str) -> list[int]:
        return self.ids(pattern)

    def in_weight(self, idx: int) -> float:
        return sum(e.weight for e in self.chem if e.dst == idx)

    def stats(self) -> dict:
        classes: dict = {}
        for n in self.neurons:
            classes[n.role] = classes.get(n.role, 0) + 1
        return {"nodes": len(self.neurons), "chemical": len(self.chem), "gap": len(self.gap),
                "synapses": sum(e.weight for e in self.chem), "classes": classes,
                "passthrough": len(self.passthrough)}

    def subgraph_names(self, pat: str, hops: int = 1) -> list[str]:
        """Окрестность узлов под regexp - это и есть форма тела «из данных»."""
        cur, keep = set(self.ids(pat)), set(self.ids(pat))
        for _ in range(max(0, hops - 1)):
            nxt = set()
            for e in self.chem + self.gap:
                if e.src in cur:
                    nxt.add(e.dst)
                if e.dst in cur:
                    nxt.add(e.src)
            nxt -= keep
            keep |= nxt
            cur = nxt
        return [self.neurons[i].name for i in sorted(keep)]


@dataclass
class NetParams:
    dt: float = 1.0            # мс
    alpha: float = 0.35        # доля обновления активации за шаг
    gain: float = 0.5          # масштаб синаптического входа (подобран, см. E3)
    gap_gain: float = 1.0
    ext_gain: float = 1.5      # масштаб внешнего стимула
    muscle_gain: float = 2.0
    norm: str = "sqrt"         # 'sqrt' | 'sum' | 'none'
    depress: float = 0.0       # 0 = синапс не устаёт, 0.4 = сильно
    tau_rec: float = 8000.0    # мс, восстановление синапса (субстрат привыкания)
    tau_dep: float = 200.0     # мс, шкала самой усталости
    noise: float = 0.0
    seed: int = 7


class Network:
    """Градиентная (rate) сеть по коннектому. Воспроизводима: только seed и данные."""

    def __init__(self, connectome: Connectome, params: NetParams | None = None,
                 sign_mode: str = "ach_gaba", weighted: bool = True,
                 fast: bool = True):
        self.c = connectome
        self.p = params or NetParams()
        self.sign_mode = sign_mode
        self.weighted = weighted
        self.sign_map = SIGN_MODES[sign_mode]
        self.np = _np if (fast and _np is not None) else None
        n = len(connectome.neurons)
        self.x = [0.0] * n
        self.rel = [1.0] * n
        self.t = 0.0
        self.silenced: set[int] = set()
        self.muscle_ids = [i for i, ne in enumerate(connectome.neurons) if ne.is_muscle]
        self._rng = _Rng(self.p.seed)

        in_w = [0.0] * n
        for e in connectome.chem:
            in_w[e.dst] += e.weight

        def denom(w_sum: float) -> float:
            if self.p.norm == "sum":
                return w_sum or 1.0
            if self.p.norm == "sqrt":
                return math.sqrt(w_sum) if w_sum > 0 else 1.0
            return 1.0

        self.chem_f = [(e.src, e.dst,
                        (e.weight if weighted else 1.0) / denom(in_w[e.dst]),
                        self._sign_of(e.src)) for e in connectome.chem]
        deg = [0.0] * n
        for e in connectome.gap:
            deg[e.src] += e.weight
            deg[e.dst] += e.weight
        self.gap_f = [(e.src, e.dst, e.weight / denom(deg[e.src])) for e in connectome.gap]
        self.gap_b = [(e.dst, e.src, e.weight / denom(deg[e.dst])) for e in connectome.gap]
        self._build_fast(n)

    def _sign_of(self, src: int) -> float:
        ne = self.c.neurons[src]
        if ne.is_muscle:
            return 1.0
        for t in ne.transmitters:
            if t in self.sign_map:
                return self.sign_map[t]
        return 1.0 if self.sign_mode == "excit_all" else UNKNOWN_SIGN

    def _build_fast(self, n: int):
        if self.np is None:
            return
        np = self.np
        W = np.zeros((n, n), dtype=float)
        for s, d, w, sg in self.chem_f:
            W[d, s] += w * sg
        G = np.zeros((n, n), dtype=float)
        for a, b, w in self.gap_f:
            G[a, b] += w
        for a, b, w in self.gap_b:
            G[b, a] += w
        self.W = W * self.p.gain
        self.G = G * self.p.gap_gain
        self.Grow = G.sum(axis=1) * self.p.gap_gain
        self.xv = np.zeros(n, dtype=float)
        self.relv = np.ones(n, dtype=float)
        self.muscle_mask = np.zeros(n, dtype=bool)
        self.muscle_mask[self.muscle_ids] = True
        self.alive = None

    # -- шаг --------------------------------------------------------------------
    def step(self, external: dict[int, float] | None = None) -> None:
        if self.np is not None:
            self._step_np(external)
        else:
            self._step_py(external)
        self.t += self.p.dt

    def _step_py(self, external):
        p = self.p
        n = len(self.x)
        drv = [0.0] * n
        if external:
            for k, val in external.items():
                if k not in self.silenced:
                    drv[k] += p.ext_gain * val
        if p.noise:
            for i in range(n):
                drv[i] += (self._rng.next() - 0.5) * 2.0 * p.noise
        x, rel = self.x, self.rel
        for src, dst, w, s in self.chem_f:
            if src in self.silenced or dst in self.silenced:
                continue
            drv[dst] += p.gain * w * s * x[src] * rel[src]
        for a, b, w in self.gap_f:
            if a in self.silenced or b in self.silenced:
                continue
            drv[a] += p.gap_gain * w * (x[b] - x[a])
        for a, b, w in self.gap_b:
            if a in self.silenced or b in self.silenced:
                continue
            drv[b] += p.gap_gain * w * (x[a] - x[b])
        a_, rec, dep = p.alpha, p.dt / p.tau_rec, p.depress * p.dt / p.tau_dep
        for i in range(n):
            if i in self.silenced:
                x[i] = 0.0
                continue
            if i in self.muscle_ids:
                x[i] += a_ * (-x[i] + math.tanh(p.muscle_gain * drv[i]))
            else:
                x[i] += a_ * (-x[i] + math.tanh(drv[i]))
            r = rel[i] + (1.0 - rel[i]) * rec - dep * abs(x[i]) * rel[i]
            rel[i] = 0.05 if r < 0.05 else (1.0 if r > 1.0 else r)

    # -- быстрый путь (numpy) ---------------------------------------------------
    def _step_np(self, external):
        np, p = self.np, self.p
        drive = self.W @ (self.xv * self.relv) + (self.G @ self.xv - self.Grow * self.xv)
        if external:
            for k, val in external.items():
                if k not in self.silenced:
                    drive[k] += p.ext_gain * val
        if p.noise:
            drive += (np.random.default_rng(int(self.t) + p.seed).random(len(self.xv)) - 0.5) * 2.0 * p.noise
        tgt = np.tanh(np.where(self.muscle_mask, p.muscle_gain * drive, drive))
        self.xv += p.alpha * (tgt - self.xv)
        if self.silenced:
            self.xv[list(self.silenced)] = 0.0
        rec, dep = p.dt / p.tau_rec, p.depress * p.dt / p.tau_dep
        self.relv += (1.0 - self.relv) * rec - dep * np.abs(self.xv) * self.relv
        np.clip(self.relv, 0.05, 1.0, out=self.relv)

    def run(self, steps: int, external: dict[int, float] | None = None, schedule=None) -> None:
        for _ in range(steps):
            self.step(schedule(self.t) if schedule is not None else external)

    # -- чтение -----------------------------------------------------------------
    def vec(self) -> list[float]:
        return list(self.xv) if self.np is not None else list(self.x)

    def mean_x(self, ids: list[int]) -> float:
        ids = [i for i in ids if i not in self.silenced]
        if not ids:
            return 0.0
        v = self.xv if self.np is not None else self.x
        return float(sum(v[i] for i in ids)) / len(ids)

    def muscle_effort(self, motor_re: str | None = None) -> float:
        ids = self.muscle_ids
        if motor_re:
            rx = re.compile(motor_re)
            keep = {e.dst for e in self.c.chem
                    if self.c.neurons[e.dst].is_muscle and rx.match(self.c.neurons[e.src].name)}
            ids = [i for i in ids if i in keep]
        ids = [i for i in ids if i not in self.silenced]
        return self.mean_x(ids) if ids else 0.0

    def rest(self, ms: float):
        """Пауза: сеть в покое, восстанавливается только синапс (аналитически).

        Полная симуляция паузы 20 с - это 20 000 шагов; здесь то же самое по формуле
        r ← 1−(1−r)·exp(−t/τ_rec). Поведение то же: в покое активаций нет, устаёт только синапс.
        """
        k = math.exp(-ms / self.p.tau_rec)
        if self.np is not None:
            self.relv = 1.0 - (1.0 - self.relv) * k
        self.rel = [1.0 - (1.0 - r) * k for r in self.rel]
        self.t += ms

    def reset(self, keep_plasticity: bool = False):
        if self.np is not None:
            self.xv[:] = 0.0
            if not keep_plasticity:
                self.relv[:] = 1.0
        self.x = [0.0] * len(self.x)
        self.t = 0.0
        if not keep_plasticity:
            self.rel = [1.0] * len(self.rel)

    def silence(self, ids):
        self.silenced = set(ids)
        if self.np is not None and self.silenced:
            idx = list(self.silenced)
            self.xv[idx] = 0.0


class _Rng:
    """Крошечный детерминированный ГПСЧ - воспроизводимость без random."""

    def __init__(self, seed: int):
        self.s = (seed * 2654435761) & 0xFFFFFFFF or 1

    def next(self) -> float:
        self.s = (1103515245 * self.s + 12345) & 0xFFFFFFFF
        return self.s / 0xFFFFFFFF
