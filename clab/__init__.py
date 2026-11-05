"""clab - ConnectomeLab: архитектурные примитивы, снятые с реального коннектома.

Модуль НЕ знает ни про ai_ui, ни про AIUI-BUS, ни про почтовик. Он принимает
события (key, kind, ts) и отдаёт решения. Подключение к конкретному потоку -
через adapters.py, добавление нового потока = одна функция-маппер.

Примитивы:
  habituation.Habituation   - привыкание к повтору + восстановление + dishabituation
  sparse.SparseProjector    - разреженная проекция (клетки Кеньона → MBON-readout)
  gate.ContextGate          - латеральное торможение: вырезание шума из промпта
  state.StateVector         - компактный вектор состояния вместо пересказа истории
  adapters.StreamFilter     - единый вход для любых потоков (шина, почта, логи тулов)

Зависимостей нет - только стандартная библиотека Python 3.10+.
"""

from .habituation import Habituation, Verdict, Entry
from .sparse import SparseProjector, PrototypeReadout
from .gate import (ContextGate, GatedBlock, guess_kind, signature,
                   apply_policy, gate_result, TEXT_KEYS)
from .state import StateVector
from .adapters import StreamFilter, make_filter, DEFAULT_ADAPTERS

__all__ = [
    "Habituation", "Verdict", "Entry",
    "SparseProjector", "PrototypeReadout",
    "ContextGate", "GatedBlock", "guess_kind", "signature",
    "apply_policy", "gate_result", "TEXT_KEYS",
    "StateVector",
    "StreamFilter", "make_filter", "DEFAULT_ADAPTERS",
]

__version__ = "0.1.0"
