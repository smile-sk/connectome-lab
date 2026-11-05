# ConnectomeLab

Research: real connectomes as a source of architectural ideas for agent harnesses.
The fly connectome (2026) showed that a brain is not a monolithic network but a
decentralized system: local circuits, sparse memory, hard inhibition, split fast reflex
and slow planning. The hypothesis here: some of these principles transfer to agent
architecture not as a metaphor but as numbers, taken from a concrete graph, with checks.

First subject: C. elegans (302 neurons, full connectome since 1986, every neuron named
with class, mediator and coordinates, and the full closed loop sensor -> interneuron ->
motor neuron -> muscle). The fly comes second (140k neurons).

The full research notes live in `docs/RESEARCH.md`.

## Layout

- `clab/` - graph building, dynamics, habituation, sparse memory, gates
- `scripts/` - experiments
- `tests/` - checks
