# MTG Agent Resources

## Knowledge

### RL tooling

- [Gymnasium — Env API](https://gymnasium.farama.org/api/env/)
  The contract: `reset()`, `step()`, `observation_space`, `action_space`.
  Use for: what an env is obligated to return.
- [Gymnasium — Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
  Five-minute orientation, including the `terminated` vs `truncated` split.
  Use for: first contact with the loop.
- [sb3-contrib — MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
  The PPO variant this repo trains with. Use for: how masks reach the policy —
  `ActionMasker`, `get_action_masks`, `MaskableEvalCallback`.
- [PettingZoo — SB3 Action Masked PPO for Connect Four](https://pettingzoo.farama.org/tutorials/sb3/connect_four/)
  Runnable masked-PPO-on-a-card-game tutorial. Use for: a working reference
  implementation to compare this repo against.

### Magic rules

- [Comprehensive Rules (Wizards, official)](https://magic.wizards.com/en/rules)
  Sections 405 (stack), 117 (priority), 704 (state-based actions), 613 (layers).
  Use for: settling any edge case the engine gets wrong.

### MTG + AI research

- [Causal Reinforcement Learning for Complex Card Games: A MTG Benchmark (arXiv 2605.06066)](https://arxiv.org/abs/2605.06066)
  **This repo's own paper.** The 3077-dim observation, 478-action space, five
  archetypes, and SCM come from here. Use for: why the code is shaped this way.
- [Learning With Generalised Card Representations for MTG (arXiv 2407.05879)](https://arxiv.org/pdf/2407.05879)
  Representing cards the model has never seen. Use for: scaling past a fixed
  56-card pool.

## Wisdom (Communities)

_Not yet explored — say the word and I'll stop suggesting them._

- [r/spikes](https://reddit.com/r/spikes) — competitive meta discussion.
  Use for: which decks actually define Standard/Pauper right now.
- **Forge** and **XMage** — the two mature, near-complete open-source MTG engines.
  Use for: reality-checking how hard full rules coverage really is; their issue
  trackers are where edge cases get argued.

## Gaps

- Which open-source engine is the best base for an RL agent (Forge vs XMage vs
  rolling our own). The pivotal decision for the mission. Needs research.
- Machine-readable Pauper/Standard meta decklists.
