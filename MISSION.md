# Mission: Training a competitive MTG agent

## Why

Build an agent that plays real Magic: The Gathering meta decks (Standard or Pauper)
well enough to be interesting, in a public repo people want to look at. Gameplay
first; sideboarding later.

## Success looks like

- Change any layer of `mtg-agent` without fear: rules engine, Gymnasium env,
  action masking, agents, training pipeline.
- Judge whether the 56-card engine can scale to real meta decks, and know what the
  alternatives cost.
- Train an agent that beats the heuristic baselines on a real archetype, and
  defend the result statistically.
- Ship a README that makes a stranger want to clone it.

## Constraints

- No deadline. Sessions will be irregular — lessons must be resumable.
- Strong Python. New to Gymnasium and the RL tooling stack.
- Knows MTG rules well; gaps are in edge cases (layers, priority minutiae,
  state-based actions), not fundamentals.

## Out of scope

- Deckbuilding / drafting — battle play first.
- Sideboarding — revisit once gameplay is solid.
- Causal RL theory (SCM, CGFA) beyond navigating this codebase. It's the inherited
  repo's research angle, not the mission.
