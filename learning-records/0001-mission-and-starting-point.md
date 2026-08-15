# Mission established: competitive MTG bot on real meta decks

Train a strong MTG agent on Standard or Pauper meta decks, gameplay first,
sideboarding later. No deadline; a public repo is the artifact. This makes "does
the 56-card engine scale to real decks?" the central open question of the
workspace, not a footnote.

## Prior knowledge (stated)

- **Python: strong.** Don't teach language mechanics.
- **Gymnasium / RL tooling: new.** The current floor.
- **MTG rules: solid fundamentals, weak on edge cases.** Never explain what the
  stack or priority *is*. Teach how this engine models them and where it diverges
  from the CR (layers, SBAs, priority minutiae).

## Implications

- Bias lessons toward visible artifacts.
- The inherited causal/CGFA layer is out of scope except as navigation. Decide
  early whether to keep or strip it.
- Sessions are irregular; every lesson must stand alone.
