
## Demo review correction
**What:** The original demo displayed strategy gate failures and an unexplained flat exposure curve.
**Why:** Code correctness was demonstrated, but neither a positive detection control nor allocation-specific diagnostics made that visible to a skimming reviewer.
**How:** Pair a documented planted signal with a noise control, label guard pauses/resets, and evaluate allocation mechanics separately from return significance. Fix synthetic assumptions before evaluating outcomes; never tune the effect to obtain PASS.

## Detection-rate presentation correction
**What:** The headline combined a detection-rate estimate with one primary-seed failure.
**Why:** A single illustrative trajectory was too easy to read as the experiment's result, and the original noise fixture did not hold the positive-control noise and horizon constant.
**How:** Lead with a predeclared effect-size table using identical seeds, noise, sessions, costs and gates; label zero effect as false-positive frequency. Keep primary-seed charts explicitly illustrative. Preserve coordinator ownership of README edits.
