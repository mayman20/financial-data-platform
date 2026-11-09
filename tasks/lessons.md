
## Demo review correction
**What:** The original demo displayed strategy gate failures and an unexplained flat exposure curve.
**Why:** Code correctness was demonstrated, but neither a positive detection control nor allocation-specific diagnostics made that visible to a skimming reviewer.
**How:** Pair a documented planted signal with a noise control, label guard pauses/resets, and evaluate allocation mechanics separately from return significance. Fix synthetic assumptions before evaluating outcomes; never tune the effect to obtain PASS.
