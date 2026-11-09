"""Loss-streak and session-loss pause adapted from the private paper guard.

In-memory, explicit reset only: restarting this educational backtest resets state.
This is not a live trading safety system.
"""

import math
from dataclasses import dataclass


@dataclass
class CircuitBreaker:
    max_consecutive_losses: int = 3
    max_session_loss: float = 0.05
    consecutive_losses: int = 0
    session_pnl: float = 0.0
    paused_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            self.max_consecutive_losses < 1
            or not math.isfinite(self.max_session_loss)
            or self.max_session_loss <= 0
        ):
            raise ValueError("Positive loss limits required")

    def allow_trade(self) -> bool:
        return self.paused_reason is None

    def record_result(self, pnl: float) -> None:
        if not math.isfinite(pnl):
            raise ValueError("Finite PnL required")
        self.session_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
        if self.paused_reason is None:
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.paused_reason = "max_consecutive_losses"
            elif self.session_pnl <= -self.max_session_loss:
                self.paused_reason = "max_session_loss"

    def reset(self) -> None:
        self.consecutive_losses = 0
        self.session_pnl = 0.0
        self.paused_reason = None
