"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) *
            min(step^(-0.5),
                step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler used in the Transformer paper.

    Learning rate increases linearly during warmup,
    then decays proportionally to the inverse square root
    of the step number.

    Args:
        optimizer (torch.optim.Optimizer):
            Wrapped optimizer.

        d_model (int):
            Transformer embedding dimension.

        warmup_steps (int):
            Number of warmup steps.

        last_epoch (int):
            The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    # ============================================================
    # COMPUTE SCALE FACTOR
    # ============================================================

    def _get_lr_scale(self) -> float:
        """
        Compute Noam learning rate scale factor.

        Formula:
            d_model^(-0.5) *
            min(step^(-0.5),
                step * warmup_steps^(-1.5))
        """

        step = max(self.last_epoch + 1, 1)

        scale = (
            (self.d_model ** -0.5)
            * min(
                step ** -0.5,
                step * (self.warmup_steps ** -1.5),
            )
        )

        return scale

    # ============================================================
    # GET CURRENT LEARNING RATE
    # ============================================================

    def get_lr(self) -> list[float]:
        """
        Compute updated learning rates
        for all parameter groups.
        """

        scale = self._get_lr_scale()

        return [
            base_lr * scale
            for base_lr in self.base_lrs
        ]


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate LR schedule for visualization/testing.
    """

    dummy_model = torch.nn.Linear(1, 1)

    optimizer = optim.Adam(
        dummy_model.parameters(),
        lr=1.0,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
    )

    history = []

    for _ in range(total_steps):

        history.append(
            optimizer.param_groups[0]["lr"]
        )

        optimizer.step()
        scheduler.step()

    return history


# ============================================================
# VISUAL TEST
# ============================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20000

    lrs = get_lr_history(
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(lrs)

    plt.axvline(
        WARMUP_STEPS,
        color="red",
        linestyle="--",
        label=f"warmup={WARMUP_STEPS}",
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")

    plt.title(
        f"Noam Scheduler (d_model={D_MODEL})"
    )

    plt.legend()

    plt.tight_layout()
    plt.show()