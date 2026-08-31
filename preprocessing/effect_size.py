"""Effect size helpers backed by pingouin."""

from __future__ import annotations      # Allows annotations like tuple[np.ndarray, np.ndarray

from dataclasses import dataclass   # Allows using @dataclass decorator
from typing import Literal

import numpy as np
import pingouin as pg


EffectType = Literal['cohen', 'hedges']


@dataclass
class EffectSizeCalculator:
    """
    Compute effect sizes and confidence intervals for two groups.
    Automatically excludes NaNs from comparisons

    Public methods:
        cohens_d(): Compute and return Cohen's d.
        hedges_g(): Compute and return Hedges' g.
        confidence_interval(effect='cohen', confidence=0.95, alternative='two-sided'):
            Compute and return the confidence interval for the selected effect size.

    Args:
        x: First sample.
        y: Second sample.
        paired: Whether samples are paired.
    """

    x: np.ndarray           # Those are @dataclass type hinting
    y: np.ndarray           # Also auto adds self.x = x; self.y = y, etc. inside the constructor
    paired: bool = False

    def __init__(self, x, y, paired: bool = False):
        self.paired = bool(paired)
        self.x, self.y = self._prepare_samples(x, y, self.paired)

    @staticmethod
    def _prepare_samples(x, y, paired: bool) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()

        if paired:
            if x_arr.shape[0] != y_arr.shape[0]:
                raise ValueError('Paired samples must have the same length.')

            valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)
            x_arr = x_arr[valid]
            y_arr = y_arr[valid]
        else:
            x_arr = x_arr[~np.isnan(x_arr)]
            y_arr = y_arr[~np.isnan(y_arr)]

        if x_arr.size < 2 or y_arr.size < 2:
            raise ValueError('Each sample must contain at least two valid observations.')

        return x_arr, y_arr

    @property
    def nx(self) -> int:
        return int(self.x.size)

    @property
    def ny(self) -> int:
        return int(self.y.size)

    def cohens_d(self) -> float:
        """Return Cohen's d."""
        return float(pg.compute_effsize(self.x, self.y, paired=self.paired, eftype='cohen'))

    def hedges_g(self) -> float:
        """Return Hedges' g."""
        return float(pg.compute_effsize(self.x, self.y, paired=self.paired, eftype='hedges'))

    def confidence_interval(
        self,
        effect: EffectType = 'cohen',
        confidence: float = 0.95,
        alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
    ) -> tuple[float, float]:
        """Return confidence interval for Cohen's d or Hedges' g.

        Args:
            effect: Effect type to compute CI for ('cohen' or 'hedges').
            confidence: Confidence level in the (0, 1) interval.
            alternative: Tail for CI computation.
        """
        if effect not in {'cohen', 'hedges'}:
            raise ValueError("effect must be either 'cohen' or 'hedges'.")

        if not 0 < confidence < 1:
            raise ValueError('confidence must be between 0 and 1.')

        stat = self.cohens_d() if effect == 'cohen' else self.hedges_g()
        ci_low, ci_high = pg.compute_esci(
            stat=stat,
            nx=self.nx,
            ny=self.ny,
            paired=self.paired,
            eftype=effect,
            confidence=confidence,
            alternative=alternative,
        )
        return float(ci_low), float(ci_high)
