from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    original: np.ndarray
    processed: np.ndarray or None
    small: int
    medium: int
    big: int
    fly: int
