import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MDP:
    """
    Tiny finite MDP container.
    All actions are assumed to be available at all states.

    S: number of states
    A: number of (global) actions
    P: transition tensor with shape (S, A, S). P[s, a, s'] = Pr(S' = s' | s, a)
    R: immediate reward matrix with shape (S, A): r(s,a)
    beta: discount factor in (0,1)
    names_s / names_a: optional labels
    """
    S: int
    A: int
    P: np.ndarray
    R: np.ndarray
    beta: float
    names_s: Optional[List[str]] = None
    names_a: Optional[List[str]] = None


    def expected_value(self, s: int, a: int, V: np.ndarray) -> float:
        """E[V(S') | s, a]. (Assumes P row is valid; caller should avoid forbidden a.)"
        """
        return float(self.P[s, a, :] @ V)
