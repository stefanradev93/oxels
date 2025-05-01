import numpy as np
from typing import Sequence, Callable

class SequenceStrategy:
    def __init__(self,
                 augmentations: Sequence[Callable],
                 probabilities: Sequence[float] = None,
                 ordered = False,
                 at_most_one = False):
        
        """
        Parameters
        ----------
        augmentations : callable augmentations
        probabilities : list of prabilities to call each augmentation
        ordered       : whether the augmentations should be called in the provided order
        at_most_one   : whether just one qugmentation should be called at a time
        """

        self.augmentations = augmentations
        self.probabilities = probabilities
        
        self.ordered = ordered
        self.at_most_one = at_most_one
        assert(not (ordered and at_most_one))

    def __call__(self, im):
        if self.probabilities is None:
            active_indices = np.arange(len(self.augmentations))
        else:
            active_indices = np.where(np.random.random(len(self.probabilities)) < self.probabilities)[0]
       
        if not self.ordered and len(active_indices):
            active_indices = active_indices[np.random.choice(len(active_indices), len(active_indices), False)]

        for i in active_indices:
            im = self.augmentations[i](im).astype(np.float32)
            if self.at_most_one: break

        return im