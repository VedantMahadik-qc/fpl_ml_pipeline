"""
FPL ML Pipeline — shared model classes
======================================

`TwoStageModel` lives here, and not in `fpl_train.py`, for one reason: pickle
records a class by the module it was defined in at dump time.

When `fpl_train.py` is run as a script its module name is `__main__`, so a class
defined there is pickled as `__main__.TwoStageModel`. Loading it from a
different entry point — `python fpl_transfers.py`, `python fpl_optimizer.py`, a
notebook — makes pickle look for `TwoStageModel` on *that* process's `__main__`,
which does not have it:

    AttributeError: Can't get attribute 'TwoStageModel' on <module '__main__'>

A top-level `from fpl_train import TwoStageModel` in the consumer happens to fix
it, because that binds the name into the consumer's `__main__`. But it only
works while the consumer IS `__main__`; import that consumer as a module and the
lookup fails again. Defining the class in an ordinary importable module removes
the failure mode entirely: pickle records `fpl_models.TwoStageModel`, which
resolves from anywhere.

`register_legacy_pickle_aliases()` runs on import and rescues model directories
that were already written with the `__main__` reference, so upgrading does not
force a retrain.
"""

from __future__ import annotations

import sys
from typing import Sequence, Tuple

import numpy as np


class TwoStageModel:
    """
    P(plays) x E[points | plays], behind a single `.predict()`.

    Exists so Phase 4, the transfer solver and anything else can keep calling
    `bundle["model"].predict(X)` without knowing the architecture is two-stage.
    The individual stages are also saved separately (`_prob` / `_pts`) for
    inspection and for callers that want the components.
    """

    def __init__(self, classifier, regressor, features: Sequence[str],
                 position: str = "", clip_low: float = 0.0) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.features = list(features)
        self.position = position
        self.clip_low = clip_low

    def predict_parts(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (probability of playing, expected points given he plays)."""
        prob = self.classifier.predict_proba(X)[:, 1]
        conditional = np.clip(self.regressor.predict(X), self.clip_low, None)
        return prob, conditional

    def predict(self, X) -> np.ndarray:
        prob, conditional = self.predict_parts(X)
        return prob * conditional

    def __repr__(self) -> str:  # pragma: no cover
        return f"TwoStageModel(position={self.position!r}, features={len(self.features)})"


def register_legacy_pickle_aliases() -> None:
    """
    Make pre-existing `__main__.TwoStageModel` pickles loadable from any process.

    Model directories written before the class moved here recorded the class as
    `__main__.TwoStageModel`. Pointing this process's `__main__` at the real
    class lets those bundles unpickle: the attribute names in the stored
    `__dict__` are unchanged, so the rehydrated object is identical.

    Never overwrites an existing attribute, so it cannot interfere with
    `fpl_train.py` running as `__main__`.
    """
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "TwoStageModel"):
        setattr(main, "TwoStageModel", TwoStageModel)


register_legacy_pickle_aliases()
