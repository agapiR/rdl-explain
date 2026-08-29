"""Turning learned (continuous) masks into discrete explanations.

Mask learning optimises a continuous relaxation: each explanation element
carries a real-valued logit, and ``sigmoid(logit) in [0, 1]`` is its importance.
An *explanation* is discrete, so the learned mask is thresholded (paper,
Section 5.2):

    m'_x = 1 if m_x >= delta, else 0

with ``delta = 0.1`` in the paper's experiments (Section 6). Elements with
``m'_x = 1`` are included in the explanation; the rest are omitted.

Two conventions matter and are easy to get wrong:

* ``learn_masks`` returns **logits** (it deliberately does not apply the
  sigmoid), so thresholding must activate first. Pass
  ``already_activated=True`` for a mask that is already in [0, 1].
* The perturbation and devDelta code requires **bool** tensors, where
  ``True`` = retained and ``False`` = perturbed. Float tensors of 0.0/1.0 are
  silently wrong there, because ``~`` on a float tensor raises and ``~`` on an
  int tensor yields nonzero (truthy) values for both states.

``discretize_mask`` handles both, so the usual flow is::

    mask, mask_vals, metrics = explainer.learn_masks(...)
    hard = discretize_mask(mask)                      # delta = 0.1
    dev  = explainer.estimate_devdelta(mask=hard, ...)
"""

from typing import Any, Dict, List, Union

import torch

#: Discretization threshold used in the paper's experiments (Section 6).
DEFAULT_DELTA = 0.1

__all__ = ["DEFAULT_DELTA", "discretize_mask", "mask_scores", "explanation_size"]


def _activate(value: torch.Tensor, already_activated: bool) -> torch.Tensor:
    value = value.detach()
    return value if already_activated else torch.sigmoid(value)


def _is_filter_mask(mask: Any) -> bool:
    """A 'filter' mask is a dict carrying its predicate alongside 'params'."""
    return isinstance(mask, dict) and "params" in mask and "values" in mask


def discretize_mask(
    mask: Union[Dict, List[Dict]],
    delta: float = DEFAULT_DELTA,
    already_activated: bool = False,
) -> Union[Dict, List[Dict]]:
    """Threshold a learned mask into a discrete (boolean) explanation.

    Args:
        mask: A learned mask, in any of the shapes ``learn_masks`` returns:
            a flat ``{element_key: tensor}`` dict ('column', 'fkpk', 'table',
            'layer-wise', ...), a 'filter' mask dict (with ``params`` /
            ``values``), or a list of 'filter' mask dicts.
        delta: Inclusion threshold; elements with importance ``>= delta`` are
            kept. Defaults to the paper's 0.1. The paper reports results are
            stable for delta in [0.05, 0.2] and sensitive to small variations
            outside it (Section 6.1).
        already_activated: Set True if ``mask`` already holds sigmoid outputs
            in [0, 1]. By default the mask is assumed to hold logits, matching
            ``learn_masks``, and a sigmoid is applied first.

    Returns:
        The same structure, with every mask tensor replaced by a ``torch.bool``
        tensor: ``True`` = element retained (in the explanation), ``False`` =
        element perturbed. This is the form
        ``estimate_devdelta`` / ``perturb_instance`` expect.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError(
            f"delta must be a probability in [0, 1] (got {delta}). It is "
            "compared against sigmoid(mask), not against the raw logit."
        )

    if isinstance(mask, (list, tuple)):
        return [discretize_mask(m, delta, already_activated) for m in mask]

    if _is_filter_mask(mask):
        out = dict(mask)
        out["params"] = {
            k: _activate(v, already_activated) >= delta
            for k, v in mask["params"].items()
        }
        return out

    return {
        key: _activate(value, already_activated) >= delta
        for key, value in mask.items()
    }


def mask_scores(
    mask: Dict,
    already_activated: bool = False,
) -> Dict[Any, float]:
    """Importance score in [0, 1] per explanation element, for inspection/plots.

    Mask tensors are single-element, so each is reduced to a Python float. Use
    ``discretize_mask`` (not this) to build a mask for devDelta evaluation.
    """
    return {
        key: float(_activate(value, already_activated).reshape(-1)[0])
        for key, value in mask.items()
    }


def explanation_size(bool_mask: Dict) -> int:
    """Number of explanation elements retained by a discretized mask.

    This is the paper's explanation size ``k``: retained data attributes for
    Projection, join conditions for FKJoin, predicates for Selection.
    """
    return sum(int(bool(v.any())) for v in bool_mask.values())
