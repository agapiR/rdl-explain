"""Smoke tests and integration tests for estimate_deviation_from_determinacy.

Run with:
    python src/rdl_explain/explain/test_eval.py

Smoke tests: all external I/O (perturb_instance, torch.save/load, os.remove)
is patched so they run without real data files or model checkpoints.

Integration tests: use real HeteroData, real perturb_instance, and a simple
model that reads a numerical feature, so we can assert devΔ is ~0 when the
signal feature is retained and >0 when it is perturbed.
"""

import os
import sys
import math
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

_root = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

from relbench.base import TaskType
from rdl_explain.explain.eval import estimate_deviation_from_determinacy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_INSTANCES = 20   # total number of prediction instances
BATCH_SIZE   = 10  # instances per loader batch
N_BATCHES    = N_INSTANCES // BATCH_SIZE
N_SAMPLES    = 5   # perturbation draws S


def make_loader(n_batches: int) -> list:
    """Fake loader: list of mock batches that support .to()."""
    batches = []
    for _ in range(n_batches):
        b = MagicMock()
        b.to.return_value = b
        batches.append(b)
    return batches


def make_mock_data(node_types=('entity',), edge_types=()):
    """Fake HeteroData supporting node_types, edge_types, and attribute setting."""
    data = MagicMock()
    data.node_types = list(node_types)
    data.edge_types = list(edge_types)
    # torch.save will try to serialize this; patch torch.save to avoid issues
    return data


def make_mock_model(outputs: list) -> MagicMock:
    """Fake model whose calls consume successive tensors from `outputs`."""
    model = MagicMock()
    model.eval.return_value = None
    model.side_effect = outputs   # each __call__ pops the next item
    return model


def run_with_patches(model, data, task, mask, loader_factory, **kwargs):
    """Run estimate_deviation_from_determinacy with all I/O patched out."""
    clean_data = MagicMock()
    clean_data.node_types = data.node_types
    clean_data.edge_types = data.edge_types

    with patch('rdl_explain.explain.eval.perturb_instance', side_effect=lambda d, *a, **kw: d), \
         patch('rdl_explain.explain.eval.torch.save'),                                            \
         patch('rdl_explain.explain.eval.torch.load', return_value=clean_data),                   \
         patch('rdl_explain.explain.eval.os.remove'),                                             \
         patch('rdl_explain.explain.eval.gc.collect'):
        return estimate_deviation_from_determinacy(
            model=model, data=data, task=task, mask=mask, loader_factory=loader_factory, **kwargs
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shapes_binary_classification():
    """All seven return values have correct types and shapes."""
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    # total calls = 2 * N_SAMPLES * N_BATCHES (original + perturbed per draw)
    n_calls = N_BATCHES * 2 * N_SAMPLES
    outputs = [torch.randn(BATCH_SIZE, 1) for _ in range(n_calls)]
    model  = make_mock_model(outputs)
    data   = make_mock_data()
    loader = make_loader(N_BATCHES)

    dev_delta, dev_std, per_inst_dev, per_inst_std, per_sample, orig_preds, pert_preds = run_with_patches(
        model, data, task, mask={}, loader_factory=lambda: loader,
        num_samples=N_SAMPLES, prediction_type='soft',
    )

    assert isinstance(dev_delta, float),  "dev_delta must be float"
    assert isinstance(dev_std,   float),  "dev_std must be float"
    assert per_inst_dev.shape  == (N_INSTANCES,), f"expected ({N_INSTANCES},), got {per_inst_dev.shape}"
    assert per_inst_std.shape  == (N_INSTANCES,), f"expected ({N_INSTANCES},), got {per_inst_std.shape}"
    assert per_sample.shape    == (N_SAMPLES,),   f"expected ({N_SAMPLES},), got {per_sample.shape}"
    assert orig_preds.shape    == (N_SAMPLES, N_INSTANCES), f"expected ({N_SAMPLES}, {N_INSTANCES}), got {orig_preds.shape}"
    assert pert_preds.shape    == (N_SAMPLES, N_INSTANCES), f"expected ({N_SAMPLES}, {N_INSTANCES}), got {pert_preds.shape}"
    print("PASS: test_output_shapes_binary_classification")


def test_dev_delta_in_range_binary_classification():
    """devΔ and per-instance values lie in [0, 1] for binary classification."""
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    n_calls = N_BATCHES * 2 * N_SAMPLES
    outputs = [torch.randn(BATCH_SIZE, 1) for _ in range(n_calls)]

    dev_delta, dev_std, per_inst_dev, per_inst_std, per_sample, orig_preds, _ = run_with_patches(
        make_mock_model(outputs), make_mock_data(), task, mask={},
        loader_factory=lambda: make_loader(N_BATCHES), num_samples=N_SAMPLES,
    )

    assert 0.0 <= dev_delta <= 1.0,             f"dev_delta={dev_delta} out of [0,1]"
    assert dev_std >= 0.0,                       f"dev_std={dev_std} is negative"
    assert np.all(per_inst_dev >= 0),            "per_instance_dev contains negatives"
    assert np.all(per_inst_dev <= 1),            "per_instance_dev exceeds 1"
    assert np.all(orig_preds  >= 0),             "orig_preds (probs) contains negatives"
    assert np.all(orig_preds  <= 1),             "orig_preds (probs) exceeds 1"
    print("PASS: test_dev_delta_in_range_binary_classification")


def test_aggregate_consistency():
    """dev_delta == mean of per_sample_mean_distances == mean of per_instance_dev."""
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    n_calls = N_BATCHES * 2 * N_SAMPLES
    outputs = [torch.randn(BATCH_SIZE, 1) for _ in range(n_calls)]

    dev_delta, dev_std, per_inst_dev, _, per_sample, *_ = run_with_patches(
        make_mock_model(outputs), make_mock_data(), task, mask={},
        loader_factory=lambda: make_loader(N_BATCHES), num_samples=N_SAMPLES,
    )

    np.testing.assert_allclose(dev_delta, per_sample.mean(),    rtol=1e-5)
    np.testing.assert_allclose(dev_delta, per_inst_dev.mean(),  rtol=1e-5)
    expected_std = per_sample.std() / math.sqrt(N_SAMPLES)
    np.testing.assert_allclose(dev_std, expected_std, rtol=1e-5)
    print("PASS: test_aggregate_consistency")


def test_zero_devdelta_when_predictions_unchanged():
    """If perturbed predictions equal original, devΔ should be 0."""
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    # Same fixed logits for every call → identical predictions before and after
    fixed_logit = torch.zeros(BATCH_SIZE, 1)   # sigmoid(0) = 0.5 for all
    n_calls = N_BATCHES * 2 * N_SAMPLES
    outputs = [fixed_logit.clone() for _ in range(n_calls)]

    dev_delta, dev_std, per_inst_dev, per_inst_std, per_sample, *_ = run_with_patches(
        make_mock_model(outputs), make_mock_data(), task, mask={},
        loader_factory=lambda: make_loader(N_BATCHES), num_samples=N_SAMPLES,
    )

    np.testing.assert_allclose(dev_delta,   0.0, atol=1e-6)
    np.testing.assert_allclose(per_inst_dev, 0.0, atol=1e-6)
    np.testing.assert_allclose(per_inst_std, 0.0, atol=1e-6)
    print("PASS: test_zero_devdelta_when_predictions_unchanged")


def test_regression_smape_range():
    """SMAPE/2 distance for regression output lies in [0, 1]."""
    task = SimpleNamespace(
        task_type=TaskType.REGRESSION,
        entity_table='entity',
    )
    n_calls = N_BATCHES * 2 * N_SAMPLES
    # Positive-valued regression outputs to exercise SMAPE
    outputs = [torch.rand(BATCH_SIZE, 1) * 10 for _ in range(n_calls)]

    dev_delta, dev_std, per_inst_dev, _, per_sample, *_ = run_with_patches(
        make_mock_model(outputs), make_mock_data(), task, mask={},
        loader_factory=lambda: make_loader(N_BATCHES), num_samples=N_SAMPLES,
    )

    assert 0.0 <= dev_delta <= 1.0, f"regression dev_delta={dev_delta} out of [0,1]"
    assert np.all(per_inst_dev >= 0) and np.all(per_inst_dev <= 1)
    print("PASS: test_regression_smape_range")


def test_invalid_prediction_type_raises():
    """'ground_truth' prediction_type raises ValueError."""
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    try:
        run_with_patches(
            make_mock_model([]), make_mock_data(), task, mask={},
            loader_factory=lambda: [], num_samples=1, prediction_type='ground_truth',
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert 'ground_truth' in str(e)
    print("PASS: test_invalid_prediction_type_raises")


def test_unsupported_task_type_raises():
    """Unsupported task types raise NotImplementedError."""
    task = SimpleNamespace(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table='entity',
    )
    try:
        run_with_patches(
            make_mock_model([]), make_mock_data(), task, mask={},
            loader_factory=lambda: [], num_samples=1,
        )
        raise AssertionError("Expected NotImplementedError was not raised")
    except NotImplementedError:
        pass
    print("PASS: test_unsupported_task_type_raises")


# ---------------------------------------------------------------------------
# Integration tests — real data, real perturb_instance
# ---------------------------------------------------------------------------

import torch_frame
from torch_geometric.data import HeteroData


class _SimpleTF:
    """Duck-typed TensorFrame sufficient for perturb_instance and our model.

    Stores feat_dict and col_names_dict exactly as torch_frame.TensorFrame would.
    Plain tensors in feat_dict support in-place slice assignment, which is what
    perturb_instance uses to permute column values.
    """
    def __init__(self, feat_dict: dict, col_names_dict: dict):
        self.feat_dict = feat_dict
        self.col_names_dict = col_names_dict
        self.num_rows = next(iter(feat_dict.values())).shape[0]


class _SimpleLazyLoader:
    """Iterable loader that lazily reads from data on every __iter__ call.

    Because perturb_instance modifies data's TensorFrame in-place, each new
    iteration sees the current (possibly perturbed) state of the features.
    """
    def __init__(self, data: HeteroData, entity_table: str, batch_size: int):
        self.data = data
        self.entity_table = entity_table
        self.batch_size = batch_size

    def __iter__(self):
        # Read the current feature tensor fresh on every iteration pass.
        features = self.data[self.entity_table].tf.feat_dict[torch_frame.numerical].clone()
        col_names = self.data[self.entity_table].tf.col_names_dict[torch_frame.numerical]
        N = features.shape[0]
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = HeteroData()
            batch[self.entity_table].tf = _SimpleTF(
                {torch_frame.numerical: features[start:end]},
                {torch_frame.numerical: col_names},
            )
            yield batch


class _SimpleModel(torch.nn.Module):
    """Model that returns the first numerical feature (signal) as a logit.

    Ignores any other features, so only permuting the signal column affects
    predictions. Noise column permutations have zero effect.
    """
    def forward(self, batch, entity_table):
        feat = batch[entity_table].tf.feat_dict[torch_frame.numerical]  # (n, n_cols)
        return feat[:, 0:1]  # signal column only, shape (n, 1)


def _make_integration_data(n_group: int = 10):
    """Build HeteroData with two numerical features: signal (0/1) and noise (random).

    Returns:
        data:       HeteroData with data['entity'].tf populated.
        col_names:  ['signal', 'noise']
        N:          total number of entities (2 * n_group)
    """
    N = 2 * n_group
    signal = torch.cat([torch.zeros(n_group), torch.ones(n_group)]).unsqueeze(1)  # (N,1)
    noise  = torch.rand(N, 1)
    features = torch.cat([signal, noise], dim=1)  # (N, 2)

    data = HeteroData()
    data['entity'].tf = _SimpleTF(
        {torch_frame.numerical: features},
        {torch_frame.numerical: ['signal', 'noise']},
    )
    return data, ['signal', 'noise'], N


def test_integration_retain_signal_gives_zero_devdelta():
    """Retaining the signal feature (perturbing only noise) → devΔ == 0.

    The model ignores noise, so permuting noise has no effect on predictions.
    devΔ must be exactly 0 for every perturbation draw.
    """
    n_group  = 10
    n_samples = 20
    data, _, N = _make_integration_data(n_group)
    model  = _SimpleModel()
    loader_factory = lambda: _SimpleLazyLoader(data, 'entity', batch_size=N)
    task   = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    # Retain signal (True = retained), perturb noise (False = perturbed)
    mask = {
        ('entity', 'signal'): torch.tensor(True,  dtype=torch.bool),
        ('entity', 'noise'):  torch.tensor(False, dtype=torch.bool),
    }

    dev_delta, _, per_inst_dev, _, per_sample, *_ = \
        estimate_deviation_from_determinacy(
            model=model, data=data, task=task, mask=mask, loader_factory=loader_factory,
            explanation_type='column',
            perturbation_strategy='permutation_joint',
            num_samples=n_samples,
            prediction_type='soft',
            device='cpu',
        )

    np.testing.assert_allclose(dev_delta,    0.0, atol=1e-6, err_msg="devΔ should be 0 when signal is retained")
    np.testing.assert_allclose(per_inst_dev, 0.0, atol=1e-6, err_msg="per-instance devΔ should be 0")
    np.testing.assert_allclose(per_sample,   0.0, atol=1e-6, err_msg="per-sample devΔ should be 0")
    print("PASS: test_integration_retain_signal_gives_zero_devdelta")


def test_integration_perturb_signal_gives_positive_devdelta():
    """Perturbing the signal feature (retaining only noise) → devΔ > 0.

    The model reads the signal column directly. After joint permutation of the
    signal column across 10 zeros and 10 ones, roughly half the entities receive
    a different value, producing nonzero prediction distances.

    Expected devΔ ≈ 0.5 × |sigmoid(1) - sigmoid(0)| ≈ 0.115.
    We assert devΔ > 0.05 (conservative lower bound over many samples).
    """
    n_group   = 10
    n_samples = 50
    data, _, N = _make_integration_data(n_group)
    model  = _SimpleModel()
    loader_factory = lambda: _SimpleLazyLoader(data, 'entity', batch_size=N)
    task   = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    # Perturb signal (False = perturbed), retain noise (True = retained)
    mask = {
        ('entity', 'signal'): torch.tensor(False, dtype=torch.bool),
        ('entity', 'noise'):  torch.tensor(True,  dtype=torch.bool),
    }

    dev_delta, dev_std, _, _, per_sample, *_ = \
        estimate_deviation_from_determinacy(
            model=model, data=data, task=task, mask=mask, loader_factory=loader_factory,
            explanation_type='column',
            perturbation_strategy='permutation_joint',
            num_samples=n_samples,
            prediction_type='soft',
            device='cpu',
        )

    assert dev_delta > 0.05, f"devΔ={dev_delta:.4f} should be >0.05 when signal is perturbed"
    assert dev_delta < 1.0,  f"devΔ={dev_delta:.4f} should be <1.0"
    assert dev_std   >= 0,   "dev_std must be non-negative"
    # Per-sample means should vary (permutation is random) but all be positive
    assert np.all(per_sample >= 0), "all per-sample distances must be non-negative"
    print(f"PASS: test_integration_perturb_signal_gives_positive_devdelta  (devΔ={dev_delta:.4f})")


def test_integration_full_mask_gives_zero_devdelta():
    """Retaining ALL features → devΔ == 0 (nothing is perturbed)."""
    n_group   = 10
    n_samples = 10
    data, _, N = _make_integration_data(n_group)
    model  = _SimpleModel()
    loader_factory = lambda: _SimpleLazyLoader(data, 'entity', batch_size=N)
    task   = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table='entity',
    )
    mask = {
        ('entity', 'signal'): torch.tensor(True, dtype=torch.bool),
        ('entity', 'noise'):  torch.tensor(True, dtype=torch.bool),
    }

    dev_delta, *_ = estimate_deviation_from_determinacy(
        model=model, data=data, task=task, mask=mask, loader_factory=loader_factory,
        explanation_type='column',
        perturbation_strategy='permutation_joint',
        num_samples=n_samples,
        prediction_type='soft',
        device='cpu',
    )

    np.testing.assert_allclose(dev_delta, 0.0, atol=1e-6,
                               err_msg="devΔ should be 0 when all features are retained")
    print("PASS: test_integration_full_mask_gives_zero_devdelta")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_output_shapes_binary_classification,
        test_dev_delta_in_range_binary_classification,
        test_aggregate_consistency,
        test_zero_devdelta_when_predictions_unchanged,
        test_regression_smape_range,
        test_invalid_prediction_type_raises,
        test_unsupported_task_type_raises,
        # Integration tests — real data and real perturb_instance
        test_integration_retain_signal_gives_zero_devdelta,
        test_integration_perturb_signal_gives_positive_devdelta,
        test_integration_full_mask_gives_zero_devdelta,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)
