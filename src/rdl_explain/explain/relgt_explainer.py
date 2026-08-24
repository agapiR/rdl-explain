"""Minimal RelGT-flavored explainer.

Adapts RDLExplainer to RelGT's token-sequence data interface
(`RelGTTokens` + `forward_to_explain(neighbor_types, node_indices, ...)`)
instead of GraphSAGE's HeteroData NeighborLoader batches.

Scope (intentionally narrow): only what cohort_discovery.ipynb needs —
`learn_masks` with `node_id_filter`. The 'avg' / 'permutation_*' elimination
strategies require `get_intermediate_encodings_for_replacement`, which is
HeteroData-flavored on the base class; for RelGT use `elimination_strategy='zero'`.
Fidelity estimation is also not ported here.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from relbench.base import EntityTask

from rdl_explain.explain.explainer import RDLExplainer
from rdl_explain.model.relgt.utils import RelGTTokens


class RelGTExplainer(RDLExplainer):
    """Minimal explainer subclass for RelGT.

    Overrides four methods from the base class so they speak the RelGT batch
    dict instead of HeteroData mini-batches:
      * create_loader        — wraps RelGTTokens + DataLoader; supports node_id_filter via Subset.
      * _unpack_batch        — static helper for the RelGT collated dict.
      * _auto_eps            — invokes forward_to_explain with the RelGT signature.
      * learn_masks_single_epoch — one training step in RelGT format.

    Everything else (initialize_masks, process_mask, mask_learning_loss_fn,
    etc.) is GraphSAGE-agnostic and inherited unchanged.
    """

    name = "node_relgt_explainer"

    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        data: HeteroData,
        task: EntityTask,
        K: int = 100,
        precomputed_dir: str = "./relgt-cache",
        num_dataset_workers: int = 4,
        num_loader_workers: int = 0,
    ):
        # Store before super().__init__ — base may call create_loader during setup.
        self.K = K
        self.precomputed_dir = precomputed_dir
        self.num_dataset_workers = num_dataset_workers
        self.num_loader_workers = num_loader_workers

        super().__init__(config, model, data, task)

        # Build (and precompute) RelGTTokens datasets per split. The HDF5
        # cache is reused if already on disk (shared with training).
        table_inputs = {"train": self.train_table,
                        "val":   self.val_table,
                        "test":  self.test_table}
        self.split_datasets: Dict[str, RelGTTokens] = {}
        for split in ("train", "val", "test"):
            ds = RelGTTokens(
                data=self.data, task=self.explanation_task, K=self.K,
                split=split, undirected=True, precompute=True,
                precomputed_dir=self.precomputed_dir,
                num_workers=self.num_dataset_workers,
            )
            # The base class' NodeTrainTableInput has the authoritative target
            # (the explanation_task may inject prediction columns the dataset
            # didn't see); patch it onto the dataset.
            ds.target = table_inputs[split].target
            self.split_datasets[split] = ds

    # ── Loader ───────────────────────────────────────────────────────────────

    def create_loader(self, data, split: str, shuffle: bool = False,
                      node_id_filter: Optional[Tensor] = None):
        """Build a DataLoader over the split's RelGTTokens dataset.

        If `node_id_filter` is given, wrap with a Subset of the positions whose
        seed-node local id is in the filter — this is the per-group filtering
        used by cohort mask learning.
        """
        full_ds = self.split_datasets[split]
        collate = full_ds.collate

        if node_id_filter is None:
            ds = full_ds
        else:
            keep_set = set(int(x) for x in node_id_filter.cpu().tolist())
            node_idxs_list = (
                full_ds.node_idxs.cpu().tolist()
                if torch.is_tensor(full_ds.node_idxs)
                else list(full_ds.node_idxs)
            )
            keep_positions = [i for i, n in enumerate(node_idxs_list)
                              if int(n) in keep_set]
            if not keep_positions:
                raise ValueError(
                    f"node_id_filter selected 0 seeds in split={split}; "
                    f"check that the filter values are local node indices for "
                    f"{self.explanation_task.entity_table}."
                )
            ds = Subset(full_ds, keep_positions)

        return DataLoader(
            ds, batch_size=self.config.inference_batch_size, shuffle=shuffle,
            collate_fn=collate, num_workers=self.num_loader_workers,
            persistent_workers=(self.num_loader_workers > 0),
        )

    # ── Batch unpacking ──────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch, device):
        """Pull tensors out of a RelGTTokens collated batch dict."""
        return (
            batch["neighbor_types"].to(device),
            batch["node_indices"].to(device),
            batch["neighbor_hops"].to(device),
            batch["neighbor_times"].to(device),
            {"grouped_tfs":     batch["grouped_tfs"],
             "grouped_indices": batch["grouped_indices"],
             "flat_batch_idx":  batch["flat_batch_idx"],
             "flat_nbr_idx":    batch["flat_nbr_idx"]},
            batch["edge_index"].to(device),
            batch["batch"].to(device),
            batch["labels"].to(device) if batch["labels"] is not None else None,
        )

    # ── Auto-eps (single batch, no grad) ─────────────────────────────────────

    def _auto_eps(self, data_loader, mask, explanation_type,
                  elimination_strategy, default_feat_vector) -> float:
        all_params = torch.cat([p.detach() for p in mask.values()])
        reg_loss_0 = all_params.sigmoid().sum().item()

        batch = next(iter(data_loader))
        nt, ni, nh, ntime, gtf, ei, bv, tgt = self._unpack_batch(batch, self.device)
        with torch.no_grad():
            out = self.model_to_explain.forward_to_explain(
                explanation_type, mask, nt, ni, nh, ntime, gtf,
                edge_index=ei, batch=bv,
                elimination_strategy=elimination_strategy,
                uninformative_feat_vector=default_feat_vector,
            )
            out = out.view(-1) if out.size(-1) == 1 else out
            task_loss_0 = self._task_loss_fn()(out, tgt).item()

        eps = task_loss_0 / reg_loss_0 if reg_loss_0 > 0 else 1.0
        eps_rounded = self._round_eps(eps)
        print(f"[Auto-eps] task_loss_0={task_loss_0:.4f}, reg_loss_0={reg_loss_0:.2f} "
              f"→ eps={eps:.6f} → rounded={eps_rounded:.6g}")
        return eps_rounded

    # ── One epoch ────────────────────────────────────────────────────────────

    def learn_masks_single_epoch(
        self,
        loader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        mask: Dict,
        explanation_type: str = "column",
        elimination_strategy: str = "zero",
        default_feat_vector: Dict[NodeType, Tensor] = None,
    ) -> Tuple[Dict, float, float, float, torch.Tensor, torch.Tensor]:
        loss_acc = task_acc = reg_acc = count = 0
        out_list, gt_list = [], []
        for batch in loader:
            nt, ni, nh, ntime, gtf, ei, bv, tgt = self._unpack_batch(batch, self.device)
            optimizer.zero_grad()
            out = self.model_to_explain.forward_to_explain(
                explanation_type, mask, nt, ni, nh, ntime, gtf,
                edge_index=ei, batch=bv,
                elimination_strategy=elimination_strategy,
                uninformative_feat_vector=default_feat_vector,
            )
            out = out.view(-1) if out.size(-1) == 1 else out
            loss, task_loss, reg_loss = loss_fn(out, tgt, mask)
            loss.backward()
            optimizer.step()
            loss_acc += loss.detach().item() * out.size(0)
            task_acc += task_loss * out.size(0)
            reg_acc  += reg_loss  * out.size(0)
            count    += out.size(0)
            out_list.append(out.detach().cpu())
            gt_list.append(tgt.detach().cpu())

        return (mask,
                loss_acc / count,
                task_acc / count,
                reg_acc  / count,
                torch.cat(out_list, dim=0),
                torch.cat(gt_list,  dim=0))

    # ── Guards for not-yet-ported paths ──────────────────────────────────────
    #
    # Replacement vectors (used by 'avg' / 'permutation_*'): the base class'
    # `inference_to_get_intermediate_encodings` builds a per-node-type
    # TensorFrameDataLoader and calls `model.get_intermediate_encoding(stub_data,
    # node_type)`. RelGT's `get_intermediate_encoding` only reads
    # `batch.tf_dict[entity_table]`, the same interface as GraphSAGE — so the
    # path is model-agnostic. No override needed; inheriting from the base.
    #
    # Everything below is inherited-but-broken for RelGT: the base class walks
    # its loader assuming PyG mini-batches (`batch.to(device)`) and calls the
    # model as `model(batch, entity_table)`. RelGT loaders yield plain dicts and
    # `RelGT.forward` takes (neighbor_types, node_indices, neighbor_hops, ...),
    # so these raise here rather than failing deep inside the base class with a
    # confusing AttributeError/TypeError.
    #
    # Unblocking them is one shared piece of work: give the base class (and
    # `eval.py`) a `forward_fn` abstraction so the batch-unpacking + model-call
    # convention is supplied by the explainer instead of hard-coded. Tracked as
    # a follow-up; `_unpack_batch` above is the RelGT half of that seam.

    _PORT_HINT = (
        "This path assumes PyG HeteroData mini-batches and the "
        "`model(batch, entity_table)` calling convention; RelGT uses token-dict "
        "batches and a different forward signature. Porting it requires a "
        "`forward_fn` abstraction in RDLExplainer/eval.py — see the guard "
        "comment in relgt_explainer.py."
    )

    def estimate_devdelta(self, *args, **kwargs):
        # Base version delegates to eval.estimate_deviation_from_determinacy,
        # which does `batch.to(device)` / `model(batch, entity_table)` internally.
        raise NotImplementedError(
            f"RelGTExplainer does not support estimate_devdelta yet. {self._PORT_HINT}"
        )

    def inference_to_explain_predictions(self, *args, **kwargs):
        raise NotImplementedError(
            "RelGTExplainer does not support inference_to_explain_predictions yet. "
            f"{self._PORT_HINT}"
        )

    def estimate_fidelity(self, *args, **kwargs):
        # NOTE: `estimate_fidelity` / `_calculate_fidelity` were removed from
        # RDLExplainer in the devΔ migration; `estimate_devdelta` replaced them.
        # Kept as an explicit guard so older call sites get a clear message.
        raise NotImplementedError(
            "estimate_fidelity was replaced by estimate_devdelta during the devΔ "
            f"migration, and is not ported to RelGT. {self._PORT_HINT}"
        )
