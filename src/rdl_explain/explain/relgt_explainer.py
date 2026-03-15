import os
import time
import gc

from typing import Dict, Tuple, List
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from relbench.base import EntityTask, TaskType

from rdl_explain.explain.explainer import RDLExplainer
from rdl_explain.explain.explain_utils import perturb_instance
from rdl_explain.model.relgt.utils import RelGTTokens


class RelGTExplainer(RDLExplainer):
    """
    Explainer subclass for the RelGT (Relational Graph Transformer) model.

    Overrides data loading and batch unpacking to work with RelGTTokens
    instead of PyG NeighborLoader.

    NOTE: This explainer currently only supports 'column' and 'table'
    explanation types. Other modalities (fkpk, filter, layer-wise, etc.)
    are left as future work.
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
        """
        Initialize the RelGTExplainer.

        Args:
            config: Explainer configuration.
            model: The RelGT model to explain.
            data: HeteroData graph.
            task: The relbench EntityTask.
            K: Number of neighbor tokens per seed node (must match model's sample_node_len).
            precomputed_dir: Directory for precomputed HDF5 neighbor token caches.
            num_dataset_workers: Number of workers for RelGTTokens precomputation.
            num_loader_workers: Number of workers for the DataLoader.
        """
        # Store RelGT-specific params before super().__init__ (which may use create_loader)
        self.K = K
        self.precomputed_dir = precomputed_dir
        self.num_dataset_workers = num_dataset_workers
        self.num_loader_workers = num_loader_workers

        super().__init__(config, model, data, task)

        # Build RelGTTokens datasets for each split.
        # We use the table inputs already built by _initialize_task_tables()
        # to ensure the target column is correctly resolved, since
        # RelGTTokens calls task.get_table() internally which may not
        # return tables with injected prediction columns (due to lru_cache
        # on the deep-copied task).
        table_inputs = {
            "train": self.train_table,
            "val": self.val_table,
            "test": self.test_table,
        }
        self.split_datasets = {}
        for split in ["train", "val", "test"]:
            ds = RelGTTokens(
                data=self.data,
                task=self.explanation_task,
                K=self.K,
                split=split,
                undirected=True,
                precompute=True,
                precomputed_dir=self.precomputed_dir,
                num_workers=self.num_dataset_workers,
            )
            # Override target from the authoritative table input
            ds.target = table_inputs[split].target
            self.split_datasets[split] = ds

    def create_loader(self, data: HeteroData, split: str, shuffle: bool = False):
        """Create a DataLoader wrapping RelGTTokens for the given split."""
        dataset = self.split_datasets[split]
        return DataLoader(
            dataset,
            batch_size=self.config.inference_batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate,
            num_workers=self.num_loader_workers,
            persistent_workers=self.num_loader_workers > 0,
        )

    @staticmethod
    def _unpack_batch(batch, device):
        """
        Unpack a RelGTTokens collated batch into the tensors expected by
        RelGT.forward / forward_to_explain.

        Returns:
            Tuple of (neighbor_types, node_indices, neighbor_hops,
                      neighbor_times, grouped_tf_dict, edge_index, batch_vec, labels)
        """
        neighbor_types = batch["neighbor_types"].to(device)
        node_indices   = batch["node_indices"].to(device)
        neighbor_hops  = batch["neighbor_hops"].to(device)
        neighbor_times = batch["neighbor_times"].to(device)
        edge_index     = batch["edge_index"].to(device)
        batch_vec      = batch["batch"].to(device)
        labels         = batch["labels"].to(device) if batch["labels"] is not None else None
        grouped_tf_dict = {
            "grouped_tfs":     batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx":  batch["flat_batch_idx"],
            "flat_nbr_idx":    batch["flat_nbr_idx"],
        }
        return neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index, batch_vec, labels

    def learn_masks_single_epoch(
        self,
        loader,
        loss_fn: callable,
        optimizer: torch.optim.Optimizer,
        mask: Dict,
        explanation_type: str = 'table',
        elimination_strategy: str = 'zero',
        default_feat_vector: Dict[NodeType, Tensor] = None,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Train the mask for one epoch using RelGT batch format."""
        out_list, gt_list = [], []
        loss_accum = count_accum = 0
        task_loss_accum = reg_loss_accum = 0

        for batch in loader:
            (neighbor_types, node_indices, neighbor_hops, neighbor_times,
             grouped_tf_dict, edge_index, batch_vec, targets) = self._unpack_batch(batch, self.device)

            optimizer.zero_grad()
            out = self.model_to_explain.forward_to_explain(
                explanation_type, mask,
                neighbor_types, node_indices, neighbor_hops, neighbor_times,
                grouped_tf_dict, edge_index=edge_index, batch=batch_vec,
                elimination_strategy=elimination_strategy,
                uninformative_feat_vector=default_feat_vector,
            )
            out = out.view(-1) if out.size(1) == 1 else out
            loss, task_loss, reg_loss = loss_fn(out, targets, mask)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * out.size(0)
            task_loss_accum += task_loss * out.size(0)
            reg_loss_accum += reg_loss * out.size(0)
            count_accum += out.size(0)
            out_list.append(out.detach().cpu())
            gt_list.append(targets.detach().cpu())

        output = torch.cat(out_list, dim=0)
        ground_truth = torch.cat(gt_list, dim=0)
        return mask, loss_accum / count_accum, task_loss_accum / count_accum, reg_loss_accum / count_accum, output, ground_truth

    @torch.no_grad()
    def estimate_fidelity(
        self,
        split: str,
        mask: Dict,
        explanation_type: str = 'column',
        perturbation_strategy: str = 'permutation_independent',
        num_samples: int = 1000,
    ) -> Tuple[Tensor, Tensor, List[int]]:
        """Estimate the fidelity of the explanation masks using RelGT inference."""

        # Collect explanation targets for the split
        loader = self.create_loader(self.data, split, shuffle=False)
        gt_list = []
        for batch in loader:
            (_, _, _, _, _, _, _, targets) = self._unpack_batch(batch, self.device)
            gt_list.append(targets.detach().cpu())
        targets = torch.cat(gt_list, dim=0).numpy()

        predictions_per_sample = np.empty((num_samples, len(targets)))

        start_time = time.time()

        # Save the original data to a backup file
        start_time_to_store = time.time()
        cache_id = np.random.randint(0, 1000000)
        torch.save(self.data, f'graph_data_backup_{cache_id}.pt')
        end_time_to_store = time.time()
        print(f"Original data stored in backup file 'graph_data_backup_{cache_id}.pt' in {round(end_time_to_store - start_time_to_store, 2)} seconds.")

        for i in range(num_samples):
            # Perturb the underlying HeteroData
            self.data = perturb_instance(self.data, mask, mask_type=explanation_type, perturbation_type=perturbation_strategy)

            start_time_for_inference = time.time()

            # Re-create loader (RelGTTokens reads TFs from self.data at __getitem__ time)
            loader = self.create_loader(self.data, split, shuffle=False)

            pred_list = []
            for batch in loader:
                (neighbor_types, node_indices, neighbor_hops, neighbor_times,
                 grouped_tf_dict, edge_index, batch_vec, _) = self._unpack_batch(batch, self.device)
                out = self.model_to_explain(
                    neighbor_types, node_indices, neighbor_hops, neighbor_times,
                    grouped_tf_dict, edge_index=edge_index, batch=batch_vec,
                )
                out = out.view(-1) if out.size(1) == 1 else out
                if self.explanation_task.explanation_target_type == 'soft':
                    pred = self.process_output(out)
                elif self.explanation_task.explanation_target_type == 'hard':
                    pred = self.get_predictions(out)
                pred_list.append(pred.detach().cpu())

            predictions_per_sample[i] = torch.cat(pred_list, dim=0).numpy()

            end_time_for_inference = time.time()
            print(f"Successfully generated predictions ({len(targets)}) for perturbed instance. Elapsed time: {round(end_time_for_inference - start_time_for_inference, 2)} seconds.")

        # Restore the original data
        start_time_to_load = time.time()
        gc.collect()
        self.data = torch.load(f'graph_data_backup_{cache_id}.pt', weights_only=False)
        end_time_to_load = time.time()
        print(f"Original data restored from backup file 'graph_data_backup_{cache_id}.pt' in {round(end_time_to_load - start_time_to_load, 2)} seconds.")

        end_time = time.time()

        try:
            os.remove(f'graph_data_backup_{cache_id}.pt')
        except Exception as e:
            print(f"Failed to delete backup file: {e}")

        fid_mean, fid_std = self._calculate_fidelity(
            predictions=predictions_per_sample,
            targets=targets,
            distance_metric='abs_difference' if self.explanation_task.task_type == TaskType.BINARY_CLASSIFICATION else 'symmetric_mean_absolute_percentage_change',
        )

        print(f"Fidelity estimation with {num_samples} samples completed. Total time elapsed: {round(end_time - start_time, 2)} seconds.")

        return fid_mean, fid_std, predictions_per_sample, targets

    @torch.no_grad()
    def inference_to_explain_predictions(self, *args, **kwargs):
        """Not implemented for RelGT. Will be removed from the base class in the future."""
        raise NotImplementedError("inference_to_explain_predictions is not used and not implemented for RelGTExplainer.")
