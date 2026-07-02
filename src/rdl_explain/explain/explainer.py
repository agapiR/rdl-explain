import time
import math

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import numpy as np
from collections import deque

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.loader import NeighborLoader
from torch_frame import stype
from torch_frame.data import DataLoader as TensorFrameDataLoader

# Explain module imports
from rdl_explain.explain.explain_utils import node_type_to_col_names, node_type_to_col_names_by_stype
from rdl_explain.explain.eval import estimate_deviation_from_determinacy

# Relbench imports
from relbench.base import EntityTask, TaskType
from relbench.modeling.graph import NodeTrainTableInput, get_node_train_table_input
from relbench.modeling.graph import AttachTargetTransform


class RDLExplainer(ABC):

    name = "node_rdl_explainer"

    def __init__(self, config: Dict, model: torch.nn.Module, data: HeteroData, task: EntityTask):
        """Initialize the RDLExplainer."""
        self.config = config
        self.data = data
        self.explanation_task = task
        self.model_to_explain = model
        self.device = self.config.device
        self.model_to_explain.to(self.device) # Move the model to the device
        # Initialize task tables
        self.train_table, self.val_table, self.test_table = self._initialize_task_tables()

    def _initialize_task_tables(self):
        train_table_input = get_node_train_table_input(table=self.explanation_task.train_table, task=self.explanation_task)
        val_table_input = get_node_train_table_input(table=self.explanation_task.val_table, task=self.explanation_task)
        test_table_input = get_node_train_table_input(table=self.explanation_task.test_table, task=self.explanation_task)
        return train_table_input, val_table_input, test_table_input

    def initialize_masks(self, explanation_type: str, mu: float = 10, std: float = 1, filter_predicate: Optional[Tuple[str, str, stype, str, List]] = None) -> Dict:
        """Initialize explanation masks based on the explanation type."""
        if explanation_type == 'table' or explanation_type == 'table_right_before_GNN':
            mask = {node_type: Parameter(torch.randn(1, device=self.device) * std + mu).to(self.device) for node_type in self.data.node_types}
        elif explanation_type == 'column':
            masked_elements = [(node_type, col_name) for node_type in self.data.node_types for col_name in node_type_to_col_names(self.data)[node_type]]
            mask = {(node_type, col_name): Parameter(torch.randn(1, device=self.device) * std + mu).to(self.device) for node_type, col_name in masked_elements}
        elif explanation_type == 'filter':
            assert filter_predicate is not None, "filter_predicate must be provided for 'filter' explanation type"
            node_type, col_name, col_type, op, values = filter_predicate
            mask_params = Parameter(torch.randn(len(values), device=self.device) * std + mu).to(self.device) # create a mask for each value in values
            mask = {'params': {'mask_vals': mask_params}, 'node_type': node_type, 'col_name': col_name, 'col_type': col_type, 'op': op, 'values': values}
            indices = torch.zeros(self.data[node_type].tf.num_rows, dtype=torch.long, device=self.device)
            node_type_to_col_names_by_stype_dict = node_type_to_col_names_by_stype(self.data)   # get the mapping of node types to column names by stype
            col_idx = node_type_to_col_names_by_stype_dict[node_type][col_type].index(col_name) # find the column index for the given node type and column name
            if col_idx < 0:
                raise ValueError(f"Column '{col_name}' not found in node type '{node_type}' with stype '{col_type}'")
            if op == 'equality':
                for i, v in enumerate(values):
                    idxs = torch.where(self.data[node_type].tf.feat_dict[col_type][:, col_idx] == v)[0]
                    indices[idxs] = i
            elif op == 'range':
                for i, v in enumerate(values):
                    # if a feature is equal to the boundary of a value -- i.e., between two ranges --, it is included in the later range
                    idxs = torch.where((self.data[node_type].tf.feat_dict[col_type][:, col_idx] >= v[0]) & (self.data[node_type].tf.feat_dict[col_type][:, col_idx] <= v[1]))[0]
                    indices[idxs] = i 
            elif op == 'keyword':
                raise NotImplementedError("Filter operation 'keyword' is not implemented yet.")
            else:
                raise ValueError(f"Unknown filter operation: {op}")
            mask['indices'] = indices.to(self.device)  # Store the mask index for each node in the node type
            # print("Batch mask vals from initialize_masks: ", mask['params']['mask_vals'])
        elif explanation_type == 'fkpk':
            masked_elements = [edge_type for edge_type in self.data.edge_types]
            f2p_edges = [edge_type for edge_type in masked_elements if 'rev_' not in edge_type[1]]
            p2f_edges = [edge_type for edge_type in masked_elements if 'rev_' in edge_type[1]]
            mask = {edge_type: Parameter(torch.randn(1, device=self.device) * std + mu).to(self.device) for edge_type in p2f_edges}
            mask.update({edge_type: mask[(edge_type[2], 'rev_' + edge_type[1], edge_type[0])] for edge_type in f2p_edges})
        elif explanation_type == 'fkpk-layer-wise':
            masked_elements = [(edge_type, layer) for edge_type in self.data.edge_types for layer in range(self.config.gnn_layers+1)]
            mask = {(edge_type, layer): Parameter(torch.randn(1, device=self.device) * std + mu).to(self.device) for edge_type, layer in masked_elements}
        elif explanation_type == 'layer-wise':
            masked_elements = [(node_type, layer) for node_type in self.data.node_types for layer in range(self.config.gnn_layers+1)]
            mask = {(node_type, layer): Parameter(torch.randn(1, device=self.device) * std + mu).to(self.device) for node_type, layer in masked_elements}
        else:
            raise ValueError(f"Unknown explanation type: '{explanation_type}'")
        return mask

    def _create_loader(self, data: HeteroData, split: str, table_input: NodeTrainTableInput, shuffle: bool = False, node_id_filter: Optional[Tensor] = None) -> NeighborLoader:
        """Create a NeighborLoader for the given split.

        When `node_id_filter` is provided we must filter THREE things in lock-
        step so the target attached to each batch corresponds to the seed node:
        input_nodes, input_time, AND the target tensor inside the transform.
        AttachTargetTransform indexes target by `batch.input_id`, which is the
        position in the (filtered) input-nodes list — so the target tensor
        must be subset to the same `keep` mask, otherwise `target[input_id]`
        returns values from arbitrary other entities.
        """
        input_nodes = table_input.nodes
        input_time = table_input.time
        transform  = table_input.transform
        if node_id_filter is not None:
            node_type, all_ids = input_nodes
            keep = torch.isin(all_ids, node_id_filter.to(all_ids.device))
            input_nodes = (node_type, all_ids[keep])
            input_time  = input_time[keep] if input_time is not None else None
            if table_input.target is not None:
                transform = AttachTargetTransform(node_type, table_input.target[keep])
        return NeighborLoader(
            data,
            num_neighbors=self.config.num_neighbors,
            time_attr="time" if input_time is not None else None,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            batch_size=self.config.inference_batch_size,
            temporal_strategy=self.config.temporal_strategy,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )

    def create_loader(self, data: HeteroData, split: str, shuffle: bool = False, node_id_filter: Optional[Tensor] = None) -> NeighborLoader:
        """Create data loader for the given split."""
        if split == 'train':
            data_loader = self._create_loader(data, split, self.train_table, shuffle=shuffle, node_id_filter=node_id_filter)
        elif split == 'val':
            data_loader = self._create_loader(data, split, self.val_table, shuffle=shuffle, node_id_filter=node_id_filter)
        elif split == 'test':
            data_loader = self._create_loader(data, split, self.test_table, shuffle=shuffle, node_id_filter=node_id_filter)
        else:
            raise ValueError(f"Invalid split: {split}")
        return data_loader

    def create_tf_loader(self, node_type: NodeType, shuffle: bool = False) -> TensorFrameDataLoader:
        """Create a DataLoader for the tensor frame of the given node type."""
        if node_type not in self.data.node_types:
            raise ValueError(f"Node type '{node_type}' not found in the data.")
        tf_for_node_type = self.data[node_type].tf
        return TensorFrameDataLoader(
            tf_for_node_type, 
            batch_size=self.config.inference_batch_size,
            shuffle=shuffle,
        )

    def process_output(self, output: Tensor) -> Tensor:
        if self.explanation_task.task_type == TaskType.BINARY_CLASSIFICATION:
            # The processed output is the probability of the positive class
            processed_output = torch.sigmoid(output)
        elif self.explanation_task.task_type == TaskType.REGRESSION:
            # TODO: add clamping here?
            processed_output = output
        else:
            raise NotImplementedError(f"Task type {self.explanation_task.task_type} is not supported.")
        return processed_output

    def get_predictions(self, output: Tensor) -> Tensor:
        processed_output = self.process_output(output)
        if self.explanation_task.task_type == TaskType.BINARY_CLASSIFICATION:
            # The predicted labels column will have the label of class 0 if the predicted probability is less than prob_threshold else 1
            probs = processed_output
            prob_threshold = 0.5
            pred = torch.FloatTensor([0 if x < prob_threshold else 1 for x in probs])
        elif self.explanation_task.task_type == TaskType.REGRESSION:
            pred = processed_output
        else:
            raise NotImplementedError(f"Task type {self.explanation_task.task_type} is not supported.")
        return pred

    def process_mask(self, mask: Dict, apply_activation: bool = True) -> Dict:
        """Process the mask to get the final mask values."""
        processed_mask = {}
        for masked_element in mask.keys():
            if apply_activation:
                processed_mask[masked_element] = mask[masked_element].sigmoid().detach().cpu()
            else:
                processed_mask[masked_element] = mask[masked_element].detach().cpu()
        return processed_mask

    def _task_loss_fn(
        self,
    ) -> torch.nn.Module:
        """
        Configure task loss based on the task type.
        Returns loss function (torch.nn.Module)
        """
        if self.explanation_task.task_type == TaskType.BINARY_CLASSIFICATION:
            return BCEWithLogitsLoss()
        elif self.explanation_task.task_type == TaskType.REGRESSION:
            return L1Loss()
        else:
            raise NotImplementedError(f"Task type {self.explanation_task.task_type} is not supported.")
    
    def _mask_regularization_loss_fn(
        self,
        reg_type: str = 'l1',
        mask_size_budget: Optional[int] = None,
    ) -> torch.nn.Module:
        """
        Configure mask regularization loss.
        Returns loss function (torch.nn.Module)
        """
        if reg_type == 'l1':
            def reg_fn(mask: Dict):
                mask_params = torch.cat([p for p in mask.values()]).sigmoid()
                l1_reg = torch.norm(mask_params, 1)
                return l1_reg
        elif reg_type == 'relu':
            assert mask_size_budget is not None, "mask_size_budget must be provided for 'relu' mask regularization"
            def reg_fn(mask: Dict):
                mask_params = torch.cat([p for p in mask.values()]).sigmoid()
                relu_reg = (torch.norm(mask_params, 1) - mask_size_budget).relu()
                return relu_reg
        else:
            raise ValueError(f"Unsupported regularization type: {reg_type}")
        return reg_fn

    def mask_learning_loss_fn(
        self,
        eps: float = 10,
        reg_type: str = 'l1',
        mask_size_budget: Optional[int] = None,
    ) -> torch.nn.Module:
        """
        Configure the loss function for mask learning.
        Returns loss function (torch.nn.Module)
        """
        task_loss_fn = self._task_loss_fn()
        mask_regularization_loss_fn = self._mask_regularization_loss_fn(
            reg_type=reg_type,
            mask_size_budget=mask_size_budget,
        )
        
        def loss_fn(output: Tensor, targets: Tensor, mask: Dict):
            task_loss = task_loss_fn(output, targets)
            reg_loss = mask_regularization_loss_fn(mask)
            loss = task_loss + eps * reg_loss
            return loss, task_loss.item(), reg_loss.item()
        
        return loss_fn

    @staticmethod
    def _round_eps(eps: float) -> float:
        """Round eps to the nearest value in {1×10^i, 5×10^i} for any integer i.

        Examples: 0.0234 → 0.05,  0.123 → 0.1,  3.7 → 5.0,  0.0007 → 0.0005
        """
        if eps <= 0:
            return eps
        log_eps = math.log10(eps)
        floor_log = math.floor(log_eps)
        candidates = []
        for i in range(floor_log - 1, floor_log + 3):
            candidates.append(10.0 ** i)
            candidates.append(5.0 * 10.0 ** i)
        return min(candidates, key=lambda c: abs(math.log10(c) - log_eps))

    def _auto_eps(
        self,
        data_loader,
        mask: Dict,
        explanation_type: str,
        elimination_strategy: str,
        default_feat_vector,
    ) -> float:
        """Compute eps so that task_loss ≈ eps * reg_loss at initialization,
        then round to the nearest value in {1×10^i, 5×10^i}.

        Runs a single no-grad forward pass on the first batch and returns
        task_loss_0 / reg_loss_0 rounded to a nice value.
        """
        # reg_loss at init: analytically from current mask parameter values
        if explanation_type == 'filter':
            all_params = torch.cat([mask_fi['params']['mask_vals'].detach() for mask_fi in mask.values()])
        else:
            all_params = torch.cat([p.detach() for p in mask.values()])
        reg_loss_0 = all_params.sigmoid().sum().item()

        # task_loss at init: one forward pass on the first batch
        batch = next(iter(data_loader))
        batch = batch.to(self.device)
        with torch.no_grad():
            if explanation_type == 'filter':
                batch_mask = {}
                for fi, mask_fi in mask.items():
                    nt = mask_fi['node_type']
                    vals = mask_fi['params']['mask_vals'][mask_fi['indices'][batch[nt].n_id]]
                    batch_mask[nt] = vals if nt not in batch_mask else torch.add(batch_mask[nt], vals)
                batch_mask = {k: torch.clamp(v, max=1) for k, v in batch_mask.items()}
            else:
                batch_mask = mask
            out = self.model_to_explain.forward_to_explain(
                explanation_type, batch_mask, batch, self.explanation_task.entity_table,
                elimination_strategy=elimination_strategy,
                uninformative_feat_vector=default_feat_vector,
            )
            targets = batch[self.explanation_task.entity_table].y
            out = out.view(-1) if out.size(1) == 1 else out
            task_loss_0 = self._task_loss_fn()(out, targets).item()

        eps = task_loss_0 / reg_loss_0 if reg_loss_0 > 0 else 1.0
        eps_rounded = self._round_eps(eps)
        print(f"[Auto-eps] task_loss_0={task_loss_0:.4f}, reg_loss_0={reg_loss_0:.2f} → eps={eps:.6f} → rounded={eps_rounded:.6g}")
        return eps_rounded

    def learn_masks_single_epoch(
        self, 
        loader: NeighborLoader,
        loss_fn: callable,
        optimizer: torch.optim.Optimizer,
        mask: Dict,
        explanation_type: str = 'table',
        elimination_strategy: str = 'zero',
        default_feat_vector: Dict[NodeType, Tensor] = None,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Train the mask for one epoch."""
        out_list, gt_list = [], []
        loss_accum = count_accum = 0
        task_loss_accum = reg_loss_accum = 0

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            if explanation_type == 'filter':
                batch_mask = {}
                # batch_mask = {mask['node_type']: mask['params']['mask_vals'][mask['indices'][batch[mask['node_type']].n_id]]}
                for fi, mask_fi in mask.items():
                    # For filter explanation type, we need to create a batch mask
                    # The mask values are the values of the mask parameters for the nodes in the batch
                    # In case we have multiple masks for the same node type, we need to combine them
                    # NOTE: Lukasiewicz combination strategy: min(1, sum(mask_values_for_node_type))
                    if mask_fi['node_type'] not in batch_mask:
                        batch_mask[mask_fi['node_type']] = mask_fi['params']['mask_vals'][mask_fi['indices'][batch[mask_fi['node_type']].n_id]]
                    else: # if there is more filters for a given node type, add them up
                        batch_mask[mask_fi['node_type']] = torch.add(batch_mask[mask_fi['node_type']], mask_fi['params']['mask_vals'][mask_fi['indices'][batch[mask_fi['node_type']].n_id]])
                batch_mask = {k: torch.clamp(v, min=None, max=1) for k, v in batch_mask.items()} # clamp the mask values to be at most 1
                out = self.model_to_explain.forward_to_explain(explanation_type, batch_mask, batch, self.explanation_task.entity_table, elimination_strategy=elimination_strategy, uninformative_feat_vector=default_feat_vector)
            else:
                out = self.model_to_explain.forward_to_explain(explanation_type, mask, batch, self.explanation_task.entity_table, elimination_strategy=elimination_strategy, uninformative_feat_vector=default_feat_vector)
            targets = batch[self.explanation_task.entity_table].y
            out = out.view(-1) if out.size(1) == 1 else out
            # NOTE: loss uses unnormalized logits
            if explanation_type == 'filter':
                mask_params = {fi: mask_fi['params']['mask_vals'] for fi, mask_fi in mask.items()}
                loss, task_loss, reg_loss = loss_fn(out, targets, mask_params)
            else:
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

    def learn_masks(
        self,
        eps: float | str = 10,
        explanation_type: str = 'table',
        elimination_strategy: str = 'zero',
        n_epochs: int = 1000,
        lr: float = 0.01,
        split: str = 'train',
        filter_predicates: Optional[List[Tuple[str, str, stype, str, List]]] = None,
        # New params are keyword-only (after `*`) so the original positional
        # signature (eps, explanation_type, elimination_strategy, n_epochs, lr,
        # split, filter_predicates) is preserved for existing callers.
        *,
        n_replacement_samples: int = 2000,
        node_id_filter: Optional[Tensor] = None,
        mask_init_mu: float = 10,
        mask_init_std: float = 1,
        pinned_columns: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[Dict, Dict]:
        """Learn explanation masks for the given model.

        Args:
            mask_init_mu: Mean of the Gaussian used to initialise mask logits.
                Default 10 (sigmoid≈1, all columns start active). Use 0 to
                initialise in the maximally sensitive sigmoid region — useful
                when exploring multiple optima via different random seeds.
            mask_init_std: Std of the initialisation Gaussian. Has negligible
                effect when mask_init_mu=10 (sigmoid is saturated), but
                produces meaningful variation when mask_init_mu≈0.
        """

        start_time_for_mask_learning = time.time()

        # Create data loader for the given split (optionally restricted to a node subset)
        data_loader = self.create_loader(self.data, split, shuffle=True, node_id_filter=node_id_filter)

        # Initialize masks
        if explanation_type == 'filter':
            assert filter_predicates is not None, "filter_predicates must be provided for 'filter' explanation type"
            mask = {}
            for fi, filter_predicate in enumerate(filter_predicates):
                mask[fi] = self.initialize_masks(explanation_type, mu=mask_init_mu, std=mask_init_std, filter_predicate=filter_predicate)
        else:
            mask = self.initialize_masks(explanation_type, mu=mask_init_mu, std=mask_init_std)

        # Collect the replacement feature vectors based on the elimination strategy
        if elimination_strategy in ["avg", "avg_with_noise", "permutation_joint", "permutation_independent"]:
            replacement_strategy = 'column_'+ elimination_strategy if explanation_type=='column' else 'row_' + elimination_strategy
            default_feat_vector = self.prepare_replacement_vectors(replacement_strategy, replacement_sample_size=n_replacement_samples)
            # Rename the elimination strategy to match the expected names in the model
            if elimination_strategy == "avg":
                elimination_strategy = "default"
            elif elimination_strategy == "avg_with_noise":
                elimination_strategy = "default_w_perturbation"
        else:
            default_feat_vector = None

        # Move the replacement feature vectors to the device
        if default_feat_vector is not None:
            default_feat_vector = {k: v.to(self.device) for k, v in default_feat_vector.items()}

        # Auto-tune eps so task_loss ≈ eps * reg_loss at initialization
        if eps == 'auto':
            eps = self._auto_eps(data_loader, mask, explanation_type, elimination_strategy, default_feat_vector)

        # Collect the mask parameters to optimize
        if explanation_type == 'filter':
            # For filter explanation type, we only optimize the mask parameters
            parameters = [mask_fi['params']['mask_vals'] for mask_fi in mask.values()]
        else:
            # Preserved-predicate mode: any (node_type, col) in `pinned_columns`
            # has its mask logit frozen at the initialization (mask_init_mu=10
            # → sigmoid≈1, i.e. "always keep"). The L1 reg gradient on these
            # params is ignored because requires_grad=False, so the optimizer
            # can't pull them down. Use this to force the mask to find what's
            # cohort-discriminative BEYOND the predicate column.
            if pinned_columns:
                pinned_set = {tuple(c) for c in pinned_columns}
                unknown = pinned_set - set(mask.keys())
                if unknown:
                    raise KeyError(
                        f"pinned_columns contains keys not present in the mask: "
                        f"{sorted(unknown)}. Available keys (first 10): "
                        f"{list(mask.keys())[:10]}"
                    )
                for k in pinned_set:
                    mask[k].requires_grad_(False)
                parameters = [p for k, p in mask.items() if k not in pinned_set]
                print(f"[learn_masks] pinned {len(pinned_set)} column(s): "
                      f"{sorted(pinned_set)}")
            else:
                parameters = list(mask.values())
        optimizer = torch.optim.Adam(parameters, lr=lr)

        # Setup the loss function
        loss_fn = self.mask_learning_loss_fn(
            eps=eps,
            # reg_type=self.config.explanation_parameters.regularization_type, 
            # mask_size_budget=self.config.explanation_parameters.mask_size_budget,
        )

        # Learn the explanation masks
        window_size = 5 # self.config.explanation_parameters.sliding_window_size, for early stopping
        min_delta = 1e-6  # for early stopping convergence check
        sliding_window_prev = deque(maxlen=window_size)
        sliding_window = deque(maxlen=window_size)
        if explanation_type == 'filter':
            initial_mask_vals = [self.process_mask(mask_fi['params']) for mask_fi in mask.values()]
            initial_mask_vals = {(fi, i): initial_mask_vals_fi['mask_vals'][i] for fi, initial_mask_vals_fi in enumerate(initial_mask_vals) for i in range(len(initial_mask_vals_fi['mask_vals'])) }
        else:
            initial_mask_vals = self.process_mask(mask)
        mask_vals = {masked_element: [initial_mask_vals[masked_element].item()] for masked_element in initial_mask_vals.keys()}
        metrics = {'loss': [], 'task_loss': [], 'reg_loss': []}
        self.model_to_explain.eval() # Set the model to evaluation mode, we don't want to update the model weights only the mask weights
        for epoch in range(n_epochs):
            start_time = time.time()
            mask, loss, task_loss, reg_loss, _, _ = self.learn_masks_single_epoch(data_loader, loss_fn, optimizer, mask, explanation_type, elimination_strategy, default_feat_vector)
            end_time = time.time()
            # Process the mask values
            if explanation_type == 'filter':
                # current_mask_vals = self.process_mask(mask['params'])
                current_mask_vals = [self.process_mask(mask_fi['params']) for mask_fi in mask.values()]
                # current_mask_vals = {i: current_mask_vals['mask_vals'][i] for i in range(len(current_mask_vals['mask_vals']))}
                current_mask_vals = {(fi, i): current_mask_vals_fi['mask_vals'][i] for fi, current_mask_vals_fi in enumerate(current_mask_vals) for i in range(len(current_mask_vals_fi['mask_vals'])) }
            else:
                current_mask_vals = self.process_mask(mask)
            mask_vals = {masked_element: mask_vals[masked_element] + [current_mask_vals[masked_element].item()] for masked_element in mask_vals.keys()}
            # Log the metrics
            metrics['loss'].append(loss)
            metrics['task_loss'].append(task_loss)
            metrics['reg_loss'].append(reg_loss)
            if epoch % max(1, int(n_epochs*0.1)) == 0:
                print(f"Epoch {epoch}/{n_epochs}: Loss {metrics['loss'][-1]} - Task Loss {metrics['task_loss'][-1]} - Regularization {metrics['reg_loss'][-1]} - Time-per-epoch {round(end_time - start_time, 2)}s")
            # Early stopping
            if len(sliding_window_prev) < window_size:
                sliding_window_prev.append(loss)
            elif len(sliding_window) < window_size:
                sliding_window.append(loss)
            else:
                avg_prev = sum(sliding_window_prev) / window_size
                avg = sum(sliding_window) / window_size
                # Check convergence
                if abs(avg_prev - avg) < min_delta:
                    print(f"Mask learning stopped early at epoch {epoch} due to convergence.")
                    break
                # Update the sliding windows
                sliding_window_prev = sliding_window
                sliding_window = deque([loss], maxlen=window_size)
        
        end_time_for_mask_learning = time.time()
        metrics['time'] = end_time_for_mask_learning - start_time_for_mask_learning
        metrics['sliding_window_size'] = window_size
        metrics['min_delta'] = min_delta
        print(f"Mask learning completed in {round(metrics['time'], 2)} seconds.")

        # NOTE: I return the mask without activation
        if explanation_type == 'filter':
            learned_mask = [self.process_mask(mask_fi['params'], apply_activation=False) for mask_fi in mask.values()]
        else:
            learned_mask = self.process_mask(mask, apply_activation=False) 

        return learned_mask, mask_vals, metrics
    
    def prepare_replacement_vectors(
        self,
        replacement_strategy: str,
        replacement_sample_size: int = 2000,
    ) -> Dict[NodeType, Tensor]:
        if replacement_strategy in ["column_avg", "column_avg_with_noise"]:
            replacement_vectors = self.get_intermediate_encodings_for_replacement(encoding_stage='column_wise', average=True)
        elif replacement_strategy in ["row_avg", "row_avg_with_noise"]:
            replacement_vectors = self.get_intermediate_encodings_for_replacement(encoding_stage='fused', average=True)
        elif replacement_strategy in ["column_permutation_joint", "column_permutation_independent"]:
            replacement_vectors = self.get_intermediate_encodings_for_replacement(encoding_stage='column_wise', sample_size=replacement_sample_size)
        elif replacement_strategy in ["row_permutation_joint", "row_permutation_independent"]:
            replacement_vectors = self.get_intermediate_encodings_for_replacement(encoding_stage='fused', sample_size=replacement_sample_size)
        else:
            raise ValueError(f"Unknown elimination strategy: {replacement_strategy}")
        return replacement_vectors

    @torch.no_grad()
    def estimate_devdelta(
        self,
        mask: Dict,
        split: str = 'train',
        explanation_type: str = 'column',
        perturbation_strategy: str = 'permutation_independent',
        num_samples: int = 100,
        node_id_filter: Optional[Tensor] = None,
        prediction_type: str = 'soft',
        random_seed: Optional[int] = None,
    ):
        """Estimate deviation from determinacy (devΔ) for a (hard) mask.

        devΔ is the paper's evaluation metric: lower ⇒ the explanation better
        determines the model's predictions (label-free — it compares the model's
        predictions on the original vs. perturbed database). This is a thin
        wrapper around ``estimate_deviation_from_determinacy`` (in
        ``rdl_explain.explain.eval``), which holds the corrected estimator
        (fresh-inference reference, cross-sample SEM, abs-diff / SMAPE-2 distance).

        By default it evaluates on the ``train`` split — the SAME instances used
        for mask learning (``learn_masks`` also defaults to ``'train'``). Pass a
        different ``split`` (or a ``node_id_filter``) to evaluate elsewhere.

        Returns ``(dev_delta, dev_sem, per_instance_dev, per_instance_dev_std,
        per_sample_mean_distances, original_predictions)`` — see
        ``estimate_deviation_from_determinacy``.
        """
        node_ids = node_id_filter
        if node_ids is not None and not torch.is_tensor(node_ids):
            node_ids = torch.as_tensor(node_ids)
        loader_factory = lambda: self.create_loader(
            self.data, split, shuffle=False, node_id_filter=node_ids)
        return estimate_deviation_from_determinacy(
            model=self.model_to_explain,
            data=self.data,
            task=self.explanation_task,
            mask=mask,
            loader_factory=loader_factory,
            explanation_type=explanation_type,
            perturbation_strategy=perturbation_strategy,
            num_samples=num_samples,
            prediction_type=prediction_type,
            device=str(self.device),
            random_seed=random_seed,
        )

    @torch.no_grad()
    def inference_to_explain_predictions(
        self,
        split: str,
        mask: Dict = None,
        explanation_type: str = 'table',
        elimination_strategy: str = 'zero',
        default_feat_vector: Dict[NodeType, Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[int]]:
        """Perform inference-to-explain for the given data and masks."""

        assert explanation_type in ['table', 'column', 'fkpk', 'fkpk-layer-wise', 'layer-wise'], f"Inference with mask is not supported for explanation type '{explanation_type}'"

        # Create data loader for the given split
        start_time = time.time()
        data_loader = self.create_loader(self.data, split, shuffle=False)
        end_time = time.time()
        print(f"Data loader created in {round(end_time - start_time, 2)} seconds for split '{split}'")

        # Move the mask to the device if provided
        if mask is not None:
            mask = {k: v.to(self.device) for k, v in mask.items()}

        # Move the default feature vector to the device if provided
        if default_feat_vector is not None:
            default_feat_vector = {k: v.to(self.device) for k, v in default_feat_vector.items()}

        # Generate predictions using the masks
        pred_list = []
        out_list = []
        start_time = time.time()
        # for batch in tqdm(data_loader, total=len(data_loader), desc="inference to explain"):
        for batch in data_loader:
            batch = batch.to(self.device)
            if mask is None:
                # If no mask is provided, we use the model without explanation
                out = self.model_to_explain(batch, self.explanation_task.entity_table)
            else:
                out = self.model_to_explain.forward_to_explain(explanation_type, mask, batch, self.explanation_task.entity_table, elimination_strategy=elimination_strategy, uninformative_feat_vector=default_feat_vector)
            out = out.view(-1) if out.size(1) == 1 else out
            out_list.append(out.detach().cpu())
            pred = self.get_predictions(out)
            pred_list.append(pred.detach().cpu())
        end_time = time.time()
        print(f"Inference-to-explain completed in {round(end_time - start_time, 2)} seconds for split '{split}'")

        output = torch.cat(out_list, dim=0) # return the logits
        predictions = torch.cat(pred_list, dim=0)

        return predictions, output


    @torch.no_grad()
    def inference_to_get_intermediate_encodings(
        self,
        encoding_stage: str = 'column_wise',
        max_sample_size: Optional[int] = None,
    ) -> Dict[NodeType, Tensor]:
        """Perform inference to retrieve intermediate feature encodings."""
        # TODO: handle time in some way to prevent information leakage
        
        # Generate encodings
        encodings = {}
        for node_type in self.data.node_types:
            # Create data loader for the given split
            tf_loader = self.create_tf_loader(node_type, shuffle=False if max_sample_size is None else True)
            # Get the intermediate encodings
            num_nodes = 0
            enc = []
            for tf_batch in tf_loader:
                tf_batch = tf_batch.to(self.device)
                reduced_data = HeteroData()
                reduced_data.tf_dict = {node_type: tf_batch}
                (intermediate_enc, col_names), fused_enc = self.model_to_explain.get_intermediate_encoding(reduced_data, node_type)
                if encoding_stage == 'column_wise':
                    enc.append(intermediate_enc.detach().cpu())
                elif encoding_stage == 'fused':
                    enc.append(fused_enc.detach().cpu())
                else:
                    raise ValueError(f"Invalid encoding stage: {encoding_stage}")
                num_nodes += enc[-1].size(0)
                if max_sample_size is not None and num_nodes >= max_sample_size:
                    break
            # Concatenate the encodings for the node type
            encodings[node_type] = torch.cat(enc, dim=0)

        return encodings

    @torch.no_grad()
    def inference_to_get_intermediate_encodings_without_minibatching(
        self,
        encoding_stage: str = 'column_wise',
    ) -> Dict[NodeType, Tensor]:
        """Perform inference to retrieve intermediate feature encodings."""
        # TODO: how can I do this with minibatching, for all node types
        # TODO: handle time in some way to prevent information leakage
        
        # Generate encodings
        self.data.to(self.device)
        encodings = {}
        for node_type in self.data.node_types:
            (intermediate_enc, col_names), fused_enc = self.model_to_explain.get_intermediate_encoding(self.data, node_type)
            if encoding_stage == 'column_wise':
                encodings[node_type] = intermediate_enc
            elif encoding_stage == 'fused':
                encodings[node_type] = fused_enc
            else:
                raise ValueError(f"Invalid encoding stage: {encoding_stage}")

        # Move encodings to cpu
        for node_type in encodings.keys():
            encodings[node_type] = encodings[node_type].cpu()

        # Move data back to cpu
        self.data.cpu()

        return encodings

    def get_intermediate_encodings_for_replacement(
        self, 
        encoding_stage: str = 'column_wise', # 'column_wise' or 'fused' 
        sample_size: Optional[int] = 2000,
        average: bool = False,
    ) -> Dict[NodeType, Tensor]:
        encodings = self.inference_to_get_intermediate_encodings(encoding_stage=encoding_stage, max_sample_size=sample_size)
        sample_feat_vector = {}
        if sample_size is None: # if sample_size is None, return all the encodings
            for node_type, encoding in encodings.items():
                sample_feat_vector[node_type] = encoding
        else:
            for node_type, encoding in encodings.items():
                if encoding.shape[0] < sample_size: # if samples are less than sample_size, return all the encodings
                    sample_feat_vector[node_type] = encoding
                else:
                    sample_indices = np.random.choice(encoding.shape[0], sample_size, replace=False) # sample without replacement
                    sample_feat_vector[node_type] = encoding[sample_indices]
        if average:
            for node_type in sample_feat_vector.keys():
                sample_feat_vector[node_type] = sample_feat_vector[node_type].mean(dim=0)
        return sample_feat_vector