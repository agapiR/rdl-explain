"""Candidate predicate generation for the ``Selection`` explanation language.

``Selection`` explanations filter TUPLES rather than columns (paper, Section
4.1): a view of the form ``select * from R where phi``. Mask learning needs a
finite candidate set of atomic predicates to score, and this builds one from the
columns a ``Projection`` run already found important -- categorical columns
contribute one equality predicate per category, numerical columns one range
predicate per inter-quantile band.

Extracted from ``learn_filter_masks.py`` so the notebooks and the package can
use it without importing a CLI script; that script now imports it from here, so
its behaviour is unchanged.
"""

from typing import Any, Dict, List, Tuple

import math

from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from rdl_explain.explain.explain_utils import node_type_to_col_names_by_stype

#: stypes a filter predicate can be built over.
SUPPORTED_STYPES_FOR_FILTERS = [stype.numerical, stype.categorical,
                                stype.timestamp, stype.embedding]

__all__ = ["SUPPORTED_STYPES_FOR_FILTERS",
           "collect_candidate_filters_given_important_columns"]


def collect_candidate_filters_given_important_columns(
    data: HeteroData,
    col_stats_dict: Dict[str, Any],
    important_col_ranking_top_k: list,
) -> list:
    """
    Collect candidate explanation filters based on important columns.
    
    Args:
        data (HeteroData): The graph data.
        col_stats_dict (Dict[str, Any]): Column statistics dictionary.
        important_col_ranking_top_k (list): List of important columns ranked by importance.
    
    Returns:
        list: A list of candidate filters.
        filter value mapping: A mapping of processed filter values to their original values.
    """
    node_type_to_col_names_by_stype_dict = node_type_to_col_names_by_stype(data)

    # Collect candidate filters based on important columns
    candidate_filters = []
    value_mapping = []
    for col in important_col_ranking_top_k:
        col_node_type, col_name = col

        # Find the stype for the column from node_type_to_col_names_by_stype_dict
        col_stype = None
        for stype, col_names in node_type_to_col_names_by_stype_dict[col_node_type].items():
            if col_name in col_names:
                col_stype = stype
                break
        
        # If the column's stype is not supported, skip it
        if col_stype not in SUPPORTED_STYPES_FOR_FILTERS:
            print(f"Column {col_name} with stype {col_stype} is not supported for filters. Skipping.")
            continue

        # Get the column statistics
        if col_stype == stype.categorical:
            categories, counts = col_stats_dict[col_node_type][col_name][StatType.COUNT]
        elif col_stype == stype.numerical:
            mean = col_stats_dict[col_node_type][col_name][StatType.MEAN]
            std = col_stats_dict[col_node_type][col_name][StatType.STD]
            quantiles = col_stats_dict[col_node_type][col_name][StatType.QUANTILES]

        # Collect values for the filter 
        # 1. For categorical columns, use the values. Update the value --> category mapping.
        # 2. For numerical columns, use the mean std and quantiles to create |{ri}| range filters. values = [(r1v1, r1v2), (r2v1, r2v2), ...]. Update the value --> original value mapping.
        if col_stype == stype.categorical:
            value_mapping_dict = {v: cat for v, cat in enumerate(categories)}
            values = list(range(len(categories)))
            # Sort candidate filter values in ascending order
            values.sort()   # Sort by the value itself
            op = 'equality' # Equality operation for categorical columns
        elif col_stype == stype.numerical:
            values = [
                (quantiles[i], quantiles[i + 1]) for i in range(len(quantiles) - 1)
            ]
            values = list(set(values))  # Ensure unique ranges
            # Sort candidate filter values in ascending order.
            values.sort(key=lambda x: (x[0], x[1])) # Sort by the first element of the range, break ties by the second element
            # Shift right bound to avoid range overlap, except for the last range
            eps = 1e-8
            values = [
                (values[i][0], values[i][1]+eps) if i<len(values)-1 else (values[i][0], values[i][1]) for i in range(len(values))
            ]
            # Shift left bound to avoid range overlap, except for the first range or single value ranges
            values = [
                (values[i][0]+eps, values[i][1]) if (i>0 and values[i][0]+eps < values[i][1]) else (values[i][0], values[i][1]) for i in range(len(values))
            ]
            # print(f"Numerical column {col_name} with mean {mean} and std {std}. Quantiles: {quantiles}")
            col_index = node_type_to_col_names_by_stype_dict[col_node_type][col_stype].index(col_name)
            unique_values_in_tf = data[col_node_type].tf.feat_dict[col_stype][:, col_index].unique().numpy()
            if len(unique_values_in_tf) > 100:
                # sparsify so that I get max 100 unique values
                keep_every = math.ceil(len(unique_values_in_tf) / 100)
                unique_values_in_tf_sparse = unique_values_in_tf[::keep_every]
            else:
                unique_values_in_tf_sparse = unique_values_in_tf
            # print(f"Values for numerical columns in tf: {data[col_node_type].tf.feat_dict[col_stype][:, col_index].unique()}")
            value_mapping_dict = {f'{r[0]}-{r[1]}': [
                v for v in unique_values_in_tf_sparse if r[0] <= v <= r[1]
            ] for r in values}
            value_mapping_dict['total-unique-values'] = len(unique_values_in_tf)
            value_mapping_dict['sparse-unique-values'] = len(unique_values_in_tf_sparse)
            op = 'range'  # Range operation for numerical columns
        else:
            # just store dummy values and mapping for the rest of the stypes
            values = []
            value_mapping_dict = {}
            op = 'dummy'  # Dummy operation for unsupported stypes
            print(f"Column {col_name} with stype {col_stype} is not supported for filters. Skipping.")

        # Collect the filter
        candidate_filters.append((col_node_type, col_name, col_stype, op, values))
        
        # Update the value mapping
        value_mapping.append((col_node_type, col_name, value_mapping_dict))

    return candidate_filters, value_mapping
