"""Visualization helpers for RDL explanations.

Renders the database schema with explanation overlays, in two views:

  * ``view="flat"`` — an ER-style diagram (one box per table, one row per
    column). Shows the FULL schema; tables/joins more than ``num_layers`` hops
    from the entity (outside the GNN receptive field) are dimmed.
  * ``view="dag"``  — the GNN's UNROLLED computation: the schema BFS-expanded to
    ``num_layers`` hops from the entity, where a table can reappear at deeper
    layers (e.g. readers ← books ← readers). ``allow_revisits`` toggles
    truncating vs. unrolling these revisits.

Both overlay column-mask importance (shaded/highlighted column rows) and
fkpk-mask importance (coloured, weighted foreign-key edges). The DAG view colours
every occurrence of a repeated join by the same (layer-shared) fkpk value.

Uses graphviz (HTML-like table labels) because networkx node-link drawing can't
show per-column rows. Reuses the existing ``node_type_to_col_names`` and
``make_schema_dag`` helpers; derives the FK structure from ``data.edge_types`` so
it works for both v1 (graph-only) and v2 models without a relbench Database
object. Does not modify the existing schema helpers in ``explain_utils``.
"""

import html
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch

from rdl_explain.explain.explain_utils import node_type_to_col_names, make_schema_dag


# ── value / colour helpers ───────────────────────────────────────────────────

def _to_float(v) -> float:
    """Normalize a mask value to a float in [0, 1] (sigmoid for raw logit tensors)."""
    if torch.is_tensor(v):
        return float(torch.sigmoid(v.detach()).item())
    return float(v)


def _norm_mask(mask) -> Dict:
    return {k: _to_float(v) for k, v in mask.items()} if mask else {}


def _ramp(v: float, base=(255, 140, 0)) -> str:
    """White (v=0) -> `base` colour (v=1) as a hex string."""
    v = max(0.0, min(1.0, v))
    r = int(255 + (base[0] - 255) * v)
    g = int(255 + (base[1] - 255) * v)
    b = int(255 + (base[2] - 255) * v)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── structure derived from the hetero graph ──────────────────────────────────

def _fk_edges(data):
    """Yield (child_table, fk_col, parent_table, edge_type) for each f2p edge."""
    for et in data.edge_types:
        src, rel, dst = et
        if rel.startswith("f2p_"):
            yield src, rel[len("f2p_"):], dst, et


def _schema_graph(data) -> Dict[str, List[dict]]:
    """Build the undirected schema graph ({table: [{dst, edge_name, edge_type}]})
    from data.edge_types — the same shape `make_schema_graph` produces, but
    without needing a relbench Database object."""
    sg: Dict[str, List[dict]] = defaultdict(list)
    for src, rel, dst in data.edge_types:
        if rel.startswith("f2p_"):
            sg[src].append({"dst": dst, "edge_name": rel, "edge_type": "N:1"})
        elif rel.startswith("rev_f2p_"):
            sg[src].append({"dst": dst, "edge_name": rel, "edge_type": "1:N"})
    return sg


def _hop_distances(data, entity_table: str) -> Dict[str, int]:
    """BFS hop distance from `entity_table` over the undirected FK adjacency."""
    adj: Dict[str, set] = {nt: set() for nt in data.node_types}
    for child, _col, parent, _et in _fk_edges(data):
        adj[child].add(parent)
        adj[parent].add(child)
    dist = {entity_table: 0}
    q = deque([entity_table])
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def _match_node_type(data, name: str) -> str:
    for nt in data.node_types:
        if nt.lower() == name.lower():
            return nt
    return name


def _fkpk_by_name(fmask: Dict) -> Dict[str, float]:
    """Map edge_name (f2p_* / rev_f2p_*) -> fkpk value (the two share a value)."""
    return {rel: v for (src, rel, dst), v in fmask.items()}


# ── shared table-box label ───────────────────────────────────────────────────

def _table_label(nt, cols, cmask, *, delta, show_all_columns, is_entity,
                 is_reachable, num_layers, layer_badge=None) -> str:
    important = [c for c in cols if cmask.get((nt, c), 0.0) >= delta]
    shown = cols if show_all_columns else important
    n_hidden = len(cols) - len(shown)

    if is_entity:
        hdr_bg = "#cfe8ff"
    elif not is_reachable:
        hdr_bg = "#e8e8e8"
    else:
        hdr_bg = "#f0f0f0"
    hdr_fg = "#999999" if not is_reachable else "#000000"

    badge = ""
    if layer_badge is not None:
        badge = '  <FONT POINT-SIZE="9" COLOR="#3366cc">%s</FONT>' % layer_badge
    elif is_entity:
        badge = '  <FONT POINT-SIZE="9">(entity)</FONT>'
    elif not is_reachable:
        badge = '  <FONT POINT-SIZE="9">(beyond %d hops)</FONT>' % num_layers

    rows = ['<TR><TD BGCOLOR="%s"><FONT COLOR="%s"><B>%s</B>%s</FONT></TD></TR>'
            % (hdr_bg, hdr_fg, html.escape(nt), badge)]
    for c in shown:
        v = cmask.get((nt, c), 0.0)
        if not is_reachable:
            bg, fg, cell = "#ffffff", "#aaaaaa", html.escape(c)
        elif v >= delta:
            bg, fg = _ramp(v), "#000000"
            cell = "<B>%s</B>  <FONT POINT-SIZE='9'>%.2f</FONT>" % (html.escape(c), v)
        else:
            bg, fg, cell = "#ffffff", "#666666", html.escape(c)
        rows.append('<TR><TD ALIGN="LEFT" BGCOLOR="%s"><FONT COLOR="%s">%s</FONT></TD></TR>'
                    % (bg, fg, cell))
    if n_hidden > 0:
        rows.append('<TR><TD ALIGN="LEFT"><FONT COLOR="#999999" POINT-SIZE="9">'
                    '<I>(+%d more)</I></FONT></TD></TR>' % n_hidden)
    return '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">%s</TABLE>>' % "".join(rows)


def _edge_style(v: float, used: bool, delta: float):
    if not used:
        return "#cccccc", "1", "dashed"
    if v >= delta:
        return _ramp(v, base=(214, 39, 40)), str(1 + 4 * v), "solid"
    return "#999999", "1", "solid"


# ── main entry point ─────────────────────────────────────────────────────────

def draw_schema_with_masks(
    data,
    *,
    entity_table: str,
    view: str = "flat",
    column_mask: Optional[Dict[Tuple[str, str], float]] = None,
    fkpk_mask: Optional[Dict[Tuple[str, str, str], float]] = None,
    num_layers: Optional[int] = None,
    allow_revisits: bool = False,
    delta: float = 0.1,
    show_all_columns: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    fmt: str = "png",
):
    """Draw the schema with explanation overlays.

    Args:
        data:          HeteroData graph (node types = tables; f2p_* edge types).
        entity_table:  prediction-entity table (case-insensitive).
        view:          "flat" (full schema ER diagram, receptive field dimmed) or
                       "dag" (unrolled layered receptive field to num_layers hops).
        column_mask:   {(node_type, col): value} (soft float in [0,1] or logit tensor).
        fkpk_mask:     {edge_type: value} (f2p / rev_f2p share a value).
        num_layers:    GNN layers. flat: dim tables/joins beyond this many hops
                       (None = no dimming). dag: REQUIRED — the unroll depth.
        allow_revisits: dag only. False truncates revisits (stop once a table would
                       repeat); True unrolls them (table reappears at deeper layers).
        delta:         importance threshold.
        show_all_columns: False (default) shows only important columns + "(+N more)".
        title, save_path, fmt: rendering options.

    Returns:
        graphviz.Digraph (rendered to save_path if given).
    """
    import graphviz

    cmask = _norm_mask(column_mask)
    fmask = _norm_mask(fkpk_mask)
    fk_by_name = _fkpk_by_name(fmask)
    cols_by_table = node_type_to_col_names(data)
    entity_table = _match_node_type(data, entity_table)

    dot = graphviz.Digraph("schema", format=fmt)
    dot.attr(rankdir="LR", nodesep="0.4", ranksep="0.9")
    dot.attr("node", shape="plaintext")
    if title:
        dot.attr(label=title, labelloc="t", fontsize="16")

    if view == "flat":
        return _draw_flat(dot, data, entity_table, cols_by_table, cmask, fmask,
                          num_layers, delta, show_all_columns, save_path)
    elif view == "dag":
        if num_layers is None:
            raise ValueError("view='dag' requires num_layers (the unroll depth).")
        return _draw_dag(dot, data, entity_table, cols_by_table, cmask, fk_by_name,
                         num_layers, allow_revisits, delta, show_all_columns, save_path)
    else:
        raise ValueError(f"Unknown view {view!r}; expected 'flat' or 'dag'.")


def _draw_flat(dot, data, entity_table, cols_by_table, cmask, fmask,
               num_layers, delta, show_all_columns, save_path):
    hops = _hop_distances(data, entity_table)

    def reachable(nt):
        return num_layers is None or hops.get(nt, 10**9) <= num_layers

    for nt in data.node_types:
        dot.node(nt, label=_table_label(
            nt, cols_by_table.get(nt, []), cmask, delta=delta,
            show_all_columns=show_all_columns, is_entity=(nt == entity_table),
            is_reachable=reachable(nt), num_layers=num_layers))

    for child, col, parent, et in _fk_edges(data):
        rev = (parent, "rev_f2p_" + col, child)
        v = fmask.get(et, fmask.get(rev, 0.0))
        used = reachable(child) and reachable(parent)
        color, pen, style = _edge_style(v, used, delta)
        label = (" %.2f" % v) if (used and v >= delta) else ""
        dot.edge(child, parent, label=label, color=color, penwidth=pen,
                 style=style, fontsize="9", fontcolor=color)

    if save_path is not None:
        dot.render(save_path, cleanup=True)
    return dot


def _draw_dag(dot, data, entity_table, cols_by_table, cmask, fk_by_name,
              num_layers, allow_revisits, delta, show_all_columns, save_path):
    schema_graph = _schema_graph(data)
    schema_dag = make_schema_dag(
        schema_graph, depth=num_layers, source_entity=entity_table,
        layer_specific_node_type=True, avoid_backtracking=not allow_revisits)

    def node_id(table, layer):
        return f"{table}@L{layer}"

    # nodes, grouped by layer (rank=same) for a clean layered layout
    by_layer: Dict[int, list] = defaultdict(list)
    seen = set()
    for (table, layer) in schema_dag.keys():
        by_layer[layer].append(table)
    # also include leaf nodes that appear only as edge destinations
    for (table, layer), edges in schema_dag.items():
        for e in edges:
            dt, dl = e["dst"]
            by_layer[dl].append(dt)

    for layer in sorted(by_layer.keys(), reverse=True):  # entity (high layer) first
        with dot.subgraph() as s:
            s.attr(rank="same")
            for table in dict.fromkeys(by_layer[layer]):   # dedup, keep order
                nid = node_id(table, layer)
                if nid in seen:
                    continue
                seen.add(nid)
                s.node(nid, label=_table_label(
                    table, cols_by_table.get(table, []), cmask, delta=delta,
                    show_all_columns=show_all_columns,
                    is_entity=(table == entity_table and layer == max(by_layer)),
                    is_reachable=True, num_layers=num_layers,
                    layer_badge=f"hop {num_layers - layer}"))

    # edges: message passing flows from the deeper-layer neighbour into the table
    for (table, layer), edges in schema_dag.items():
        for e in edges:
            dt, dl = e["dst"]
            v = fk_by_name.get(e["edge_name"], 0.0)
            color, pen, style = _edge_style(v, True, delta)
            label = (" %.2f" % v) if v >= delta else ""
            dot.edge(node_id(dt, dl), node_id(table, layer), label=label,
                     color=color, penwidth=pen, style=style,
                     fontsize="9", fontcolor=color)

    if save_path is not None:
        dot.render(save_path, cleanup=True)
    return dot
