"""NumPy implementation of the existing five-state discrete PC-stable routine.

Only contingency-table construction is accelerated.  The G-squared statistic,
full-state-space degrees of freedom, stratum accumulation order, adjacency
snapshots, conditioning-set order, and removal criterion match
``sapa_hexaco_pc_bic.py``.  Unsupported input forms use that original routine.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _pvalue_from_array(
    values: np.ndarray, x: int, y: int, conditioning: tuple[int, ...]
) -> float:
    """Input is a complete, zero-based, five-state integer array."""
    if conditioning:
        stratum = np.zeros(len(values), dtype=np.int64)
        for column in conditioning:
            stratum = stratum * 5 + values[:, column]
        # pandas groupby(sort=False) visits observed strata in first-row order.
        observed, first_rows = np.unique(stratum, return_index=True)
        strata = observed[np.argsort(first_rows)]
        tables = np.bincount(
            stratum * 25 + values[:, x] * 5 + values[:, y],
            minlength=25 * (5 ** len(conditioning)),
        ).reshape(-1, 5, 5)
    else:
        tables = np.bincount(
            values[:, x] * 5 + values[:, y], minlength=25
        ).reshape(1, 5, 5)
        strata = (0,)

    g_stat = 0.0
    for stratum in strata:
        # Float conversion and summation order deliberately match the original.
        table = tables[stratum].astype(float)
        n = table.sum()
        if n <= 0:
            continue
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        valid = (table > 0) & (expected > 0)
        g_stat += 2 * np.sum(table[valid] * np.log(table[valid] / expected[valid]))
    dfree = 16 * (5 ** len(conditioning))
    return float(chi2.sf(g_stat, dfree))


def conditional_mi_pvalue_full_df(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning: tuple[str, ...],
    levels: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> float:
    """Drop-in five-state equivalent of the original G-squared CI test."""
    selected = data[[x, y, *conditioning]].dropna().astype(int)
    values = selected.to_numpy(dtype=np.int64)
    if levels != (1, 2, 3, 4, 5) or np.any((values < 1) | (values > 5)):
        from sapa_hexaco_pc_bic import conditional_mi_pvalue_full_df as original
        return original(data, x, y, conditioning, levels)
    return _pvalue_from_array(values - 1, 0, 1, tuple(range(2, values.shape[1])))


def pc_stable_edges(data: pd.DataFrame, alpha: float = 0.05) -> tuple[tuple[str, str], ...]:
    """Fit the original PC-stable skeleton using NumPy contingency counts."""
    if data.isna().any().any():
        from sapa_hexaco_pc_bic import pc_stable_edges as original
        return original(data, alpha)
    values = data.to_numpy(dtype=np.int64)
    if np.any((values < 1) | (values > 5)):
        from sapa_hexaco_pc_bic import pc_stable_edges as original
        return original(data, alpha)
    values = values - 1
    nodes = list(data.columns)
    column_index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = {node: set(nodes) - {node} for node in nodes}
    level = 0
    while True:
        adjacency_at_level = {node: set(neighbors) for node, neighbors in adjacency.items()}
        edges = [
            tuple(sorted((a, b)))
            for a in nodes
            for b in adjacency_at_level[a]
            if a < b
        ]
        for x, y in edges:
            if y not in adjacency[x]:
                continue
            candidates: list[tuple[str, ...]] = []
            for base in (adjacency_at_level[x] - {y}, adjacency_at_level[y] - {x}):
                if len(base) >= level:
                    candidates.extend(itertools.combinations(sorted(base), level))
            seen: set[tuple[str, ...]] = set()
            for conditioning in candidates:
                if conditioning in seen:
                    continue
                seen.add(conditioning)
                pvalue = _pvalue_from_array(
                    values, column_index[x], column_index[y],
                    tuple(column_index[node] for node in conditioning),
                )
                if pvalue > alpha:
                    adjacency[x].discard(y)
                    adjacency[y].discard(x)
                    break
        level += 1
        if not any(len(adjacency[node]) >= level + 1 for node in nodes):
            break
        if level > len(nodes) - 2:
            break
    return tuple(sorted(tuple(sorted((a, b))) for a in nodes for b in adjacency[a] if a < b))
