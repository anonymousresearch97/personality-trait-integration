"""Faster implementation of the existing five-state exact discrete BIC search.

The score, parent-set iteration order, strict ``>`` comparisons, sink order,
and final edge ordering match ``hexaco_bic_exhaustive.exact_bic_dag``. Only
counting and reusable structural bookkeeping are optimized. Parent-state
codes are shared across children, and ``numpy.bincount`` replaces ``add.at``.
Inputs must be the same discrete DataFrames with integer states 1 through 5.
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import math

import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def _structure(p: int):
    """Cache the reference implementation's enumeration order, without scores."""
    full = (1 << p) - 1
    cardinalities = tuple(5 ** bin(mask).count("1") for mask in range(full + 1))
    encoding_steps = []
    for mask in range(1, full):
        lowest_bit = mask & -mask
        variable = lowest_bit.bit_length() - 1
        remainder = mask ^ lowest_bit
        encoding_steps.append((mask, variable, remainder, cardinalities[remainder]))

    child_masks = []
    allowed_submasks = []
    for child in range(p):
        other_nodes = [i for i in range(p) if i != child]
        child_masks.append(tuple(
            sum(1 << parent for parent in parents)
            for size in range(len(other_nodes) + 1)
            for parents in itertools.combinations(other_nodes, size)
        ))
        allowed_sets = []
        for allowed in range(1 << p):
            if allowed & (1 << child):
                continue
            masks = []
            parent_mask = allowed
            while True:
                masks.append(parent_mask)
                if parent_mask == 0:
                    break
                parent_mask = (parent_mask - 1) & allowed
            allowed_sets.append((allowed, tuple(masks)))
        allowed_submasks.append(tuple(allowed_sets))

    subsets = tuple(
        (sum(1 << node for node in subset_nodes), subset_nodes)
        for size in range(1, p + 1)
        for subset_nodes in itertools.combinations(range(p), size)
    )
    return cardinalities, tuple(encoding_steps), tuple(child_masks), tuple(allowed_submasks), subsets


def exact_bic_dag(discrete: pd.DataFrame) -> tuple[float, tuple[tuple[str, str], ...]]:
    """Return the same score and selected DAG as the existing exact BIC search."""
    nodes = list(discrete.columns)
    p = len(nodes)
    if p == 0:
        return 0.0, tuple()
    arr = discrete[nodes].to_numpy(dtype=int) - 1
    n = arr.shape[0]
    log_n = math.log(n)
    full = (1 << p) - 1
    cardinalities, encoding_steps, child_masks, allowed_submasks, subsets = _structure(p)

    # Digits follow increasing parent-column order, exactly as in the reference.
    # Every proper subset can be used by at least one child; the full set cannot.
    parent_codes = [None] * (full + 1)
    parent_codes[0] = np.zeros(n, dtype=int)
    for mask, variable, remainder, multiplier in encoding_steps:
        parent_codes[mask] = arr[:, variable] * multiplier + parent_codes[remainder]

    best_parent_score = []
    for child in range(p):
        child_values = arr[:, child]
        child_scores = {}
        for mask in child_masks[child]:
            q = cardinalities[mask]
            indices = parent_codes[mask] * 5 + child_values
            counts = np.bincount(indices, minlength=q * 5).reshape(q, 5).astype(float)
            row_totals = counts.sum(axis=1, keepdims=True)
            nonzero = counts > 0
            probs = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
            loglik = float(np.sum(counts[nonzero] * np.log(probs[nonzero])))
            n_parameters = q * (5 - 1)
            child_scores[mask] = loglik - 0.5 * n_parameters * log_n

        best_for_allowed = {}
        for allowed, masks in allowed_submasks[child]:
            best_score = -np.inf
            best_mask = 0
            for parent_mask in masks:
                score = child_scores[parent_mask]
                if score > best_score:
                    best_score = score
                    best_mask = parent_mask
            best_for_allowed[allowed] = (best_score, best_mask)
        best_parent_score.append(best_for_allowed)

    dp = {0: 0.0}
    choice = {}
    for subset, subset_nodes in subsets:
        best = -np.inf
        best_child = -1
        best_parents = 0
        for sink in subset_nodes:
            remaining = subset & ~(1 << sink)
            family_score, parent_mask = best_parent_score[sink][remaining]
            score = dp[remaining] + family_score
            if score > best:
                best = score
                best_child = sink
                best_parents = parent_mask
        dp[subset] = best
        choice[subset] = (best_child, best_parents)

    edges = []
    subset = full
    while subset:
        child, parents = choice[subset]
        for parent in range(p):
            if parents & (1 << parent):
                edges.append((nodes[parent], nodes[child]))
        subset &= ~(1 << child)
    return dp[full], tuple(sorted(edges))
