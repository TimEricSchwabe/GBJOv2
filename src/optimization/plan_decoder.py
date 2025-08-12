import itertools
import math
from typing import List, Tuple

import heapq
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian


def _score_plan(W: np.ndarray,
                join_order: Tuple[int, ...],
                table_rows: np.ndarray,
                slot_cols: np.ndarray,
                join_slots: List[int]) -> float:
    """Return ⟨W , A⟩ for the concrete plan encoded by the arguments."""
    # table→join part
    tbl_score = W[table_rows, [join_slots[c] for c in slot_cols]].sum()
    # join→join chain part
    jj_score = sum(W[join_order[k], join_order[k + 1]]
                   for k in range(len(join_order) - 1))
    return tbl_score + jj_score


def _table_assignment(W: np.ndarray,
                      tables: List[int],
                      join_slots: List[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run Hungarian on the n×n cost matrix (tables × join_slots).

    Returns   rows, cols, weight
    """
    cost = -W[np.ix_(tables, join_slots)]
    r, c = hungarian(cost)
    weight = -cost[r, c].sum()
    return r, c, weight


def _greedy_join_order(W: np.ndarray, joins: List[int]) -> List[int]:
    """O(n²) longest-path heuristic (backwards from every root candidate)."""
    best_path: List[int] = []
    best_score = -np.inf
    for root in joins:
        unused = set(joins)
        unused.remove(root)
        path = [root]
        score = 0.0
        while unused:
            # pick predecessor with max weight into current first element
            pred = max(unused, key=lambda j: W[j, path[0]])
            score += W[pred, path[0]]
            path.insert(0, pred)
            unused.remove(pred)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def project_to_leftdeep(W: np.ndarray,
                        exact_threshold: int = 10
                        ) -> np.ndarray:
    """
    Project W onto the nearest valid left-deep plan (Frobenius norm).

    Parameters
    ----------
    W : np.ndarray, shape (2n-1, 2n-1)
        Continuous adjacency matrix with entries in [0, 1].

    exact_threshold : int, optional
        Up to n == exact_threshold the search over join permutations is exhaustive
        (optimal).  Above that we switch to a greedy heuristic for speed.

    Returns
    -------
    A_hat : np.ndarray, shape (2n-1, 2n-1)
        Discrete {0,1} adjacency matrix of the projected plan.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    assert W.shape[1] == m and (m + 1) % 2 == 0, "shape must be (2n-1, 2n-1)"
    n = (m + 1) // 2
    tables = list(range(n))
    joins = list(range(n, 2 * n - 1))           # n-1 join-node indices

    best_weight = -np.inf
    best_plan = None            # (join_order, row_ind, col_ind, join_slots)

    #choose the join-node order
    if n - 1 <= exact_threshold:                 # exhaustive, optimal
        for join_order in itertools.permutations(joins):
            join_slots = [join_order[0]] + list(join_order)      # j₀ duplicated
            r, c, w_tbl = _table_assignment(W, tables, join_slots)
            w_tot = w_tbl + sum(
                W[join_order[k], join_order[k + 1]]
                for k in range(n - 2)
            )
            if w_tot > best_weight:
                best_weight = w_tot
                best_plan = (join_order, r, c, join_slots)
    else:                                        # heuristic
        join_order = tuple(_greedy_join_order(W, joins))
        join_slots = [join_order[0]] + list(join_order)
        r, c, _ = _table_assignment(W, tables, join_slots)
        best_plan = (join_order, r, c, join_slots)

    # 2) materialise the adjacency matrix
    join_order, row_ind, col_ind, join_slots = best_plan
    A = np.zeros_like(W, dtype=int)

    # table → join edges
    for r, c in zip(row_ind, col_ind):
        t = tables[r]
        j = join_slots[c]
        A[t, j] = 1

    # join → join chain
    for k in range(n - 2):                       # (n-1) joins → (n-2) edges
        A[join_order[k], join_order[k + 1]] = 1

    return A



@dataclass(order=True)
class _State:
    """Partial plan used by the beam search."""
    # heapq orders by the first field, so we negate the score to make the
    # *largest* score come out first.
    priority: float
    score:    float = field(compare=False)
    cur_join: int   = field(compare=False)
    unused_js: Set[int] = field(compare=False, repr=False)
    unused_tb: Set[int] = field(compare=False, repr=False)
    jj_edges:  List[Tuple[int, int]] = field(compare=False, repr=False)
    tj_edges:  List[Tuple[int, int]] = field(compare=False, repr=False)


def _best_two_tables(W_row: np.ndarray, tables: Set[int]) -> Tuple[List[int], float]:
    """Pick the two remaining tables with the largest weights into j0."""
    best = heapq.nlargest(2, tables, key=lambda t: W_row[t])
    return best, W_row[best].sum()


def project_leftdeep_greedy_beam(
    W: np.ndarray,
    beam_width: int = 1,
    use_product: bool = False,
) -> np.ndarray:
    """
    Greedy (beam_width=1) or beam-search projection of a soft plan adjacency matrix.

    Parameters
    ----------
    W : (2n-1, 2n-1) ndarray
        Continuous adjacency matrix (entries 0…1).

    beam_width : int, optional
        1  → pure greedy.  >1 → beam with that width.

    use_product : bool, optional
        If True, score pairs by product p_t * p_j, else by sum p_t + p_j.

    Returns
    -------
    Ahat : ndarray
        Discrete {0,1} adjacency matrix of the projected plan.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    assert m == W.shape[1] and (m + 1) % 2 == 0, "shape must be (2n-1, 2n-1)"
    n = (m + 1) // 2

    tables = set(range(n))
    joins_all = set(range(n, 2 * n - 1))
    root = 2 * n - 2                    # last index is fixed root

    #initialise the beam with the root node
    init_state = _State(
        priority=0.0,
        score=0.0,
        cur_join=root,
        unused_js=joins_all - {root},
        unused_tb=tables.copy(),
        jj_edges=[],
        tj_edges=[],
    )
    beam: List[_State] = [init_state]

    #grow the chain downward
    while beam and beam[0].unused_js:
        next_beam: List[_State] = []
        for state in beam:
            cj = state.cur_join
            for j in state.unused_js:
                for t in state.unused_tb:
                    w_t = W[t, cj]
                    w_j = W[j, cj]
                    pair_score = w_t * w_j if use_product else w_t + w_j
                    new_score = state.score + pair_score
                    new_state = _State(
                        priority=-new_score,
                        score=new_score,
                        cur_join=j,
                        unused_js=state.unused_js - {j},
                        unused_tb=state.unused_tb - {t},
                        jj_edges=state.jj_edges + [(j, cj)],
                        tj_edges=state.tj_edges + [(t, cj)],
                    )
                    heapq.heappush(next_beam, new_state)
        # keep best `beam_width` states
        beam = heapq.nsmallest(beam_width, next_beam)

    #finish the last join node (j0)
    best = beam[0]
    j0 = best.cur_join
    last_tables, add_score = _best_two_tables(W[:, j0], best.unused_tb)
    best.score += add_score
    best.tj_edges += [(last_tables[0], j0), (last_tables[1], j0)]

    #materialise adjacency matrix
    A = np.zeros_like(W, dtype=int)
    for j_from, j_to in best.jj_edges:
        A[j_from, j_to] = 1
    for t, j in best.tj_edges:
        A[t, j] = 1
    return A

if __name__ == "__main__":
    n = 5
    np.random.seed(0)
    W = np.random.rand(2*n-1, 2*n-1)
    print(W)

    # Greedy projection
    A_greedy = project_leftdeep_greedy_beam(W, beam_width=1)

    # Beam search with width 6, multiplicative scoring
    A_beam   = project_leftdeep_greedy_beam(W, beam_width=6, use_product=True)

    A_hat = project_to_leftdeep(W, exact_threshold=7)   # exact up to n=7

    print("Projected adjacency matrix:")
    print(A_hat.astype(int))