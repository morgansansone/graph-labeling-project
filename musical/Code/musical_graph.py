# musical_graph.py — edge irregularity strength of C(n,2)
# Morgan Sansone | graph labeling project, UMKC spring 2026
#
# edge irregularity strength (Ahmad et al. 2014):
#   - assign integer labels f: V -> {1, 2, ..., k}  (labels can repeat)
#   - edge weight of (u,v) = f(u) + f(v)
#   - all edge weights must be distinct
#   - es(G) = minimum k for which a valid labeling exists
#
# lower bound (theorem 1, Ahmad et al. 2014):
#   es(G) >= max( ceil((|E|+1)/2), Delta(G) )

import math
import os
import sys
import time
from graph_builder import build_graph, classify_edges, bfs_order

# check if assigning label c to vertex v causes any edge weight conflict
def is_valid(v, c, adj, label, used_weights):
    new_weights = set()
    for u in adj[v]:
        if label.get(u, 0) == 0:
            continue
        w = c + label[u]
        if w in used_weights or w in new_weights:
            return False, set()
        new_weights.add(w)
    return True, new_weights

# assign label c to v and record all new edge weights
def commit_label(v, c, adj, label, used_weights):
    label[v] = c
    for u in adj[v]:
        if label.get(u, 0) != 0:
            used_weights.add(c + label[u])

# remove v's label and all its edge weights from the used set
def revoke_label(v, adj, label, used_weights):
    c = label[v]
    for u in adj[v]:
        if label.get(u, 0) != 0:
            used_weights.discard(c + label[u])
    label[v] = 0

# try to find a valid labeling using labels in {1, ..., max_k}
# returns the label dict if successful, None if no valid labeling exists,
# or 'timeout' if the time limit is reached before a conclusion
def try_labeling(n, max_k, time_limit=30.0):
    adj, edges = build_graph(n)
    num_v = 2 * n
    order = bfs_order(adj, start=0)
    label = {}
    used_weights = set()
    cand = [1] * num_v
    pos = 0
    t0 = time.time()
    while 0 <= pos < num_v:
        if time.time() - t0 > time_limit:
            return 'timeout'
        v = order[pos]
        found = False
        for c in range(cand[pos], max_k + 1):
            valid, _ = is_valid(v, c, adj, label, used_weights)
            if valid:
                commit_label(v, c, adj, label, used_weights)
                cand[pos] = c
                found = True
                pos += 1
                break
        if not found:
            cand[pos] = 1
            pos -= 1
            if pos < 0:
                return None
            revoke_label(order[pos], adj, label, used_weights)
            cand[pos] += 1
    return label

# find the minimum k (edge irregularity strength) by searching from the lower bound up
# returns (label dict, k, adj, edges, status)
# status is 'exact', 'upper_bound' (if a smaller k timed out), or 'timeout'
def c2_backtrack(n, verbose=True):
    adj, edges = build_graph(n)
    num_v = 2 * n
    num_e = len(edges)
    lb = max(math.ceil((num_e + 1) / 2), 5)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  C2-Backtrack: Musical Graph C_({n},2)")
        print(f"  |V| = {num_v},  |E| = {num_e}")
        print(f"  Lower bound = {lb}")
        print(f"{'='*60}\n")

    status = 'exact'
    for k in range(lb, num_e + 1):
        result = try_labeling(n, k, time_limit=30.0)
        if result == 'timeout':
            # can't prove k infeasible — skip and try k+1 with a short limit
            if verbose:
                print(f"  k={k}: timeout (skipping)")
            status = 'upper_bound'
            continue
        if result is None:
            if verbose:
                print(f"  k={k}: infeasible")
            continue
        # found a valid labeling
        if verbose:
            print(f"  k={k}: feasible  [{status}]")
        return result, k, adj, edges, status

    return None, None, adj, edges, 'timeout'

# verify all edge weights are distinct and print a summary
def verify_solution(n, adj, edges, label, k, status='exact', verbose=True):
    weights = [label[u] + label[v] for (u, v) in edges]
    all_distinct = len(weights) == len(set(weights))
    if verbose:
        print(f"\n{'='*60}")
        print(f"  VERIFICATION")
        print(f"{'='*60}")
        print(f"  Total edges:          {len(weights)}")
        print(f"  Expected edges:       {5*n}")
        print(f"  All weights distinct: {all_distinct}")
        print(f"  Min edge weight:      {min(weights)}")
        print(f"  Max edge weight:      {max(weights)}")
        esline = f"  es(C_({n},2)) = k =    {k}"
        if status == 'upper_bound':
            esline += "  (upper bound — smaller k timed out)"
        print(esline)
    return all_distinct

# print vertex labels for outer and inner rings
def print_labels(n, label):
    print(f"\n{'='*60}")
    print(f"  VERTEX LABELS")
    print(f"{'='*60}")
    outer = [label.get(i, '-') for i in range(n)]
    inner = [label.get(n + i, '-') for i in range(n)]
    print(f"  Outer ring: {outer}")
    print(f"  Inner ring: {inner}")

# print all edge weights by category
def print_edge_weights(n, edges, label):
    print(f"\n{'='*60}")
    print(f"  EDGE WEIGHTS")
    print(f"{'='*60}")
    outer_e, inner_e, spokes_e, cross_e = classify_edges(n, edges)
    weights = {(u, v): label[u] + label[v] for (u, v) in edges}
    print("  Outer cycle edges:")
    for (u, v) in sorted(outer_e):
        print(f"    ({u}) -- ({v}) : {weights[(u,v)]}")
    print("  Inner cycle edges:")
    for (u, v) in sorted(inner_e):
        print(f"    ({u}) -- ({v}) : {weights[(u,v)]}")
    print("  Spoke edges:")
    for (u, v) in sorted(spokes_e):
        print(f"    ({u}) -- ({v}) : {weights[(u,v)]}")
    print("  Cross-chord edges:")
    for (u, v) in sorted(cross_e):
        print(f"    ({u}) -- ({v}) : {weights[(u,v)]}")

# run the algorithm for multiple values of n and print a results table
def run_experiment(n_values):
    print(f"\n{'='*60}")
    print(f"  RESULTS TABLE")
    print(f"{'='*60}")
    print(f"  {'n':>5} {'V':>6} {'E':>6} {'LowerBound':>12} {'es(G)':>8}  {'Status'}")
    print(f"  {'-'*55}")
    results = []
    for n in n_values:
        num_v = 2 * n
        num_e = 5 * n
        lb = max(math.ceil((num_e + 1) / 2), 5)
        result = c2_backtrack(n, verbose=False)
        label, k, adj, edges, status = result
        if label is not None:
            valid = verify_solution(n, adj, edges, label, k, status, verbose=False)
            results.append((n, num_v, num_e, lb, k, valid, status))
            mark = '✓' if valid else '✗'
            flag = '' if status == 'exact' else '  (upper bound)'
            print(f"  {n:>5} {num_v:>6} {num_e:>6} {lb:>12} {k:>8}  {mark}{flag}")
        else:
            print(f"  {n:>5} {num_v:>6} {num_e:>6} {lb:>12}  TIMEOUT")
    return results

if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data): [s.write(data) for s in self.streams]
        def flush(self): [s.flush() for s in self.streams]

    output_file = open(results_path, "w")
    sys.stdout = Tee(sys.__stdout__, output_file)

    # detailed run for n=5
    n = 5
    label, k, adj, edges, status = c2_backtrack(n)
    if label is not None:
        print_labels(n, label)
        verify_solution(n, adj, edges, label, k, status)
        print_edge_weights(n, edges, label)

    # tabulate results for n = 3, 4, 5, 6
    print("\n")
    run_experiment([3, 4, 5, 6])
    print(f"\n{'='*60}")

    sys.stdout = sys.__stdout__
    output_file.close()
    print(f"Results saved to {results_path}")
