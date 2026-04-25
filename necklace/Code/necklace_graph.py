from graph_builder import build_necklace_graph

# ──────────────────────────────────────────────────────────────────────────────
# Vertex Ordering
# ──────────────────────────────────────────────────────────────────────────────
# Determines the order in which vertices are processed during backtracking.
# High-degree vertices (m2 and m_{l-1}) are placed last to reduce backtracking.
# Returns: List of vertices in the order they will be labeled.

def build_order(l):
    """Fixed vertex processing order for backtracking.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
                 Determines the structure and size of the vertex ordering.
    
    Returns:
        list: Ordered list of vertices (tuples) to be processed during backtracking.
              High-degree vertices (m2 and m_{l-1}) are placed last to reduce backtracks.
    """
    order = []
    
    # Left cross section (m2 last — degree 5)
    order += [('x',1), ('x',2), ('iL',0), ('y',1), ('y',2), ('m',2)]
    
    # Interior columns left to right
    for i in range(3, l-1):
        order += [('x',i), ('m',i), ('y',i)]
    
    # Right cross section (m_{l-1} last — degree 5)
    order += [('x',l-1), ('x',l), ('iR',0), ('y',l-1), ('y',l), ('m',l-1)]
    
    return order


# ──────────────────────────────────────────────────────────────────────────────
# Validation and Constraint Checking
# ──────────────────────────────────────────────────────────────────────────────
# These functions verify that a vertex label assignment doesn't violate the
# edge irregularity constraint (all edge weights must be distinct).

def is_valid(v, c, adj, label, used_weights):
    """
    Check if assigning label c to vertex v causes
    any edge weight conflict with already-labeled neighbors.
    
    Args:
        v (tuple): Vertex identifier (type, index) to check.
        c (int): Candidate label value to test for vertex v.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Current vertex labels, mapping vertices to their assigned values.
        used_weights (set): Set of edge weights already committed in the current solution.
    
    Returns:
        tuple: (bool, set) - True with set of new edge weights if valid,
               False with empty set if conflict detected.
    """
    new_weights = set()
    
    for u in adj[v]:
        if label.get(u, 0) == 0:
            continue                    # neighbor not yet labeled
        
        w = c + label[u]
        
        if w in used_weights:           # conflict with existing edge
            return False, set()
        
        if w in new_weights:            # conflict within v's own edges
            return False, set()
        
        new_weights.add(w)
    
    return True, new_weights


# ──────────────────────────────────────────────────────────────────────────────
# Label State Management
# ──────────────────────────────────────────────────────────────────────────────
# Functions to commit and revoke vertex labels as the backtracking algorithm
# explores the search space.

def commit_label(v, c, adj, label, used_weights):
    """Assign label c to v and record all new edge weights.
    
    Args:
        v (tuple): Vertex identifier (type, index) to label.
        c (int): Label value to assign to vertex v.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Current vertex labels (modified in-place).
        used_weights (set): Set of edge weights (modified in-place with new values).
    
    Returns:
        None (modifies label and used_weights in-place).
    """
    label[v] = c
    for u in adj[v]:
        if label.get(u, 0) != 0:
            used_weights.add(c + label[u])


def revoke_label(v, adj, label, used_weights):
    """Remove v's label and all its edge weights from used set.
    
    Args:
        v (tuple): Vertex identifier (type, index) whose label is to be removed.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Current vertex labels (modified in-place, vertex v reset to 0).
        used_weights (set): Set of edge weights (modified in-place, v's weights removed).
    
    Returns:
        None (modifies label and used_weights in-place).
    """
    c = label[v]
    for u in adj[v]:
        if label.get(u, 0) != 0:
            used_weights.discard(c + label[u])
    label[v] = 0


# ──────────────────────────────────────────────────────────────────────────────
# Main Backtracking Algorithm
# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm that assigns integer labels to vertices using depth-first
# backtracking with constraint satisfaction. It tries labels in increasing order
# and backtracks when a conflict is detected. Returns the minimal labeling that
# achieves edge irregularity (all edge weights distinct) and the maximum label
# used (edge irregularity strength).

def nl3_backtrack(l):
    """
    Main backtracking algorithm for N_{l,3} necklace graph.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
                 Defines the necklace graph N_{l,3} structure.
    
    Returns:
        tuple: (dict, int) - vertex labels mapping vertices to their assigned values
               and edge irregularity strength k, or (None, None) if no solution found.
    """
    print(f"\n{'='*60}")
    print(f"  NL3-Backtrack: Necklace Graph N_({l},3)")
    print(f"  |V| = {3*l},  |E| = {5*l+5}")
    print(f"  Lower bound = {(5*l+6)//2}")
    print(f"{'='*60}\n")
    
    adj          = build_necklace_graph(l)
    order        = build_order(l)
    label        = {}                   # vertex → label
    used_weights = set()                # committed edge weights
    cand         = [1] * len(order)     # current candidate per position
    pos          = 0                    # current position in order
    k            = 0                    # max label used
    backtracks   = 0
    
    MAX_LABEL = 10 * l                  # safe upper bound
    
    while 0 <= pos < len(order):
        v = order[pos]
        found = False
        
        # Try candidates from cand[pos] upward
        for c in range(cand[pos], MAX_LABEL + 1):
            valid, new_weights = is_valid(v, c, adj, label, used_weights)
            
            if valid:
                commit_label(v, c, adj, label, used_weights)
                cand[pos] = c
                k = max(k, c)
                found = True
                pos += 1               # move forward
                break
        
        if not found:
            # ── BACKTRACK ───────────────────────────────
            backtracks += 1
            cand[pos] = 1              # reset this position
            pos -= 1                   # go back one step
            
            if pos < 0:
                print("No solution found!")
                return None, None
            
            # Revoke previous vertex and try next candidate
            prev_v = order[pos]
            revoke_label(prev_v, adj, label, used_weights)
            cand[pos] += 1             # increment to try next
    
    print(f"  Backtracks: {backtracks}")
    return label, k


# ──────────────────────────────────────────────────────────────────────────────
# Solution Analysis Functions
# ──────────────────────────────────────────────────────────────────────────────
# Functions to compute and verify edge weights from a valid vertex labeling.

def compute_edge_weights(l, adj, label):
    """Compute all edge weights and return as dict.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Vertex labels mapping vertices to their assigned values.
    
    Returns:
        dict: Maps (vertex, vertex) pairs to their edge weight (sum of labels).
    """
    edge_weights = {}
    seen = set()
    
    for v in label:
        for u in adj[v]:
            edge = tuple(sorted([str(v), str(u)]))
            if edge not in seen:
                seen.add(edge)
                edge_weights[(v, u)] = label[v] + label[u]
    
    return edge_weights


# Verify that the computed edge weights satisfy the irregularity constraint
# (all edge weights are distinct). Reports statistics and identifies duplicates.

def verify_solution(l, adj, label, k):
    """Verify all edge weights are distinct.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Vertex labels mapping vertices to their assigned values.
        k (int): Maximum label used (edge irregularity strength).
    
    Returns:
        bool: True if all edge weights are unique, False otherwise.
    """
    edge_weights = compute_edge_weights(l, adj, label)
    weights      = list(edge_weights.values())
    all_distinct = len(weights) == len(set(weights))
    
    print(f"\n{'='*60}")
    print(f"  VERIFICATION")
    print(f"{'='*60}")
    print(f"  Total edges:          {len(weights)}")
    print(f"  Expected edges:       {5*l+5}")
    print(f"  All weights distinct: {all_distinct}")
    print(f"  Min edge weight:      {min(weights)}")
    print(f"  Max edge weight:      {max(weights)}")
    print(f"  es(N_{{{l},3}}) = k =    {k}")
    
    if not all_distinct:
        # Find and report duplicates
        from collections import Counter
        counts = Counter(weights)
        dups   = {w: c for w, c in counts.items() if c > 1}
        print(f"  DUPLICATE WEIGHTS:    {dups}")
    
    return all_distinct


# ──────────────────────────────────────────────────────────────────────────────
# Output and Reporting Functions
# ──────────────────────────────────────────────────────────────────────────────
# Functions to display and visualize the results in organized formats.

def print_labels(l, label):
    """Print vertex labels in grid format.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
        label (dict): Vertex labels mapping vertices to their assigned values.
    
    Returns:
        None (displays output to console).
    """
    print(f"\n{'='*60}")
    print(f"  VERTEX LABELS")
    print(f"{'='*60}")
    
    top    = [label.get(('x',  i), '-') for i in range(1, l+1)]
    middle = [' '] + [label.get(('m', i), '-') for i in range(2, l)] + [' ']
    bottom = [label.get(('y',  i), '-') for i in range(1, l+1)]
    
    print(f"  Top row:    {top}")
    print(f"  Middle row: {middle}")
    print(f"  Bottom row: {bottom}")
    print(f"  iL = {label.get(('iL',0), '-')}")
    print(f"  iR = {label.get(('iR',0), '-')}")


# Display all edge weights organized by category (top path, middle path,
# bottom path, vertical connections, cross sections, and outer arcs).

def print_edge_weights(l, adj, label):
    """Print all edge weights organized by category.
    
    Args:
        l (int): Number of vertices in each of the top and bottom rows.
        adj (defaultdict): Adjacency list mapping vertices to their neighbors.
        label (dict): Vertex labels mapping vertices to their assigned values.
    
    Returns:
        None (displays output to console).
    """
    print(f"\n{'='*60}")
    print(f"  EDGE WEIGHTS")
    print(f"{'='*60}")
    
    edge_weights = compute_edge_weights(l, adj, label)
    
    # Organize by category
    top_edges    = []
    mid_edges    = []
    bot_edges    = []
    vert_edges   = []
    cross_edges  = []
    arc_edges    = []
    
    for (u, v), w in edge_weights.items():
        u_type = u[0]
        v_type = v[0]
        
        if u_type == 'x' and v_type == 'x':
            if abs(u[1] - v[1]) == 1:
                top_edges.append((u, v, w))
            else:
                arc_edges.append((u, v, w))
        elif u_type == 'y' and v_type == 'y':
            if abs(u[1] - v[1]) == 1:
                bot_edges.append((u, v, w))
            else:
                arc_edges.append((u, v, w))
        elif u_type == 'm' and v_type == 'm':
            mid_edges.append((u, v, w))
        elif (u_type in ['x','y'] and v_type == 'm') or \
             (v_type in ['x','y'] and u_type == 'm'):
            vert_edges.append((u, v, w))
        else:
            cross_edges.append((u, v, w))
    
    print(f"\n  Top path edges:")
    for u,v,w in sorted(top_edges, key=lambda x: x[0][1]):
        print(f"    {u} -- {v} : {w}")
    
    print(f"\n  Middle path edges:")
    for u,v,w in sorted(mid_edges, key=lambda x: x[0][1]):
        print(f"    {u} -- {v} : {w}")
    
    print(f"\n  Bottom path edges:")
    for u,v,w in sorted(bot_edges, key=lambda x: x[0][1]):
        print(f"    {u} -- {v} : {w}")
    
    print(f"\n  Vertical edges (top/bot <-> middle):")
    for u,v,w in sorted(vert_edges):
        print(f"    {u} -- {v} : {w}")
    
    print(f"\n  Cross section edges (iL/iR):")
    for u,v,w in sorted(cross_edges):
        print(f"    {u} -- {v} : {w}")
    
    print(f"\n  Outer arc edges:")
    for u,v,w in arc_edges:
        print(f"    {u} -- {v} : {w}")


# ──────────────────────────────────────────────────────────────────────────────
# Experimental Framework
# ──────────────────────────────────────────────────────────────────────────────
# Execute and tabulate the backtracking algorithm across multiple graph sizes
# to analyze the edge irregularity strength as a function of parameter l.

def run_experiment(l_values):
    """
    Run algorithm for multiple values of l and tabulate results.
    
    Args:
        l_values (list): List of integers representing the parameter l values
                         to test for different graph sizes.
    
    Returns:
        list: Tuples of (l, V, E, lower_bound, edge_strength, is_valid)
              for each tested graph size.
    """
    print(f"\n{'='*60}")
    print(f"  RESULTS TABLE")
    print(f"{'='*60}")
    print(f"  {'l':>5} {'V':>6} {'E':>6} "
          f"{'LowerBound':>12} {'es(G)':>8}")
    print(f"  {'-'*45}")
    
    results = []
    
    for l in l_values:
        label, k = nl3_backtrack(l)
        
        if label is not None:
            adj    = build_necklace_graph(l)
            valid  = verify_solution(l, adj, label, k)
            lb     = (5*l + 6) // 2
            V      = 3 * l
            E      = 5 * l + 5
            
            results.append((l, V, E, lb, k, valid))
            
            print(f"  {l:>5} {V:>6} {E:>6} {lb:>12} {k:>8}"
                  f"  {'✓' if valid else '✗'}")
    
    return results


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # ── Detailed run for l=8 (the image graph) ───────────
    l = 8
    adj      = build_necklace_graph(l)
    label, k = nl3_backtrack(l)
    
    if label is not None:
        print_labels(l, label)
        verify_solution(l, adj, label, k)
        print_edge_weights(l, adj, label)
    
    # ── Tabulate results for multiple values of l ─────────
    print("\n")
    run_experiment([4, 5, 6, 7, 8, 10, 15, 20, 50])
