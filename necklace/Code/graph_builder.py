from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────────────────────────────────────
# Constructs the necklace graph N_{l,3} as an undirected graph represented by
# an adjacency list. The graph models a necklace structure with three parallel
# rows (top, middle, bottom) connected by vertical edges, cross-sections at both
# ends, and outer arcs that complete the necklace topology.
#
# Vertex structure:
#  - Top row: x₁, x₂, ..., x_l
#  - Middle row: m₂, m₃, ..., m_{l-1} (shorter by 2 vertices)
#  - Bottom row: y₁, y₂, ..., y_l
#  - Left interior vertex: iL (connects to left cross-section)
#  - Right interior vertex: iR (connects to right cross-section)
# Returns: defaultdict(list) - adjacency list mapping vertices to their neighbors.

def build_necklace_graph(l):
    """Build adjacency list for N_{l,3} necklace graph.
    Returns: defaultdict(list) - undirected graph where each vertex maps to a list
             of its adjacent vertices in the necklace graph structure.
    """
    adj = defaultdict(list)
    
    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
    
    # ── Top path: x1-x2-...-xl ──────────────────────────
    for i in range(1, l):
        add_edge(('x', i), ('x', i+1))
    
    # ── Bottom path: y1-y2-...-yl ───────────────────────
    for i in range(1, l):
        add_edge(('y', i), ('y', i+1))
    
    # ── Middle path: m2-m3-...-m(l-1) ───────────────────
    for i in range(2, l-1):
        add_edge(('m', i), ('m', i+1))
    
    # ── Interior verticals (i=2 to l-1) ─────────────────
    for i in range(2, l):
        add_edge(('x', i), ('m', i))
        add_edge(('m', i), ('y', i))
    
    # ── Left cross section (6 new edges) ────────────────
    add_edge(('x', 1), ('iL', 0))
    add_edge(('x', 2), ('iL', 0))
    add_edge(('x', 1), ('m',  2))
    add_edge(('iL', 0), ('y', 1))
    add_edge(('iL', 0), ('y', 2))
    add_edge(('m',  2), ('y', 1))
    
    # ── Right cross section (6 new edges) ───────────────
    add_edge(('x', l),   ('iR', 0))
    add_edge(('x', l-1), ('iR', 0))
    add_edge(('x', l),   ('m',  l-1))
    add_edge(('iR', 0),  ('y', l))
    add_edge(('iR', 0),  ('y', l-1))
    add_edge(('m',  l-1),('y', l))
    
    # ── Outer arcs ───────────────────────────────────────
    add_edge(('x', 1), ('x', l))
    add_edge(('y', 1), ('y', l))
    
    return adj
