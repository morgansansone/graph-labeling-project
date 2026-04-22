# graph_builder.py — musical graph C(n,2) construction
# Morgan Sansone | graph labeling project, UMKC spring 2026

from collections import deque

# build C(n,2): outer ring (0..n-1) + inner ring (n..2n-1)
# each outer vertex i connects to: outer cycle neighbors, inner spoke, inner cross-chords
def build_graph(n):
    num_v = 2 * n
    adj = [[] for _ in range(num_v)]
    edge_set = set()

    def add_edge(u, v):
        a, b = min(u, v), max(u, v)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            adj[u].append(v)
            adj[v].append(u)

    for i in range(n):
        add_edge(i, (i + 1) % n)           # outer cycle
        add_edge(i, (i - 1) % n)
        add_edge(n + i, n + (i + 1) % n)   # inner cycle
        add_edge(n + i, n + (i - 1) % n)
        add_edge(i, n + i)                  # spoke
        add_edge(i, n + (i + 1) % n)       # cross-chords
        add_edge(i, n + (i - 1) % n)

    for i in range(num_v):
        adj[i].sort()

    return adj, sorted(edge_set)

# sort edges into outer cycle, inner cycle, spokes, and cross-chords
def classify_edges(n, edges):
    outer, inner, spokes, cross = [], [], [], []
    for (u, v) in edges:
        u_out = u < n
        v_out = v < n
        if u_out and v_out:
            outer.append((u, v))
        elif not u_out and not v_out:
            inner.append((u, v))
        else:
            o = u if u_out else v
            iv = v if u_out else u
            if iv == n + o:
                spokes.append((u, v))
            else:
                cross.append((u, v))
    return outer, inner, spokes, cross

# return vertices in bfs order starting from start
def bfs_order(adj, start=0):
    visited = [False] * len(adj)
    order, q = [], deque([start])
    visited[start] = True
    while q:
        v = q.popleft()
        order.append(v)
        for nb in adj[v]:
            if not visited[nb]:
                visited[nb] = True
                q.append(nb)
    return order
