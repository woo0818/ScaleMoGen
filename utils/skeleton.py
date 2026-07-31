def adj_list_to_edges(adj_list):
    edges = []
    for i, adj in enumerate(adj_list):
        for j in adj:
            if i < j:
                edges.append((i, j))
    return edges

def edges_to_adj_list(edges):
    max_idx = -1
    for i, j in edges:
        max_idx = max(max_idx, i, j)

    adj_list = [[] for _ in range(max_idx + 1)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return adj_list