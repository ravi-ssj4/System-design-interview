### DFS - 1D

```python
def solution(vertices, edges):
    adjList = { i: [] for i in range(vertices) }
    for a, b in edges:
        adjList[a].append(b)
        adjList[b].append(a)

    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in adjList[node]:
            if neighbor in visited:
                continue
            else:
                dfs(neighbor)
    dfs(0)
```

### DFS - 2D

```python
def solution(graph):
    ROWS, COLS = len(graph), len(graph[0])
    visited = set()

    def dfs(r, c):
        if (r not in range(ROWS) or
            c not in range(COLS) or
            (r, c) in visited):
            return False
        
        visited.add((r, c))

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    dfs(0, 0)

```

### BFS - 1D

```python

```

### BFS - 2D

```python

```