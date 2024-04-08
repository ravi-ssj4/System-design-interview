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
        # process node + other things
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
            return
        
        visited.add((r, c))
        # process (r, c) + other things
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    dfs(0, 0)

```

### BFS - 1D

```python
def solution(vertices, connections):
    adjList = { i:[] for i in range(vertices) }
    for a, b in edges:
        adjList[a].append(b)
        adjList[b].append(a)

    visited = set()

    def bfs(node):
        q = collections.deque(0)
        q.append(node)
        visited.add(node)
        while q:
            node = q.popleft()
            # process node + other things
            for neighbor in adjList[node]:
                if neighbor in visited:
                    continue
                else:
                    visited.add(neighbor)
                    q.append(neighbor)

    for node in range(vertices):
        if some_condition:
            if node not in visited:
                visited.add(node)
                bfs(node)


    
```

### BFS - 2D

```python
    def solution(graph):
        ROWS, COLS = len(graph), len(graph[0])
        visited = set()

        def bfs(r, c):
            q = collections.deque()
            q.append([r, c])
            visited.add((r, c))
            while q:
                row, col = q.popleft()
                # process (row, col) + other things
                neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
                for nr, nc in neighbors:
                    r, c = row + nr, col + nc
                    if (r not in range(ROWS) or
                        c not in range(COLS) or
                        (r, c) in visited):
                        continue
                    visited.add((r, c))
                    q.append([r, c])
        
        for r in range(ROWS):
            for c in range(COLS):
                if some_condition:
                    bfs(r, c)
    
```

# Some important code snippets:

### Detect Cycle Undirected using BFS
```python
def Solution(adjList):
    num_vertices = len(adjList)
    visited = set()

    def detectCycleBFS(node, parent):
        q = deque()
        q.append([node, parent])
        visited.add(node)
        while q:
            node, parent = q.popleft()
            for neighbor in adjList[node]:
                if neighbor not in visited:
                    visited.add(node)
                    q.append([neighbor, node])
                else:
                    if neighbor != parent:
                        return True
        return False

    # to take care of multiple Connected components
    for i in range(num_vertices):
        if i not in visited:
            if detectCycleBFS(i, -1) == True:
                return True
    return False
```

### Detect Cycle Undirected using DFS
```python
def Solution(adjList):
    num_vertices = len(adjList)
    visited = set()

    def detectCycleDFS(node, parent):
        visited.add(node)
        for neighbor in adjList[node]:
            if neighbor not in visited:
                if detectCycleDFS(neighbor, node) == True:
                    return True
            else:
                if neighbor != parent:
                    return True
        return False
    
    # to take care of multiple connected components
    for i in range(num_vertices):
        if i not in visited:
            if detectCycleDFS(i, -1) == True:
                return True
    return False
```