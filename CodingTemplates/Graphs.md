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

### BFS Level by level (min steps, min time, parallely capture/convert cells)

```python
def solution(grid):
    ROWS, COLS = len(grid), len(grid[0])

    distance = [[0] * COLS for r in range(ROWS)]
    q = deque()

    for r in range(ROWS):
        for c in range(COLS):
            if some_condition:
            # if (grid[r][c] == 1 and 
            #     r in [0, ROWS - 1] or 
            #     c in [0, COLS - 1]):
                visited.add((r, c))
                q.append([r, c, 0])
    
    while q:
        r, c, steps = q.popleft()
        # process the values popped
        distance[r][c] = steps
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for nr, nc in directions:
            row, col = r + nr, c + nc
            if (row not in range(ROWS) or
                col not in range(COLS) or
                (row, col) in visited):
                continue
            else:
                visited.add((row, col))
                q.append([row, col, steps + 1])


```

### Graph Coloring: Check if graph is bipartite ( 2 colorable ) - BFS

```python
def solution(n, edges):
    # create the adj list
    # visited set not required in this case!
    color = [-1] * n # we have color array instead!

    def isBipartiteBFS(start):
        q = deque()
        q.append(start)
        color[start] = 0

        while q:
            node = q.popleft()
            for neighbor in adjList[node]:
                # if the neighbor is not colored
                if color[neighbor] == -1:
                    color[neighbor] = not color[node]
                    q.append(neighbor)
                else: # if the neighbor is already colored beforehand
                    # if neighbor's color ==== my color: not a bipartite!
                    if color[neighbor] == color[node]:
                        return False
        return True

    for i in range(n):
        if color[i] == -1:
            if isBipartiteDFS(i) == False:
                return False
    return True

```

### Graph Coloring: Check if graph is bipartite ( 2 colorable ) - DFS

```python
def solution(n, edges):
    # create the adj list
    # visited set not required in this case!
    color = [-1] * n # we have color array instead!

    def isBipartiteDFS(node, newColor):
        color[node] = newColor
        for neighbor in adjList[node]:
            # if the neighbor is not colored
            if color[neighbor] == -1:
                # pass opp of current color for next call
                isBipartiteDFS(neighbor, not newColor)
            else: # if the neighbor is already colored beforehand
                # if neighbor's color ==== my color: not a bipartite!
                if color[neighbor] == color[node]:
                    return False
        return True

    for i in range(n):
        if color[i] == -1:
            if isBipartiteDFS(i, 0) == False:
                return False
    return True

```

### Detect Cycle Directed Graph - DFS
```python
def solution(n, edges):
    # create adjList
    visited = set()
    pathVisited = set()

    def detectCycleDFS(node):
        visited.add(node)
        pathVisited.add(node)

        for neighbor in adjList[node]:
            if neighbor not in visited:
                if detectCycleDFS(node) == True:
                    return True
            else:
                if neighbor in pathVisited:
                    return True
        pathVisited.remove(node)
        return False
    
    for i in range(n):
        if i not in visited:
            if detectCycleDFS(i) == True:
                return True
    return False

```

### Find eventual safe states - DFS
```python
def solution(n, edges):
    # create adjList
    res = []
    visited = set()
    pathVisited = set()

    # starting from this node, if below 2 conditions are met, the node is safe
    # 1. its part of a cycle
    # 2. it leads to a cycle
    # in both cases, we return True from the for loop itself
    # otherwise, we exit the for loop and add the node into our answer as safe
    # and return False
    def dfsCheck(node):
        visited.add(node)
        pathVisited.add(node)

        for neighbor in adjList[node]:
            if neighbor not in visited:
                if dfsCheck(node) == True:
                    return True
            else:
                if neighbor in pathVisited:
                    return True
        res.append(node)
        pathVisited.remove(node)
        return False

    for i in range(n):
        if i not in visited:
            dfsCheck(i)

    return res


```

### Topological Sort
```python

```

### Detect cycle Directed Graph - Topological sort
```python

```

### Find eventual safe states - Topological sort
```python

```

