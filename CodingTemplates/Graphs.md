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
    
    for i in range(vertices):
        if i not in visited:
            dfs(i)
```

### DFS - 2D

```python
def solution(graph):
    ROWS, COLS = len(graph), len(graph[0])
    visited = set()

    def dfs(row, col):
        visited.add((row, col)) # we always visit right at the start of the dfs function
        # process node(row, col)
        neighbors = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for nr, nc in neighbors:
            r = row + nr
            c = col + nc
            if (r in range(ROWS) or
                c in range(COLS) or
                (r, c) not in visited):
                
                dfs(r, c)
            
    # to take care of multiple connected components
    for r in range(ROWS):
        for c in range(COLS):
        if (r, c) not in visited:
            dfs(r, c)

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
        
        q = collections.deque()
        visited.add(node) # we always visit before adding to the queue
        q.append(node)
        
        while q:
            node = q.popleft()
            
            # process node + other things
            
            for neighbor in adjList[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)

    for node in range(vertices):
        if some_condition:
            if node not in visited:
                bfs(node)


    
```

### BFS - 2D

```python
    def solution(graph):
        ROWS, COLS = len(graph), len(graph[0])
        visited = set()

        def bfs(row, col):
            q = collections.deque()
            visited.add((row, col))
            q.append([row, col])

            while q:

                row, col = q.popleft()
                
                # process node(row, col) + other things
                
                neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]

                for nr, nc in neighbors:
                    r = row + nr
                    c = col + nc
                    if (r in range(ROWS) or
                        c in range(COLS) or
                        (r, c) not in visited):

                        visited.add((r, c))
                        q.append([r, c])
        
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in visited:
                    bfs(r, c)
    
```

# Some important code snippets:

### Number of provinces | adj Matrix given instead of adj List | VVI

```python
class Solution:
    def numProvinces(self, adj, V):
        ROWS, COLS = len(adj), len(adj[0])
        
        provinces = 0
        
        visited = set()
        
        def dfs(node):
            visited.add(node)
            
            for vertex, connection in enumerate(adj[node]):
                if connection == 1:
                    if vertex not in visited:
                        dfs(vertex)
        
        for i in range(V):
            if i not in visited:
                dfs(i)
                provinces += 1
        
        return provinces
```

### Number of islands

```python
def numIslands(self,grid):
        ROWS, COLS = len(grid), len(grid[0])
        
        islands = 0
        
        visited = set()
        
        def dfs(row, col):
            
            visited.add((row, col))
            
            neighbors = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [1, -1], [-1, -1], [1, 1]]
            
            for nr, nc in neighbors:
                r = row + nr
                c = col + nc
                if (r in range(ROWS) and
                    c in range(COLS) and
                    (r, c) not in visited and
                    grid[r][c] == 1):
                    
                    dfs(r, c)
            
        
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in visited and grid[r][c] == 1:
                    dfs(r, c)
                    islands += 1
                    
        return islands
```

### Flood Fill Algorithm

```python
class Solution:
	def floodFill(self, image, sr, sc, newColor):
	    ROWS, COLS = len(image), len(image[0])
	    
	    answer = [[0] * COLS for _ in range(ROWS)]
	    
	    for r in range(ROWS):
	        for c in range(COLS):
	            answer[r][c] = image[r][c]
	    
	    startingPixelColor = image[sr][sc]
	    
	    def dfs(row, col):
	        answer[row][col] = newColor
		    
		    neighbors = [[-1, 0], [1, 0], [0, 1], [0, -1]]
		    
		    for nr, nc in neighbors:
		        r = row + nr
		        c = col + nc
		        
		        if (r in range(ROWS) and
		            c in range(COLS) and
		            answer[r][c] != newColor and
		            image[r][c] == startingPixelColor):
		            
		            dfs(r, c)
		
		dfs(sr, sc)
		
		return answer
```

### Rotting Oranges

```python
def orangesRotting(self, grid):
		ROWS, COLS = len(grid), len(grid[0])
		
		visited = set()
		
		q = collections.deque()
		fresh = 0
		
		for r in range(ROWS):
		    for c in range(COLS):
		        if (r, c) not in visited:
		            if grid[r][c] == 2:
		                q.append((r, c))
		                visited.add((r, c))
		          
		            if grid[r][c] == 1:
		                fresh += 1
	    
	    time = 0
	    while q:
	        n = len(q)
	        for _ in range(n):
	            
	            row, col = q.popleft()
	            
	            neighbors = [[-1, 0], [1, 0], [0, 1], [0, -1]]
	            
	            for drow, dcol in neighbors:
	                r = row + drow
	                c = col + dcol
	                
	                if (r in range(ROWS) and
	                    c in range(COLS) and
	                    (r, c) not in visited and
	                    grid[r][c] == 1):
	                   
	                   visited.add((r, c))
	                   q.append((r, c))
	                   grid[r][c] = 2
	                   fresh -= 1
	        if q:
	            time += 1
	   
	    if fresh == 0:
	        return time
	    else:
	        return -1
```

### Detect Cycle Undirected Graph using BFS
Always keep track of the parent. <br>
If during BFS, a node's neighbor is already visited:
<br>If its a parent, then fine, otherwise it contains a cycle

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

### Detect Cycle Undirected Graph using DFS
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

### Multisource BFS General structure (min steps, min time, parallely capture/convert cells)

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

Q-1. Distance of nearest Cell having 1(gfg) | 0/1 matrix(leetcode)
```python
class Solution:
    '''
    * All cells having a value of 0 will have a distance of 0 from the nearest 0 -> push all such cells onto the queue with distance = 0
    * do a bfs on the elements of the queue:
        * pop [r, c, distance] from the queue and update distance[r][c] = dist
        * iterate through the neighbors and if unvisited, push them also onto the queue with (distance = distance + 1)
    * return the distance grid
    '''
    #Function to find distance of nearest 1 in the grid for each cell.
	def nearest(self, grid):
	    
		ROWS, COLS = len(grid), len(grid[0])
        
        distance = [[float("inf")] * COLS for r in range(ROWS)]
        
        q = collections.deque()
        
        visited = set()

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1:
                    q.append([r, c, 0])
                    visited.add((r, c))

        while q:
            # get the node with distance
            row, col, dist = q.popleft()
            # update the distance matrix
            distance[row][col] = dist
            # traverse the neighbors
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in directions:
                r = row + dr
                c = col + dc
                if (r >= 0 and r < ROWS and 
                    c >= 0 and c < COLS and 
                    (r, c) not in visited and
                    grid[r][c] == 0):
                    
                    q.append([r, c, dist + 1])
                    visited.add((r, c))
                    
        return distance


```

Q-2. Surrounded Regions

```python
class Solution:
    def fill(self, n, m, grid):
        ROWS, COLS = len(grid), len(grid[0])
        visited = set()
        
        def dfs(r, c):
            visited.add((r, c))
            grid[r][c] = "T"
            directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                
                if (nr in range(ROWS) and
                    nc in range(COLS) and
                    (nr, nc) not in visited and
                    grid[nr][nc] == "O"):
                    
                    dfs(nr, nc)

        for r in range(ROWS):
            for c in range(COLS):
                if (r in [0, ROWS - 1] or c in [0, COLS - 1]):
                    if (r, c) not in visited and grid[r][c] == "O":
                        dfs(r, c)
                    
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == "O":
                    grid[r][c] = "X"
        
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == "T":
                    grid[r][c] = "O"
        
        return grid
```

Q-3. Number of enclaves
```python
def solution(grid):
    '''
        * all cells at the boundaries with a 1 are added to the Queue and visited
        * all cells connected to those cells are also added to the queue and visited
        * in the end, the cells having 1s and not visited == Answer! -> count those and return
    '''
    class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visited = set()
        q = collections.deque()

        for r in range(ROWS):
            for c in range(COLS):
                if r in [0, ROWS - 1] or c in [0, COLS - 1]:
                    if grid[r][c] == 1:
                        visited.add((r, c))
                        q.append([r, c])
        
        while q:
            row, col = q.popleft()
            # visit the neighbors and update if required
            neighbors = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            for nr, nc in neighbors:
                r = row + nr
                c = col + nc
                if (r >= 0 and r < ROWS and 
                    c >= 0 and c < COLS and 
                    (r, c) not in visited and 
                    grid[r][c] == 1):

                    q.append([r, c])
                    visited.add((r, c))
        
        cnt = 0
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1 and (r, c) not in visited:
                    cnt += 1
        return cnt

# using DFS
def numberOfEnclaves(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visited = set()
        
        def dfs(r, c):
            visited.add((r, c))
            directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                
                if (nr >= 0 and nr < ROWS and
                    nc >= 0 and nc < COLS and
                    (nr, nc) not in visited and
                    grid[nr][nc] == 1):
                    
                    dfs(nr, nc)

        for r in range(ROWS):
            for c in range(COLS):
                if (r in [0, ROWS - 1] or c in [0, COLS - 1]):
                    if (r, c) not in visited and grid[r][c] == 1:
                        dfs(r, c)
                        
        enclaves = 0
        
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1 and (r, c) not in visited:
                    enclaves += 1
        
        return enclaves
    
```

### Number of distinct islands | 2D DFS | 2D BFS | base coordinate trick!
```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])

        visited = set()

        def dfs(r, c, coordinates, baser, basec):
            visited.add((r, c))
            coordinates.append(tuple([r - baser, c - basec]))

            neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for nr, nc in neighbors:
                row = r + nr
                col = c + nc
                if row in range(ROWS) and col in range(COLS) and (row, col) not in visited and grid[row][col] == 1:
                    dfs(row, col, coordinates, baser, basec)
        
        islandSet = set()

        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in visited and grid[r][c] == 1:
                    islandRep = []
                    dfs(r, c, islandRep, r, c)
                    islandSet.add(tuple(islandRep))
        return len(islandSet)
```

### Graph Coloring: Check if graph is bipartite ( 2 colorable ) - BFS

```python
def solution(n, edges):
    # create the adj list
    # visited set not required in this case!
    # allowed 2 colors assumed: 0 and 1
    color = [-1] * n # we have color array instead! and initially all the nodes have no color ie. -1

    def isBipartiteBFS(start):
        q = deque()
        q.append(start)
        color[start] = 0 # by default we start by coloring the starting node with a 0

        while q:
            node = q.popleft()
            for neighbor in adjList[node]:
                # if the neighbor is not colored, color it with opposite of my color
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
            if isBipartiteBFS(i) == False:
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
                if isBipartiteDFS(neighbor, not newColor) == False:
                    return False
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

    # starting from this node, if below 2 conditions are met, the node is un-safe
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
        res.append(node) # if we are here, we are sure that the node is neither part of the cycle nor it leads to one
        pathVisited.remove(node)
        return False

    for i in range(n):
        if i not in visited:
            dfsCheck(i)

    return res


```

## The something before something pattern -> Topological Sort

### Topological Sort - DFS
```python
def solution(n, edges):
    # create the adjList
    visited = set()
    stack = []

    def topoSortDFS(node):
        visited.add(node)
        for neighbor in adjList[node]:
            if neighbor not in visited:
                topoSortDFS(neighbor)
        stack.append(node)
            
    for i in range(n):
        if i not in visited:
            topoSortDFS(i)
    while stack:
        res.append(stack.pop())
    return res

```

### Topological Sort - BFS (Kahn's Algo)
```python
def solution(n, edges):
    # create the adjList
    indegree = [0] * n

    # calculate the indegrees of all vertices
    for i in range(n):
        for neighbor in adjList[i]:
            indegree[neighbor] += 1
    # add all the vertices with indegree 0 to the queue
    q = collections.deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
    
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    return res

```

### Detect cycle Directed Graph - BFS - Topological sort
```python
def solution(n, edges):
    # create the adjList
    indegree = [0] * n

    # calculate the indegrees of all vertices
    for i in range(n):
        for neighbor in adjList[i]:
            indegree[neighbor] += 1
    # add all the vertices with indegree 0 to the queue
    q = collections.deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
    
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    if len(res) == n:
        return False # there is no cycle
    else:
        return True # there is a cycle

```

### Course Schedule I - BFS - Topological sort - can also be done via DFS cycle detection algo
```python
def solution(n, prerequisites):
    # create the adjList
    indegree = [0] * n

    # calculate the indegrees of all vertices
    for i in range(n):
        for neighbor in adjList[i]:
            indegree[neighbor] += 1
    # add all the vertices with indegree 0 to the queue
    q = collections.deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
    
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    if len(res) == n:
        return False # there is no cycle -> ordering of courses possible
    else:
        return True # there is a cycle -> impossible to do all courses

```

### Course Schedule II (ordering needed) - BFS - Topological sort only way!
```python
def solution(n, prerequisites):
    # create the adjList
    indegree = [0] * n

    # calculate the indegrees of all vertices
    for i in range(n):
        for neighbor in adjList[i]:
            indegree[neighbor] += 1
    # add all the vertices with indegree 0 to the queue
    q = collections.deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
    
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    if len(res) == n:
        return res # there is no cycle -> ordering of courses possible
    else:
        return [] # there is a cycle -> impossible to do all courses

```

### Find eventual safe states - BFS - Topological sort 
```python
def solution(n, edges):
    # create adjList while reversing the edges -> 
    # indegree becomes outdegree and vice-versa
    adjList = { i: [] for i in range(n) }
    for a, b in edges:
        adjList[b].append[a]
    # do a BFS topoSort (kahn's algo)
    ## create an array - indegree and fill it
    indegree = [0] * n
    for i in range(n):
        for neighbor in adjList[i]:
            indegee[i] += 1
    ## do a BFS topo sort
    q = collections.deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
    
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    # res === all the safe nodes!
    return res
```

### Alien Dictionary - Topological Sort (BFS)

```python
def solution(words, N, K):
    '''
        create Directed Graph (adjList) from the given words
        Do a TopoSort
        res = ordering = toposort
    '''
    adjList = {i: [] for i in range(K)}
    for i in range(N - 1):
        word1 = words[i]
        word2 = words[i + 1]
        length = min(len(word1), len(word2))
        f = 0
        for j in range(length):
            if word1[j] != word2[j]:
                adjList[ord(word1[j]) - ord('a')].append(ord(word2[j]) - ord['a'])
                f = 1
                break
        if f == 0: # means we did not find any char diff
            if len(word2) < len(word1):
                return [] # ordering not possible, eg: w1 = abcd, w2 = abc
    
    # start of toposort algo:
    # create and populate indegree array
    indegree = [0] * K
    for i in range(K):
        for neighbor in adjList[i]:
            indegree[neighbor] += 1
        
    # add nodes to the queue(with indegree == 0)
    q = collections.deque()
    for i in range(K):
        if indegree[i] == 0:
            q.append(i)
    
    # do the toposort
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for neighbor in adjList[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    # means there is a cycle -> Alien letter ordering not possible!
    if len(res) != K:
        return []
    else:
        return res

``` 

### Shortest Path in Weighted DAG - Topological Sort (DFS)
#### Note: we can use Topo Sort algo here because there are no cycles, otherwise we need to use advanced algos like Dijkstra's and Bellmanford's algos
```python
def solution(n, m, edges):
    '''
        Create the adjList with pairs = <node, wt>
        Do a topoSort-DFS as it uses stack and we need elements in stack pop order
        Create a distance array with each value = inf
        Update the source distance = 0
        Relaxation: Update the distance array by popping elements from the stack in that order
    '''
    # Creation of adjList
    adjList = { i:[] for i in range(n) }

    for a, b, c in edges:
        adjList[a].append([b, c])

    # topoSort
    stack = []
    visited = set()

    # O(n + m)
    def topoSort(node):
        visited.add(node)
        for neighbor in adjList[node]:
            if neighbor[0] not in visited:
                topoSort(neighbor[0])
        stack.append(node)

    for i in range(n):
        if i not in visited:
            topoSort(i)
    
    # Distance calculations
    distance = [float("inf")] * n
    distance[0] = 0 # assumed source -> can be anything

    # O(n + m)
    while stack:
        u = stack.pop()
        for v, wt in adjList[u]: # relaxation of edges
            if distance[u] + wt < distance[v]:
                distance[v] = distance[u] + wt
    
    return distance



```

### Shortest Path in Undirected Weighted Graph with unit weights - Topological Sort (BFS)

```python

```

### Word Ladder I
```python

```

### Word Ladder II
```python

```