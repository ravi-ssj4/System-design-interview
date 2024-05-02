## Major steps:
* Form the recurrance relation
* write the recursive solution code
* Draw out the recursion tree(optional - for better understanding / analysis)
* Figure out overlapping subproblems
* Memoization: Convert recursive solution to memoized solution
* Tabulation: Convert Memoized solution to Tabulated solution
* Optimization for space: Try to optimize the space for Tabulation solution if possible

## Method to form the recurrance relation
* Represent the problem in terms of indices(always possible)
* Do all possible stuffs to these indices
* Figure out based on the problem:
    * Sum of all possible stuffs
    * min. of all possible stuffs
    * max. of all possible stuffs, etc.

### General Pattern for 1D - DP Problems

Question | DP - 3 | Frog Jump

```python
from os import *
from sys import *
from collections import *
from math import *

from typing import *

# Raw Recursion based on Recurrance relation
def frogJump(n: int, heights: List[int]) -> int:

    def f(i):
        if i == 0:
            return 0
        
        left = f(i - 1) + abs(heights[i] - heights[i - 1])
        right = float("inf")
        if i > 1:
            right = f(i - 2) + abs(heights[i] - heights[i - 2])
        
        return min(left, right)
    
    return f(n - 1)

# Memoization
def frogJump(n: int, heights: List[int]) -> int:

    def f(i):
        if i == 0:
            return 0
        if dp[i] != -1:
            return dp[i]

        left = f(i - 1) + abs(heights[i] - heights[i - 1])
        right = float("inf")
        if i > 1:
            right = f(i - 2) + abs(heights[i] - heights[i - 2])
        
        dp[i] = min(left, right)
        return dp[i] 
    
    dp = [-1] * n
    f(n - 1)
    return dp[n - 1]

# Tabulation
def frogJump(n: int, heights: List[int]) -> int:
    
    dp = [-1] * n
    
    dp[0] = 0

    for i in range(1, n):
        left = dp[i - 1] + abs(heights[i] - heights[i - 1])
        right = float("inf")
        if i > 1:
            right = dp[i - 2] + abs(heights[i] - heights[i - 2])
        
        dp[i] = min(left, right)
    
    return dp[n - 1]

# Tabulation space optimization
def frogJump(n: int, heights: List[int]) -> int:
    
    prev1, prev2 = 0, 0

    for i in range(1, n):
        left = prev1 + abs(heights[i] - heights[i - 1])
        right = float("inf")
        if i > 1:
            right = prev2 + abs(heights[i] - heights[i - 2])
        
        curi = min(left, right)
    
        prev2 = prev1
        prev1 = curi

    return prev1
```

### Question extension: Frog Jump | K jumps allowed

```python

```

### Maximum sum of non-adjacent elements | House Robber 1
```python
# Simple recursion
def maximumNonAdjacentSum(nums):    
    
    def dfs(i):
        if i == 0:
            return nums[i]
        if i < 0:
            return 0

        # pick and not pick
        pick = nums[i] + dfs(i - 2)
        notPick = 0 + dfs(i - 1)
    
        return max(pick, notPick)

    return dfs(len(nums) - 1)

# Memoization
def maximumNonAdjacentSum(nums):    
    
    def dfs(i):
        if i == 0:
            return nums[i]
        if i < 0:
            return 0
        
        if dp[i] != -1:
            return dp[i]

        # pick and not pick
        pick = nums[i] + dfs(i - 2)
        notPick = 0 + dfs(i - 1)

        dp[i] = max(pick, notPick)
        
        return dp[i]

    n = len(nums)
    dp = [-1] * n
    return dfs(n - 1)

# Tabulation
def maximumNonAdjacentSum(nums):    

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    for i in range(1, n):
        pick = nums[i] + dp[i - 2]
        notPick = 0 + dp[i - 1]

        dp[i] = max(dp[i], max(pick, notPick))


    return dp[n - 1]

# Tabulation: space optimized
def maximumNonAdjacentSum(nums):    

    n = len(nums)
    # dp = [0] * n
    prev1 = nums[0]
    prev2 = 0

    curi = 0
    for i in range(1, n):
        pick = nums[i]
        if i > 1:
            pick += prev2
        notPick = 0 + prev1

        curi = max(curi, max(pick, notPick))
        prev2 = prev1
        prev1 = curi


    return prev1

```

### House Robber 2
```python
def houseRobber(valueInHouse):
    
    def houseRobber1(nums):
        n = len(nums)
        prev1 = nums[0]
        prev2 = 0
        curi = 0
        for i in range(1, n):
            pick = nums[i]
            if i > 1:
                pick += prev2
            notPick = 0 + prev1
            curi = max(curi, max(pick, notPick))
            prev2 = prev1
            prev1 = curi
        return prev1

    n = len(valueInHouse)
    if n == 0:
        return 0
    if n == 1:
        return valueInHouse[0]

    return max(houseRobber1(valueInHouse[1:]), houseRobber1(valueInHouse[:-1]))
```

### Ninja's Training

```python
from typing import *

# simple recursion
def ninjaTraining(n: int, points: List[List[int]]) -> int:

    def dfs(day, nextDayTask):
        # base case
        if day == 0:
            maxi = 0
            for task in range(3):
                if task != nextDayTask:
                    maxi = max(maxi, points[0][task])
            return maxi
        
        # general case
        maxi = 0
        for task in range(3):
            if task != nextDayTask:
                maxi = max(maxi, points[day][task] + dfs(day - 1, task))
        
        return maxi

    return dfs(n - 1, 3)

# memoization
def ninjaTraining(n: int, points: List[List[int]]) -> int:

    def dfs(day, nextDayTask):
        # base case
        if day == 0:
            maxi = 0
            for task in range(3):
                if task != nextDayTask:
                    maxi = max(maxi, points[0][task])
            return maxi
        
        # memoization:
        if dp[day][nextDayTask] != -1:
            return dp[day][nextDayTask]

        # general case
        maxi = 0
        for task in range(3):
            if task != nextDayTask:
                maxi = max(maxi, points[day][task] + dfs(day - 1, task))
        
        dp[day][nextDayTask] = maxi

        return maxi

    dp = [[-1] * 4 for _ in range(n)]
    
    dfs(n - 1, 3)

    return dp[n - 1][3]



# tabulation
def ninjaTraining(n: int, points: List[List[int]]) -> int:
    
    # step 1: initialize dp array
    dp = [[0] * 4 for _ in range(n)]

    # step 2: fill the base case cells
    dp[0][0] = max(points[0][1], points[0][2])
    dp[0][1] = max(points[0][0], points[0][2])
    dp[0][2] = max(points[0][0], points[0][1])
    dp[0][3] = max(points[0][0], points[0][1], points[0][2])

    # step 3: fill the rest of the 2D array bottom up
    for day in range(1, n): # 1 -> n - 1
        for nextDayTask in range(4): # 0 -> 3
            dp[day][nextDayTask] = 0

            for task in range(3): # 0 -> 2
                if task != nextDayTask: # the task 'task' can be thus performed!
                    pointsEarnedCurrentDay = points[day][task] + dp[day - 1][task] # for prev day, current day's task ie. 'task' will be its nextDayTask
                    dp[day][nextDayTask] = max(dp[day][nextDayTask], pointsEarnedCurrentDay)
                    
    return dp[n - 1][3]


# tabulation -> space optimized
def ninjaTraining(n: int, points: List[List[int]]) -> int:
    
    # step 1: initialize dp array
    dp = [0] * 4

    # step 2: fill the base case cells -> dp represents the prev day or 'day - 1'
    dp[0] = max(points[0][1], points[0][2])
    dp[1] = max(points[0][0], points[0][2])
    dp[2] = max(points[0][0], points[0][1])
    dp[3] = max(points[0][0], points[0][1], points[0][2])

    # step 3: fill the rest of the 2D array bottom up
    for day in range(1, n): # 1 -> n - 1
        tempDP = [0] * 4 # tempDP represents the current day
        for nextDayTask in range(4): # 0 -> 3
            tempDP[nextDayTask] = 0

            for task in range(3): # 0 -> 2
                if task != nextDayTask: # the task 'task' can be thus performed!
                    pointsEarnedCurrentDay = points[day][task] + dp[task] # for prev day, current day's task ie. 'task' will be its nextDayTask
                    tempDP[nextDayTask] = max(tempDP[nextDayTask], pointsEarnedCurrentDay)
        dp = tempDP.copy()

    return dp[3]

```

# DP on GRIDS

## Template for Grid based problems

### DP8 - Grid unique paths

```python

def uniquePaths(m, n):
	def f(i, j):
		if i < 0 or j < 0:
			return 0
		if i == 0 and j == 0:
			return 1
		
		up = f(i - 1, j)
		left = f(i, j - 1)

		return up + left

	return f(m - 1, n - 1)

def uniquePaths(m, n):
	def f(i, j):
		if i < 0 or j < 0:
			return 0
		if i == 0 and j == 0:
			return 1
		if dp[i][j] != -1:
			return dp[i][j]

		up = f(i - 1, j)
		left = f(i, j - 1)

		dp[i][j] = up + left
		
		return dp[i][j]

	dp = [[-1] * n for i in range(m)]

	return f(m - 1, n - 1)

def uniquePaths(m, n):

	dp = [[0] * n for _ in range(m)]

	for i in range(m):
		for j in range(n):
			
			if i == 0 and j == 0:
				dp[i][j] = 1
			else:
				up, left = 0, 0
				
				if i > 0:
					up = dp[i - 1][j]
				if j > 0:
					left = dp[i][j - 1]
				
				dp[i][j] = up + left

	return dp[m - 1][n - 1]

def uniquePaths(m, n):

	prev = [0] * n

	for i in range(m):
		cur = [0] * n
		for j in range(n):
			
			if i == 0 and j == 0:
				cur[j] = 1
			else:
				up, left = 0, 0
				
				if i > 0:
					up = prev[j]
				if j > 0:
					left = cur[j - 1]
				
				cur[j] = up + left
		prev = cur.copy()

	return prev[n - 1]



```

### Unique Paths II

```python

def mazeObstacles(n, m, mat):
    
    modulo = 10**9 + 7
    
    dp = [[-1] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if mat[i][j] == -1: # just this one base case is added to Unique Paths I problem
                dp[i][j] = 0
            elif i == 0 and j == 0:
                dp[i][j] = 1
            else:
                up, left = 0, 0
                if i > 0:
                    up = dp[i - 1][j]
                if j > 0:
                    left = dp[i][j - 1]
                dp[i][j] = (up + left) % modulo

    return dp[n - 1][m - 1]

```

###  DP 10: Minimum path sum in grid

```python

# recurrance relation
def minSumPath(grid):

    n, m = len(grid), len(grid[0])

    def f(i, j): # min path sum when travelling from idx (0, 0) to (i, j)

        # base cases
        if i == 0 and j == 0:
            return grid[i][j]
        if i < 0 or j < 0: # can only go out of bounds on the left side or upwards as those 2 directions are only allowed movement to
            return float("inf")

        # general case
        up = grid[i][j] + f(i - 1, j)
        left = grid[i][j] + f(i, j - 1)

        # cummulation
        return min(up, left)
    
    return f(n - 1, m - 1)

# memoization
def minSumPath(grid):

    n, m = len(grid), len(grid[0])

    def f(i, j): # min path sum when travelling from idx (0, 0) to (i, j)

        # base cases
        if i == 0 and j == 0:
            return grid[i][j]
        if i < 0 or j < 0: # can only go out of bounds on the left side or upwards as those 2 directions are only allowed movement to
            return float("inf")

        if dp[i][j] != -1: 
            return dp[i][j]

        # general case
        up = grid[i][j] + f(i - 1, j)
        left = grid[i][j] + f(i, j - 1)

        # cummulation
        dp[i][j] = min(up, left)
        return dp[i][j]
    
    dp = [[-1] * m for _ in range(n)]

    f(n - 1, m - 1)

    return dp[n - 1][m - 1]

# Tabulation
def minSumPath(grid):

    n, m = len(grid), len(grid[0])
    
    dp = [[0] * m for _ in range(n)]

    # base case
    # dp[0][0] = grid[i][j]

    # loop for states
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                dp[i][j] = grid[i][j]
            else:
                up = grid[i][j]
                if i > 0:
                    up += dp[i - 1][j]
                else:
                    up += 1e9

                left = grid[i][j]
                if j > 0:
                    left += dp[i][j - 1]
                else:
                    left += 1e9
                    
                dp[i][j] = min(up, left)

    return dp[n - 1][m - 1]


# Space Optimization
def minSumPath(grid):

    n, m = len(grid), len(grid[0])
    
    prev = [0] * m

    # base case
    # dp[0][0] = grid[i][j]

    # loop for states
    for i in range(n):
        cur = [0] * m
        for j in range(m):
            if i == 0 and j == 0:
                cur[j] = grid[i][j]
            else:
                up = grid[i][j]
                if i > 0:
                    up += prev[j]
                else:
                    up += 1e9

                left = grid[i][j]
                if j > 0:
                    left += cur[j - 1]
                else:
                    left += 1e9
                    
                cur[j] = min(up, left)

        prev = cur

    return prev[m - 1]

```

### DP 11: Triangle: Fixed starting and variable ending points

```python
# recurrance relation
def minimumPathSum(triangle, n):

    def f(i, j): # min path sum to go from index (i, j) to the last row(i == n - 1 and j varies from 0 to n - 1)
        # base case
        if i == n - 1:
            return triangle[i][j] # just return the cell you are at, all are valid answers
        
        # no issue of going out of bounds as it can move only down or rightDiagonal

        # general case: do all stuffs with i, j
        down = triangle[i][j] + f(i + 1, j)
        rightDiag = triangle[i][j] + f(i + 1, j + 1)

        # accumulate
        return min(down, rightDiag)
    
    return f(0, 0)

# memoization
def minimumPathSum(triangle, n):

    def f(i, j): # min path sum to go from index (i, j) to the last row(i == n - 1 and j varies from 0 to n - 1)
        # base case
        if i == n - 1:
            return triangle[n - 1][j] # just return the cell you are at, all are valid answers

        if dp[i][j] != -1:
            return dp[i][j]

        # no issue of going out of bounds as it can move only down or rightDiagonal

        # general case: do all stuffs with i, j
        down = triangle[i][j] + f(i + 1, j)
        rightDiag = triangle[i][j] + f(i + 1, j + 1)

        # accumulate
        dp[i][j] = min(down, rightDiag)
        return dp[i][j]
    
    dp = [[-1] * n for _ in range(n)]
    
    return f(0, 0)


# Tabulation
def minimumPathSum(triangle, n):
    
    dp = [[0] * n for _ in range(n)]

    # base case: i == n - 1
    for j in range(n):
        dp[n - 1][j] = triangle[n - 1][j]

    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            down = triangle[i][j] + dp[i + 1][j]
            rightDiag = triangle[i][j] + dp[i + 1][j + 1]
        
            dp[i][j] = min(down, rightDiag)
    
    return dp[0][0]
```

### DP 12: Maximum Path sum in the matrix | variable startng and variable ending points

```python
def getMaxPathSum(matrix):

    N, M = len(matrix), len(matrix[0])

    def f(i, j): # maximum path sum from index (0, j); j e [0, M - 1] to index (i, j)
        # base case
        if j < 0 or j >= M:
            return -1e8

        if i == 0:
            return matrix[i][j]
        
        if dp[i][j] != -1:
            return dp[i][j]

        # general case
        up = matrix[i][j] + f(i - 1, j)
        leftDiag = matrix[i][j] + f(i - 1, j - 1)
        rightDiag = matrix[i][j] + f(i - 1, j + 1)

        # accumulation
        dp[i][j] = max(up, leftDiag, rightDiag)
        return dp[i][j]

    dp = [[-1] * M for _ in range(N)]

    maxi = -1e8
    for j in range(M):
        maxi = max(maxi, f(N - 1, j))
    
    return maxi
```

### DP 13: Cherry Pickup | 3D - DP

```python

# Recurrance relation
def maximumChocolates(r: int, c: int, grid: List[List[int]]) -> int:

    def f(i, j1, j2): # max sum of chocolates collected by both Alice and Bob together from index i, j1 and index i, j2 till the last row
        # base cases
        if j1 < 0 or j2 < 0 or j1 >= c or j2 >= c:
            return -1e8
        if i == r - 1:
            if j1 == j2:
                return grid[i][j1]
            else:
                return grid[i][j1] + grid[i][j2]


        maxi = -1e8
        # general cases
        for dj1 in range(-1, 2): # -1, 0, 1
            for dj2 in range(-1, 2):
                if j1 == j2:
                    cost = grid[i][j1] + f(i + 1, j1 + dj1, j2 + dj2)
                else:
                    cost = grid[i][j1] + grid[i][j2] + f(i + 1,j1 + dj1, j2 + dj2)
                maxi = max(maxi, cost)
        
        return maxi
    
    return f(0, 0, c - 1)

# Memoization
def maximumChocolates(r: int, c: int, grid: List[List[int]]) -> int:

    def f(i, j1, j2): # max sum of chocolates collected by both Alice and Bob together from index i, j1 and index i, j2 till the last row
        # base cases
        if j1 < 0 or j2 < 0 or j1 >= c or j2 >= c:
            return -1e8
        if i == r - 1:
            if j1 == j2:
                return grid[i][j1]
            else:
                return grid[i][j1] + grid[i][j2]
        
        if dp[i][j1][j2] != -1:
            return dp[i][j1][j2]


        maxi = -1e8
        # general cases
        for dj1 in range(-1, 2): # -1, 0, 1
            for dj2 in range(-1, 2):
                if j1 == j2:
                    cost = grid[i][j1] + f(i + 1, j1 + dj1, j2 + dj2)
                else:
                    cost = grid[i][j1] + grid[i][j2] + f(i + 1,j1 + dj1, j2 + dj2)
                maxi = max(maxi, cost)
        
        dp[i][j1][j2] = maxi
        return dp[i][j1][j2]
    
    dp = [[[-1] * c for _ in range(c)] for _ in range(r)]
    
    return f(0, 0, c - 1)

# Tabulation
def maximumChocolates(r: int, c: int, grid: List[List[int]]) -> int:
    
    dp = [[[0] * c for _ in range(c)] for _ in range(r)]
    
    # base case
    for j1 in range(c):
        for j2 in range(c):
            if j1 == j2:
                dp[r - 1][j1][j2] = grid[r - 1][j1]
            else:
                dp[r - 1][j1][j2] = grid[r - 1][j1] + grid[r - 1][j2]
        
    # general case
    for i in range(r - 2, -1, -1):
        for j1 in range(c):
            for j2 in range(c):
                maxi = -1e8
                for dj1 in range(-1, 2):
                    for dj2 in range(-1, 2):
                        cost = 0
                        if j1 == j2:
                            cost = grid[i][j1]
                        else:
                            cost = grid[i][j1] + grid[i][j2]
                        if j1 + dj1 >= 0 and j1 + dj1 < c and j2 + dj2 >= 0 and j2 + dj2 < c:
                            cost += dp[i + 1][j1 + dj1][j2 + dj2]
                        else:
                            cost += -1e8
                        maxi = max(maxi, cost)
                        dp[i][j1][j2] = maxi
    
    return dp[0][0][c - 1]
```

# General structure of DP on subsequences / subsets + target

### Q. Subset sum equal to k
```python

# Recurrance relation
def subsetSumToK(n, k, arr):

    def f(i, target):
        # base case
        if target == 0:
            return True
        if i == 0:
            return arr[i] == target

        # general case
        notTake = f(i - 1, target)
        take = False
        if arr[i] <= target:
            take = f(i - 1, target - arr[i])

        # accumulate
        return notTake or take
    
    return f(n - 1, k)

def subsetSumToK(n, k, arr):

    def f(i, target):
        # base case
        if target == 0:
            return True
        if i == 0:
            return arr[i] == target
        if dp[i][target] != -1:
            return dp[i][target]

        # general case
        notTake = f(i - 1, target)
        take = False
        if arr[i] <= target:
            take = f(i - 1, target - arr[i])

        # accumulate
        dp[i][target] = notTake | take
        return dp[i][target]
        

    dp = [[-1] * (k + 1) for _ in range(n)]

    return f(n - 1, k)

# tabulation
def subsetSumToK(n, k, arr):

    dp = [[False] * (k + 1) for _ in range(n)]
    
    # base case
    for i in range(n):
        dp[i][0] = True
    
    if arr[0] <= k:
        dp[0][arr[0]] = True
    
    # general case: simulate states + recursion copy
    for i in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[i - 1][target]
            take = False
            if arr[i] <= target:
                take = dp[i - 1][target - arr[i]]
            dp[i][target] = take or notTake
    # return ans
    return dp[n - 1][k]
    
    
# tabulation: space optimization
def subsetSumToK(n, k, arr):

    prev = [[False] for _ in range(k + 1)]
    cur = [[False] for _ in range(k + 1)]
    
    # base case
    prev[0] = True
    cur[0] = True

    if arr[0] <= k:
        prev[arr[0]] = True
    
    # general case: simulate states + recursion copy
    for i in range(1, n):
        for target in range(1, k + 1):
            notTake = prev[target]
            take = False
            if arr[i] <= target:
                take = prev[target - arr[i]]
            cur[target] = notTake or take
        prev = cur.copy()
    # return ans
    return prev[k]
```

### Q. Partition equal subset sum
```python
def subsetSumEqualsK(arr, n, k):

    dp = [[False] * (k + 1) for _ in range(n)]
    
    # base case
    for i in range(n):
        dp[i][0] = True
    
    if arr[0] <= k:
        dp[0][arr[0]] = True
    
    # general case: simulate states + recursion copy
    for i in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[i - 1][target]
            take = False
            if arr[i] <= target:
                take = dp[i - 1][target - arr[i]]
            dp[i][target] = take or notTake
    # return ans
    return dp[n - 1][k]
    

def canPartition(arr, n):
    
    totalSum = sum(arr)

    if totalSum % 2:
        return False

    return subsetSumEqualsK(arr, n, totalSum // 2)
```

### Q. Partition a set into 2 subsets with minimum absolute difference
```python
def subsetSumEqualsKModified(arr, n, k, dp):
    # base case
    # target == 0
    for i in range(n):
        dp[i][0] = True
    
    # i == 0
    if arr[0] <= k:
        dp[0][arr[0]] = True

    # general case
    for i in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[i - 1][target]
            take = False
            if arr[i] <= target:
                take = dp[i - 1][target - arr[i]]
            dp[i][target] = take or notTake

    # # return the dp array
    # return dp

def minSubsetSumDifference(arr: List[str], n: int) -> int:
    target = sum(arr)
    dp = [[False] * (target + 1) for _ in range(n)]

    subsetSumEqualsKModified(arr, n, target, dp)

    s1, s2 = 0, 0
    res = 1e9
    for t in range(target + 1):
        if dp[n - 1][t] == True:
            s1 = t
            s2 = target - s1
            res = min(res, abs(s1 - s2))
    
    return res
```

## Counting pattern for subsequences / subsets

### Q. Counts subsets with sum k
```python
def findWays(arr: List[int], k: int) -> int:

    n = len(arr)
    
    def f(i, s):
        # base case
        if s == 0:
            return 1
        if i == 0:
            if s == arr[i]:
                return 1
            else:
                return 0
        
        # general case
        notPick = f(i - 1, s)
        pick = 0
        if arr[i] <= s:
            pick = f(i - 1, s - arr[i])
        
        return pick + notPick

    
    return f(n - 1, k)

# Memoization
def findWays(arr: List[int], k: int) -> int:

    n = len(arr)
    
    def f(i, s):

        # base case
        if s == 0:
            return 1
        if i == 0:
            if s == arr[i]:
                return 1
            else:
                return 0

        if dp[i][s] != -1:
            return dp[i][s]

        # general case
        notPick = f(i - 1, s)
        pick = 0
        if arr[i] <= s:
            pick = f(i - 1, s - arr[i])
        
        dp[i][s] = pick + notPick

        return dp[i][s]

    dp = [[-1] * (k + 1) for _ in range(n)]

    return f(n - 1, k)


# Tabulation
def findWays(arr: List[int], k: int) -> int:

    n = len(arr)

    dp = [[0] * (k + 1) for _ in range(n)]

    # base case
    # if s == 0
    for i in range(n):
        dp[i][0] = 1
    # i == 0
    if arr[i] <= k:
        dp[0][arr[0]] = 1

    # general case
    for i in range(1, n):
        for s in range(1, k + 1):
            notTake = dp[i - 1][s]
            take = 0
            if arr[i] <= s:
                take = dp[i - 1][s - arr[i]]
            
            dp[i][s] = notTake + take

    # return
    return dp[n - 1][k]


# Tabulation: space optimization
def findWays(arr: List[int], k: int) -> int:

    n = len(arr)

    prev = [0 for _ in range(k + 1)]
    cur = [0 for _ in range(k + 1)]

    # base case
    # if s == 0
    for i in range(n):
        prev[0] = 1
    # i == 0
    if arr[i] <= k:
        prev[arr[0]] = 1

    # general case
    for i in range(1, n):
        for s in range(1, k + 1):
            notTake = prev[s]
            take = 0
            if arr[i] <= s:
                take = prev[s - arr[i]]
            
            cur[s] = notTake + take
        prev = cur.copy()

    # return
    return prev[k]
```

### Q. Counts partitions with given difference
```python
from os import *
from sys import *
from collections import *
from math import *

from typing import List


mod = int(1e9 + 7)

# Tabulation
def findWays(arr: List[int], k: int) -> int:

    n = len(arr)
    
    dp = [[0] * (k + 1) for _ in range(n)]

    # base case
    
    # for i == 0 and for sum == 0
    if arr[0] == 0:
        dp[0][0] = 2
    else:
        dp[0][0] = 1
    
    # for i == 0 and for sum != 0, ie. k != 0
    if arr[0] != 0 and arr[0] <= k:
        dp[0][arr[0]] = 1

    # general case
    for i in range(1, n):
        for s in range(k + 1):
            notTake = dp[i - 1][s]
            take = 0
            if arr[i] <= s:
                take = dp[i - 1][s - arr[i]]
            
            dp[i][s] = (notTake + take) % mod

    # return
    return dp[n - 1][k]

def countPartitions(n: int, d: int, arr: List[int]) -> int:

    newTarget = sum(arr) - d

    if newTarget < 0 or newTarget % 2:
        return 0
    
    return findWays(arr, newTarget // 2)

```

### Q. 0 / 1 Knapsack
```python
from os import *
from sys import *
from collections import *
from math import *

## Read input as specified in the question.
## Print output as specified in the question.

# Memoization
def maxProfit01Knapsack(N, wt, val, W):
    # print(N, W)
    # print(wt)
    # print(val)
    def f(ind, w):
        # base case
        if ind == 0:
            if wt[0] <= w:
                return val[0]
            return 0

        if dp[ind][w] != -1:
            return dp[ind][w]

        # general case
        notTake = f(ind - 1, w)
        take = float("-inf")
        if wt[ind] <= w:
            take = val[ind] + f(ind - 1, w - wt[ind])

        # maximize
        dp[ind][w] = max(take, notTake)
        return dp[ind][w]
    
    dp = [[-1] * (W + 1) for _ in range(N)]

    return f(N - 1, W)

# Tabulation
def maxProfit01Knapsack(N, wt, val, W):
    
    dp = [[0] * (W + 1) for _ in range(N)]

    # base case
    # i == 0
    for i in range(wt[0], W + 1): # until bag wt. >= wt. of 0th element
        dp[0][i] = val[0]

    # general case + maximize
    for i in range(1, N):
        for w in range(W + 1):
            notTake = dp[i - 1][w]
            take = float("-inf")
            if wt[i] <= w:
                take = val[i] + dp[i - 1][w - wt[i]]
            
            dp[i][w] = max(take, notTake)
    
    return dp[N - 1][W]

# Tabulation: space optimization 2 rows
def maxProfit01Knapsack(N, wt, val, W):
    
    prev = [0 for _ in range(W + 1)]
    cur = [0 for _ in range(W + 1)]

    # base case
    # i == 0
    for i in range(wt[0], W + 1): # until bag wt. >= wt. of 0th element
        prev[i] = val[0]

    # general case + maximize
    for i in range(1, N):
        for w in range(W + 1):
            notTake = prev[w]
            take = float("-inf")
            if wt[i] <= w:
                take = val[i] + prev[w - wt[i]]
            
            cur[w] = max(take, notTake)
        
        prev = cur.copy()
    
    return prev[W]

# Tabulation: space optimization 1 row
def maxProfit01Knapsack(N, wt, val, W):

    prev = [0 for _ in range(W + 1)]

    # base case
    # i == 0
    for i in range(wt[0], W + 1): # until bag wt. >= wt. of 0th element
        prev[i] = val[0]

    # general case + maximize
    for i in range(1, N):
        for w in range(W, -1, -1):
            notTake = prev[w]
            take = float("-inf")
            if wt[i] <= w:
                take = val[i] + prev[w - wt[i]]
            
            prev[w] = max(take, notTake)
    
    return prev[W]

T = int(input())
for t in range(T):
    N = int(input())
    wt = [0] * N
    wt = input().split()
    wt = [int(w) for w in wt]
    
    val = [0] * N
    val = input().split()
    val = [int(v) for v in val]
    
    W = int(input())

    print(maxProfit01Knapsack(N, wt, val, W))
```