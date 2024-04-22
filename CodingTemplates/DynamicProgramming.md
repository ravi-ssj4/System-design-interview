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