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