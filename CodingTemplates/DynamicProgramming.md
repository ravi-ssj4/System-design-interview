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

## General Pattern for 1D - DP Problems

### Question | DP - 3 | Frog Jump

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
https://www.geeksforgeeks.org/problems/minimal-cost/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=minimal-cost
```python
class Solution:
    # recursion
    def minimizeCost(self, height, n, k):
        def f(i):
            # base case
            if i == 0:
                return 0
            
            mini = float("inf")
            for j in range(1, k + 1):
                if i - j >= 0:
                    energy = f(i - j) + abs(height[i] - height[i - j])
                    mini = min(mini, energy)
            return mini
        
        return f(n - 1)
    
    # Memoization
    def minimizeCost(self, height, n, k):
        def f(i):
            # base case
            if i == 0:
                return 0
            if dp[i] != -1:
                return dp[i]
            
            # gen case
            mini = float("inf")
            for j in range(1, k + 1):
                if i - j >= 0:
                    energy = f(i - j) + abs(height[i] - height[i - j])
                    mini = min(mini, energy)
            dp[i] = mini
            return dp[i]
        dp = [-1] * n
        return f(n - 1)
    
    # Tabulation
    def minimizeCost(self, height, n, k):
        dp = [0] * n
        
        # base case
        dp[0] = 0
        
        # gen case
        for i in range(1, n):
            mini = float("inf")
            for j in range(1, k + 1):
                if i - j >= 0:
                    energy = dp[i - j] + abs(height[i] - height[i - j])
                    mini = min(mini, energy)
            dp[i] = mini
            
        return dp[n - 1]


```
### Maximum sum of non-adjacent elements | House Robber 1

```python
'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 400
'''

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
    #    0  1  2  3  4  5
    #            p2 p2  c
    #    p2 p1 c
    # p2 p1 c
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
'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 3:

Input: nums = [1,2,3]
Output: 3
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 1000
'''

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
### Leetcode(M): Jump Game I
```python
class Solution:
    # def canJump(self, nums: List[int]) -> bool:
    #     n = len(nums)

    #     def f(i): # tells if we can come from index i to index n - 1 or not
    #         # base case
    #         if dp[i] != -1:
    #             return dp[i]

    #         if i >= n - 1:
    #             return True

    #         # gen case
    #         canJump = False

    #         for jumpLength in range(1, nums[i] + 1):
    #             nextIdx = i + jumpLength
    #             canJump = canJump or f(nextIdx)
                
    #         dp[i] = canJump
    #         return dp[i]
        
    #     dp = [-1] * n
    #     return f(0)

    # def canJump(self, nums: List[int]) -> bool:
    #     n = len(nums)
    #     dp = [False] * n
    #     # base case
    #     dp[n - 1] = True
    #     # gen case
    #     for i in range(n - 2, -1, -1):
    #         # gen case
    #         canJump = False
    #         for jumpLength in range(1, nums[i] + 1):
    #             nextIdx = i + jumpLength
    #             canJump = canJump or dp[nextIdx]
    #         dp[i] = canJump
    #     return dp[0]

    # Greedy (intuition)
    # in Tabulation, observation: for any index i, we are checking if 
    # it can reach any good idx by making it jump by 1, 2,.. nums[i] steps -> this is not needed
    # in the max distance jump that we can make from the index i, if its able to reach the leftmost 
    # good index, its good enough -> ie. it can reach to it -> after that it itself will become the 
    # leftmost good index for the previous indices -> it itself will become the goal for the previous indices
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        leftMostGoodIdx = n - 1
        
        for i in range(n - 2, -1, -1):
            maxJumpDistance = i + nums[i]
            if maxJumpDistance >= leftMostGoodIdx: # means the current idx can reach the final goal(leftMostGoodIdx)
                leftMostGoodIdx = i
            
        return leftMostGoodIdx == 0
```
### Ninja's Training

```python
'''
Geek is going for n day training program. He can perform any one of these three activities Running, Fighting, and Learning Practice. Each activity has some point on each day. As Geek wants to improve all his skills, he can't do the same activity on two consecutive days. Help Geek to maximize his merit points as you are given a 2D array of points points, corresponding to each day and activity.

Example:
Input:
n = 3
points = [[1,2,5],[3,1,1],[3,3,3]]
Output:
11
Explanation:
Geek will learn a new move and earn 5 point then on second
day he will do running and earn 3 point and on third day
he will do fighting and earn 3 points so, maximum point is 11.

Example:
Input:
n = 3
points = [[1,2,5],[3,1,1],[3,2,3]]
Output:
11
Explanation:
Geek will learn a new move and earn 5 point then on second
day he will do running and earn 3 point and on third day
he will do running and earn 3 points so, maximum point is 11.
Your Task:
You don't have to read input or print anything. Your task is to complete the function maximumPoints() which takes the integer n and a 2D array points and returns the maximum points he can earn.

Expected Time Complexity: O(3*n)
Expected Space Complexity: O(3*n)

Constraint:
1 <=  n <= 105
1 <=  point[i] <= 100
'''

#User function Template for python3

class Solution:
    # Recursion
    def maximumPoints(self, points, n):
        
        def f(day, nextDayTask):
            # base case
            if day == 0:
                maxi = 0
                for task in range(3):
                    if task != nextDayTask:
                        maxi = max(maxi, points[day][task])
                return maxi
            
            # gen case
            maxi = 0
            for task in range(3):
                if task != nextDayTask:
                    score = points[day][task] + f(day - 1, task)
                    maxi = max(maxi, score)
            return maxi
            
        return f(n - 1, 3)
    
    # Memoization
    def maximumPoints(self, points, n):
        
        def f(day, nextDayTask):
            # base case
            if day == 0:
                maxi = 0
                for task in range(3):
                    if task != nextDayTask:
                        maxi = max(maxi, points[day][task])
                return maxi
                
            if dp[day][nextDayTask] != -1:
                return dp[day][nextDayTask]
            
            # gen case
            maxi = 0
            for task in range(3):
                if task != nextDayTask:
                    score = points[day][task] + f(day - 1, task)
                    maxi = max(maxi, score)
            dp[day][nextDayTask] = maxi
            return dp[day][nextDayTask]
            
        dp = [[-1] * 4 for _ in range(n)]
        return f(n - 1, 3)
    
    # Tabulation
    def maximumPoints(self, points, n):
            
        dp = [[0] * 4 for _ in range(n)]
        # day == 0:
        # means next day task 0 was performed, hence we cannot perform
        # task 1 or 2(remember total we have 0, 1, 2 ie. 3 tasks)
        dp[0][0] = max(points[0][1], points[0][2])
        dp[0][1] = max(points[0][0], points[0][2])
        dp[0][2] = max(points[0][0], points[0][1])
        dp[0][3] = max(points[0][0], points[0][1], points[0][2])
        
        for day in range(1, n):
            for nextDayTask in range(4):
                maxi = 0
                for task in range(3):
                    if task != nextDayTask:
                        score = points[day][task] + dp[day - 1][task]
                        maxi = max(maxi, score)
                    dp[day][nextDayTask] = maxi

        return dp[n - 1][3]
        
    # Tabulation: space optimization
    def maximumPoints(self, points, n):
        prev = [0] * 4
        cur = [0] * 4
        # day == 0:
        # means next day task 0 was performed, hence we cannot perform
        # task 1 or 2(remember total we have 0, 1, 2 ie. 3 tasks)
        prev[0] = max(points[0][1], points[0][2])
        prev[1] = max(points[0][0], points[0][2])
        prev[2] = max(points[0][0], points[0][1])
        prev[3] = max(points[0][0], points[0][1], points[0][2])
        
        for day in range(1, n):
            for nextDayTask in range(4):
                maxi = 0
                for task in range(3):
                    if task != nextDayTask:
                        score = points[day][task] + prev[task]
                        maxi = max(maxi, score)
                    cur[nextDayTask] = maxi
            prev = cur.copy()

        return prev[3]

```

# DP on GRIDS

## Template for Grid based problems
### DP 08: Grid unique paths

```python
'''
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:


Input: m = 3, n = 7
Output: 28
Example 2:

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

Constraints:

1 <= m, n <= 100
'''

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
### DP 09: Unique Paths II

```python
'''
You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:


Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
Example 2:


Input: obstacleGrid = [[0,1],[0,0]]
Output: 1
 

Constraints:

m == obstacleGrid.length
n == obstacleGrid[i].length
1 <= m, n <= 100
obstacleGrid[i][j] is 0 or 1.
'''
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
### DP 10: Minimum path sum in grid: Fixed start and fixed ending

```python
'''
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

 

Example 1:


Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
Example 2:

Input: grid = [[1,2,3],[4,5,6]]
Output: 12
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 200
0 <= grid[i][j] <= 200
'''

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
            if i == 0 and j == 0: # covering the base case like this
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
    cur = [0] * m

    # base case
    # dp[0][0] = grid[i][j]

    # loop for states
    for i in range(n):
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

        prev = cur.copy()

    return prev[m - 1]

```
### DP 11: Triangle: Fixed starting and variable ending points

```python
'''
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

 

Example 1:

Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
Example 2:

Input: triangle = [[-10]]
Output: -10
 

Constraints:

1 <= triangle.length <= 200
triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1
-104 <= triangle[i][j] <= 104
 

Follow up: Could you do this using only O(n) extra space, where n is the total number of rows in the triangle?
'''
# recurrance relation
def minimumPathSum(triangle, n):

    def f(i, j): # min path sum to go from index (i, j) to the last row(i == n - 1 and j varies from 0 to n - 1)
        # base case
        if i == n - 1:
            return triangle[i][j] # just return the cell you are at, all are valid answers(all cols are valid)
        
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
### DP 12: Minimum/Maximum Falling Path sum in the matrix | variable startng and variable ending points

```python
def getMaxPathSum(matrix):

    N, M = len(matrix), len(matrix[0])

    def f(i, j): # maximum path sum from index (0, j); j e [0, M - 1] to index (i, j)
        # base case
        if j < 0 or j >= M:
            return -1e8

        if i == 0:
            return matrix[i][j] # all cols are valid ending points
        
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

### DP 14: Subset sum equal to k
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
### DP 15: Partition equal subset sum
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
### DP 16: Partition a set into 2 subsets with minimum absolute difference
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

### DP 17: Counts subsets with sum k
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
    if arr[0] <= k:
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
### DP 18: Counts partitions with given difference
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
## General structure of 0 / 1 knapsack type of problems - VVI for interviews

### DP 19: 0 / 1 Knapsack
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
### DP 20: Minimum Coins

```python

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        def f(i, target):
            # base cases
            if i == 0:
                if target % coins[0] == 0:
                    return target // coins[0]
                else:
                    return -1
            if target == 0:
                return 0
        
            # gen case
            notTake = 0 + f(i - 1, target)
            take = float("inf")
            if coins[i] <= target:
                take = 1 + f(i, target - coins[i])
            
            return min(take, notTake)
        
        return f(n - 1, amount)

    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        def f(i, target):
            # base cases
            if dp[i][target] != -1:
                return dp[i][target]
            if i == 0:
                if target % coins[0] == 0:
                    return target // coins[0]
                else:
                    return -1
            if target == 0:
                return 0
        
            # gen case
            notTake = 0 + f(i - 1, target)
            take = float("inf")
            if coins[i] <= target:
                take = 1 + f(i, target - coins[i])
            
            dp[i][target] = min(take, notTake)
            return dp[i][target]

        dp = [[-1] * (amount + 1) for _ in range(n)]
        return f(n - 1, amount)

    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        def f(i, target):
            # base cases
            if dp[i][target] != -1:
                return dp[i][target]

            if i == 0:
                if target % coins[0] == 0:
                    return target // coins[0]
                else:
                    return float("inf")
        
            # gen case
            notTake = 0 + f(i - 1, target)
            take = float("inf")
            if coins[i] <= target:
                take = 1 + f(i, target - coins[i])
            
            dp[i][target] = min(take, notTake)

            return dp[i][target]
            
        dp = [[-1] * (amount + 1) for _ in range(n)]
        
        res = f(n - 1, amount)
        
        if res == float("inf"):
            return -1
        else:
            return res

    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n)]
        # base cases
        # i == 0
        for target in range(1, amount + 1):
            # if target < coins[0]:
            #     dp[0][target] = float("inf")
            #     continue
            if target % coins[0]:
                dp[0][target] = float("inf")
                continue
            dp[0][target] = target // coins[0]

        # gen case
        for i in range(1, n):
            for target in range(1, amount + 1):
                # gen case
                notTake = 0 + dp[i - 1][target]
                take = float("inf")
                if coins[i] <= target:
                    take = 1 + dp[i][target - coins[i]]
                dp[i][target] = min(take, notTake)
        res = dp[n - 1][amount]
        if res == float("inf"):
            return -1
        else:
            return res

    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        prev = [0] * (amount + 1)
        cur = [0] * (amount + 1)
        # base cases
        # i == 0
        for target in range(1, amount + 1):
            # if target < coins[0]:
            #     dp[0][target] = float("inf")
            #     continue
            if target % coins[0]:
                prev[target] = float("inf")
                continue
            prev[target] = target // coins[0]

        # gen case
        for i in range(1, n):
            for target in range(1, amount + 1):
                # gen case
                notTake = 0 + prev[target]
                take = float("inf")
                if coins[i] <= target:
                    take = 1 + cur[target - coins[i]]
                cur[target] = min(take, notTake)
            prev = cur.copy()
        res = prev[amount]
        if res == float("inf"):
            return -1
        else:
            return res
```
### DP 21: Target Sum

```python
def findWays(arr: List[int], T: int) -> int:
    n = len(arr)
    dp = [[0] * (T + 1) for _ in range(n)]
    # base case
    # for i == 0 and for sum == 0
    if arr[0] == 0:
        dp[0][0] = 2
    else:
        dp[0][0] = 1
    
    # for i == 0 and for sum != 0, ie. k != 0
    if arr[0] != 0 and arr[0] <= T:
        dp[0][arr[0]] = 1

    # general case
    for i in range(1, n):
        for s in range(T + 1):
            notTake = dp[i - 1][s]
            take = 0
            if arr[i] <= s:
                take = dp[i - 1][s - arr[i]]
            dp[i][s] = (notTake + take)
    # return
    return dp[n - 1][T]

def countPartitions(n: int, d: int, arr: List[int]) -> int:
    newTarget = sum(arr) - d
    if newTarget < 0 or newTarget % 2:
        return 0
    return findWays(arr, newTarget // 2)

def targetSum(arr: List[int], target: int) -> int:
    return countPartitions(len(arr), target, arr)
```
### DP 22: Coin Change 2

```python
def countWaysToMakeChange(denominations, value) :
    n = len(denominations)
    def f(ind, T):
        if ind == 0:
            return T % denominations[ind] == 0
        
        if dp[ind][T] != -1:
            return dp[ind][T]

        notTake = f(ind - 1, T)
        take = 0
        if denominations[ind] <= T: # can take this coin and subtract from target its value but still stay at that idx
            take = f(ind, T - denominations[ind])
        
        dp[ind][T] = take + notTake
        return dp[ind][T]
    
    dp = [[-1] * (value + 1) for _ in range(n)]

    return f(n - 1, value)


def countWaysToMakeChange(denominations, value) :
    n = len(denominations)

    dp = [[0] * (value + 1) for _ in range(n)]

    # for i == 0 case
    for T in range(value + 1):
        dp[0][T] = (T % denominations[0] == 0)

    for ind in range(1, n):
        for T in range(value + 1):
            notTake = dp[ind - 1][T]

            take = 0
            if denominations[ind] <= T:
                take = dp[ind][T - denominations[ind]]
            
            dp[ind][T] = take + notTake
        
    return dp[n - 1][T]

def countWaysToMakeChange(denominations, value) :
    n = len(denominations)

    prev = [0 for _ in range(value + 1)]
    cur = [0 for _ in range(value + 1)]
    
    # for i == 0 case
    for T in range(value + 1):
        prev[T] = (T % denominations[0] == 0)

    for ind in range(1, n):
        for T in range(value + 1):
            notTake = prev[T]

            take = 0
            if denominations[ind] <= T:
                take = cur[T - denominations[ind]]
            
            cur[T] = take + notTake
        
        prev = cur.copy()
        
    return prev[T]
```
### DP 23: Unbounded Knapsack

```python
def unboundedKnapsack(n: int, w: int, profit: List[int], weight: List[int]) -> int:

    def f(ind, cap): # max profit from elements 0 -> ind having bag capacity = cap
        # base case
        if ind == 0:
            return (cap // weight[0]) * profit[0]

        if dp[ind][cap] != -1:
            return dp[ind][cap]

        # general case
        notTake = f(ind - 1, cap)
        take = float("-inf")
        if weight[ind] <= cap:
            take = profit[ind] + f(ind, cap - weight[ind])

        # return
        dp[ind][cap] = max(take, notTake)
        return dp[ind][cap]
    
    dp = [[-1] * (w + 1) for _ in range(n)]

    return f(n - 1, w)

def unboundedKnapsack(n: int, w: int, profit: List[int], weight: List[int]) -> int:
    dp = [[0] * (w + 1) for _ in range(n)]

    # base case i == 0
    for cap in range(w + 1):
        dp[0][cap] = (cap // weight[0]) * profit[0]
    
    for ind in range(1, n):
        for cap in range(w + 1):
            notTake = dp[ind - 1][cap]
            take = float("-inf")
            if weight[ind] <= cap:
                take = profit[ind] + dp[ind][cap - weight[ind]]
            dp[ind][cap] = max(take, notTake)
    return dp[n - 1][w]

def unboundedKnapsack(n: int, w: int, profit: List[int], weight: List[int]) -> int:
    prev = [0 for _ in range(w + 1)]
    cur = [0 for _ in range(w + 1)]    

    # base case i == 0
    for cap in range(w + 1):
        prev[cap] = (cap // weight[0]) * profit[0]
    
    for ind in range(1, n):
        for cap in range(w + 1):
            notTake = prev[cap]
            take = float("-inf")
            if weight[ind] <= cap:
                take = profit[ind] + cur[cap - weight[ind]]       
            cur[cap] = max(take, notTake)
        prev = cur.copy()
    return prev[w]

def unboundedKnapsack(n: int, w: int, profit: List[int], weight: List[int]) -> int:
    prev = [0 for _ in range(w + 1)]
    # base case i == 0
    for cap in range(w + 1):
        prev[cap] = (cap // weight[0]) * profit[0]
    
    for ind in range(1, n):
        for cap in range(w + 1):
            notTake = prev[cap]
            take = float("-inf")
            if weight[ind] <= cap:
                take = profit[ind] + prev[cap - weight[ind]]
            prev[cap] = max(take, notTake)
    return prev[w]
```
### DP 24: Rod Cutting Problem

```python
from sys import stdin
import sys

def cutRod(price, n):
    def f(ind, N):
        if ind == 0:
            return N * price[ind]
        if dp[ind][N] != -1:
            return dp[ind][N]

        notTake = f(ind - 1, N)
        take = float("-inf")
        rodLen = ind + 1
        if rodLen <= N:
            take = price[ind] + f(ind, N - rodLen)

        dp[ind][N] = max(take, notTake)
        return dp[ind][N]

    dp = [[-1] * (n + 1) for _ in range(n)]
    return f(n - 1, n)

def cutRod(price, n):

    dp = [[0] * (n + 1) for _ in range(n)]
    
    for i in range(n + 1):
        dp[0][i] = i * price[0]

    for ind in range(n):
        for N in range(n + 1):
            notTake = dp[ind - 1][N]
            take = float("-inf")
            rodLen = ind + 1
            if rodLen <= N:
                take = price[ind] + dp[ind][N - rodLen]
            
            dp[ind][N] = max(take, notTake)

    return dp[n - 1][n]

def cutRod(price, n):

    prev = [0 for _ in range(n + 1)]
    cur = [0 for _ in range(n + 1)]
    
    for i in range(n + 1):
        prev[i] = i * price[0]

    for ind in range(n):
        for N in range(n + 1):
            notTake = prev[N]
            take = float("-inf")
            rodLen = ind + 1
            if rodLen <= N:
                take = price[ind] + cur[N - rodLen]
            
            cur[N] = max(take, notTake)
        prev = cur.copy()

    return prev[n]

def cutRod(price, n):

    prev = [0 for _ in range(n + 1)]
    
    for i in range(n + 1):
        prev[i] = i * price[0]

    for ind in range(n):
        for N in range(n + 1):
            notTake = prev[N]
            take = float("-inf")
            rodLen = ind + 1
            if rodLen <= N:
                take = price[ind] + prev[N - rodLen]
            
            prev[N] = max(take, notTake)

    return prev[n]

# Taking input using fast I/O.
def takeInput():
    n = int(input())

    price = list(map(int, input().strip().split(" ")))

    return price, n


# Main.
t = int(input())
while t:
    price, n = takeInput()
    print(cutRod(price, n))
    t = t-1
```
### Leetcode(M): Perfect Squares
```python
class Solution:
    # def numSquares(self, n: int) -> int:
    #     num = 1
    #     ps = num * num
    #     arr = []
    #     while ps <= n:
    #         arr.append(ps)
    #         num += 1
    #         ps = num * num
        
    #     def f(i, target):
    #         # base case
    #         if target == 0:
    #             return 0
    #         if i == 0:
    #             return target // arr[0]
    #         # gen case
    #         notTake = f(i - 1, target)
    #         take = float("inf")
    #         if arr[i] <= target:
    #             take = 1 + f(i, target - arr[i])

    #         return min(take, notTake)

    #     return f(len(arr) - 1, n)

    # def numSquares(self, n: int) -> int:
    #     num = 1
    #     ps = num * num
    #     arr = []
    #     while ps <= n:
    #         arr.append(ps)
    #         num += 1
    #         ps = num * num
        
    #     def f(i, target):
    #         # base case
    #         if dp[i][target] != -1:
    #             return dp[i][target]

    #         if target == 0:
    #             return 0
    #         if i == 0:
    #             return target // arr[0]
            
    #         # gen case
    #         notTake = f(i - 1, target)
    #         take = float("inf")
    #         if arr[i] <= target:
    #             take = 1 + f(i, target - arr[i])

    #         dp[i][target] = min(take, notTake)
    #         return dp[i][target]

    #     N = len(arr)
    #     dp = [[-1] * (n + 1) for _ in range(N)]
    #     return f(N - 1, n)

    # def numSquares(self, n: int) -> int:
    #     num = 1
    #     ps = num * num
    #     arr = []
    #     while ps <= n:
    #         arr.append(ps)
    #         num += 1
    #         ps = num * num

    #     N = len(arr)
    #     dp = [[0] * (n + 1) for _ in range(N)]

    #     # base case
    #     # if target == 0
    #     # for i in range(N):
    #     #     dp[i][0] = 0
    #     # if i == 0
    #     for target in range(n + 1):
    #         if target % arr[0] == 0:
    #             dp[0][target] = target // arr[0]

    #     # general case
    #     for i in range(1, N):
    #         for target in range(1, n + 1):
    #             # gen case
    #             notTake = dp[i - 1][target]

    #             take = float("inf")
    #             if arr[i] <= target:
    #                 take = 1 + dp[i][target - arr[i]]

    #             dp[i][target] = min(take, notTake)
        
    #     return dp[N - 1][n]

    def numSquares(self, n: int) -> int:
        num = 1
        ps = num * num
        arr = []
        while ps <= n:
            arr.append(ps)
            num += 1
            ps = num * num

        N = len(arr)
        
        prev = [0] * (n + 1)
        cur = [0] * (n + 1)

        # base case
        for target in range(n + 1):
            if target % arr[0] == 0:
                prev[target] = target // arr[0]

        # general case
        for i in range(1, N):
            for target in range(1, n + 1):
                # gen case
                notTake = prev[target]

                take = float("inf")
                if arr[i] <= target:
                    take = 1 + cur[target - arr[i]]

                cur[target] = min(take, notTake)
            
            prev = cur.copy()
        
        return prev[n]
```
### Leetcode(M): Decode ways
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        def f(i):
            if i >= n:
                return 1 
            if s[i] == "0":
                return 0

            # gen case
            takeOne = f(i + 1)
            
            takeTwo = 0
            if ((i + 1) < n):
                if (s[i] == "1" or 
                    s[i] == "2" and s[i + 1] in "0123456"):
                    takeTwo = f(i + 2)
            
            return takeOne + takeTwo
        
        return f(0)

    def numDecodings(self, s: str) -> int:
        n = len(s)
        def f(i):
            if dp[i] != -1:
                return dp[i]
            if i >= n:
                return 1 
            if s[i] == "0":
                return 0

            # gen case
            takeOne = f(i + 1)
            takeTwo = 0
            if ((i + 1) < n):
                if (s[i] == "1" or 
                    s[i] == "2" and s[i + 1] in "0123456"):
                    takeTwo = f(i + 2)
            
            dp[i] = takeOne + takeTwo
            return dp[i]
        dp = [-1] * (n + 1)
        return f(0)

    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        # base case
        dp[n] = 1
        # gen case
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                continue
            takeOne = dp[i + 1]
            takeTwo = 0
            if i + 1 < n and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456')):
                takeTwo = dp[i + 2]
            
            dp[i] = takeOne + takeTwo 
        return dp[0]

    def numDecodings(self, s: str) -> int:
        n = len(s)
        # base case
        iPlusOne = 1
        iPlusTwo = 1

        # gen case
        for i in range(n - 1, -1, -1):
            
            takeOne = 0
            if s[i] != "0":
                takeOne = iPlusOne
            
            takeTwo = 0
            if i + 1 < n and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456')):
                takeTwo = iPlusTwo
            
            temp = takeOne + takeTwo

            iPlusTwo = iPlusOne
            iPlusOne = temp
            
        return iPlusOne
```