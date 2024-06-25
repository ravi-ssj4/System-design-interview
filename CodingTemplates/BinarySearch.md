### General Code:
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        low, high = 0, n - 1
        while low <= high:
            mid = low + (high - low) // 2
            if nums[mid] == target:
                return mid
            elif target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return -1

```

## Pattern 1: Answer variable trick

### Lower Bound
```python
def getLowerBound(self, nums, n, target):
        low, high = 0, n - 1
        ans = n
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] >= target:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans
```
### Upper Bound
```python
def getUpperBound(self, nums, n, target):
    low, high = 0, n - 1
    ans = n
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] > target:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans
```
### Search Insert Position
```python
def searchInsert(self, nums: List[int], target: int) -> int:
        # insert position == lower bound position ie. smallest element >= target
        # if target == an element in nums -> insert pos = idx of that elem
        # if target != any element -> element that's just larger than target = insert position
        n = len(nums)
        return self.getLowerBound(nums, n, target)
```
### Floor/Ceil in an array
```python
def getFloor(arr, n, x):
    # largest element <= x
    low, high = 0, n - 1
    ans = -1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] <= x:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    return ans
    
def getCeil(arr, n, x):
    # smallest element >= x
    low, high = 0, n - 1
    ans = -1
    while low <= high:
        mid = (low + high) // 2
        
        if arr[mid] >= x:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans
    
    

def getFloorAndCeil(arr, n, x):
    arr.sort()
    res = []
    floor = getFloor(arr, n, x)
    if floor != -1:
        res.append(arr[floor])
    else:
        res.append(-1)
    ceil = getCeil(arr, n, x)
    if ceil != -1:
        res.append(arr[ceil])
    else:
        res.append(-1)
    return res
```
### First and Last Occurance of given element
https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        def firstOccurance():
            low, high = 0, n - 1
            ans = -1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] == target:
                    ans = mid
                    high = mid - 1 # in search of a better ans on the left(smaller values)
                elif nums[mid] > target:
                    high = mid - 1 # in search of an ans on the left(smaller values)
                else: # nums[mid] < target
                    low = mid + 1 # in search of an ans on the right(larger values)
            return ans

        def lastOccurance():
            low, high = 0, n - 1
            ans = -1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] == target:
                    ans = mid
                    low = mid + 1
                elif nums[mid] > target:
                    high = mid - 1 # in search of an ans on the left(smaller values)
                else: # nums[mid] < target
                    low = mid + 1 # in search of an ans on the right(larger values)
            return ans
        
        return [firstOccurance(), lastOccurance()]
```
### Find all occurances of given element
https://www.geeksforgeeks.org/problems/number-of-occurrence2259/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=number-of-occurrence
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        def firstOccurance():
            low, high = 0, n - 1
            ans = -1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] == target:
                    ans = mid
                    high = mid - 1 # in search of a better ans on the left(smaller values)
                elif nums[mid] > target:
                    high = mid - 1 # in search of an ans on the left(smaller values)
                else: # nums[mid] < target
                    low = mid + 1 # in search of an ans on the right(larger values)
            return ans

        def lastOccurance():
            low, high = 0, n - 1
            ans = -1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] == target:
                    ans = mid
                    low = mid + 1
                elif nums[mid] > target:
                    high = mid - 1 # in search of an ans on the left(smaller values)
                else: # nums[mid] < target
                    low = mid + 1 # in search of an ans on the right(larger values)
            return ans
        
        return [firstOccurance(), lastOccurance()]
```

## Pattern 2: Array rotation: Find the sorted half first

### Search in rotated sorted array I
https://leetcode.com/problems/search-in-rotated-sorted-array/description/
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # find the half that's sorted
        # check if target lies in that
        # eliminate halves accordingly
        n = len(nums)
        low, high = 0, n - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid

            if nums[low] <= nums[mid]: # left half sorted
                if nums[low] <= target and target <= nums[mid]: # target lies in the sorted left half
                    high = mid - 1
                else:
                    low = mid + 1
            else: # right half sorted
                if nums[mid] <= target and target <= nums[high]: # target lies in the sorted right half
                    low = mid + 1
                else:
                    high = mid - 1
        return -1
```
### Search in rotated sorted array II
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/
```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # find the half that's sorted
        # check if target lies in that
        # eliminate halves accordingly
        n = len(nums)
        low, high = 0, n - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return True

            # sorted half cannot be determined, so shrink from both sides and check again for the sorted half
            if nums[low] == nums[mid] and nums[mid] == nums[high]: 
                low += 1
                high -= 1
                continue

            if nums[low] <= nums[mid]: # left half sorted
                if nums[low] <= target and target <= nums[mid]: # target lies in the sorted left half
                    high = mid - 1
                else:
                    low = mid + 1
            else: # right half sorted
                if nums[mid] <= target and target <= nums[high]: # target lies in the sorted right half
                    low = mid + 1
                else:
                    high = mid - 1
        return False
        
```
### Minimum in rotated sorted array
https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
```python
'''
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
 

Constraints:

n == nums.length
1 <= n <= 5000
-5000 <= nums[i] <= 5000
All the integers of nums are unique.
nums is sorted and rotated between 1 and n times.
'''
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        low, high = 0, n - 1
        mini = float("inf")
        while low <= high:
            mid = low + (high - low) // 2

            if nums[low] <= nums[mid]: # left half sorted
                mini = min(mini, nums[low])
                low = mid + 1
            else:
                mini = min(mini, nums[mid])
                high = mid - 1
        return mini
```
### Find out how many times has an array been rotated
https://www.geeksforgeeks.org/problems/rotation4723/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=rotation
```python
'''
Given an ascending sorted rotated array arr of distinct integers of size n. The array is right-rotated k times. Find the value of k.

Example 1:

Input:
n = 5
arr[] = {5, 1, 2, 3, 4}
Output: 1
Explanation: The given array is 5 1 2 3 4. 
The original sorted array is 1 2 3 4 5. 
We can see that the array was rotated 
1 times to the right.
Example 2:

Input:
n = 5
arr[] = {1, 2, 3, 4, 5}
Output: 0
Explanation: The given array is not rotated.
Your Task:
Complete the function findKRotation() which takes array arr and size n, as input parameters and returns an integer representing the answer. You don't have to print answers or take inputs.

Expected Time Complexity: O(log(n))
Expected Auxiliary Space: O(1)

Constraints:
1 <= n <=105
1 <= arri <= 107
'''
#User function Template for python3
class Solution:
    def findKRotation(self,arr,  n):
        def findMinimum():
            low, high = 0, n - 1
            mini = float("inf")
            miniIdx = -1
            while low <= high:
                mid = low + (high - low) // 2
                
                if arr[low] <= arr[mid]:
                    if arr[low] < mini:
                        mini = arr[low]
                        miniIdx = low
                    low = mid + 1
                else:
                    if arr[mid] < mini:
                        mini = arr[mid]
                        miniIdx = mid
                    high = mid - 1
            return miniIdx
        
        return findMinimum()


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':

	tc=int(input())
	while tc > 0:
		n=int(input())
		a=list(map(int , input().strip().split()))
		ob = Solution()
		ans=ob.findKRotation(a, n)
		print(ans)
		tc=tc-1



# } Driver Code Ends
```
### Single element in sorted array
https://leetcode.com/problems/single-element-in-a-sorted-array/description/
```python
'''
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.

Return the single element that appears only once.

Your solution must run in O(log n) time and O(1) space.

 

Example 1:

Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2
Example 2:

Input: nums = [3,3,7,7,10,11,11]
Output: 10
 

Constraints:

1 <= nums.length <= 105
0 <= nums[i] <= 105
'''
class Solution:

    # def singleNonDuplicate(self, nums: List[int]) -> int:
    #     n = len(nums)
    #     singleElem = -1
    #     if n == 1:
    #         return nums[0]
    #     for i in range(n):
    #         if i == 0:
    #             if nums[i] != nums[i + 1]:
    #                 return nums[i]
    #         if i == n - 1:
    #             if nums[i] != nums[i - 1]:
    #                 return nums[i]
    #         if nums[i] != nums[i - 1] and nums[i] != nums[i + 1]:
    #             return nums[i]


    def singleNonDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        if nums[0] != nums[1]:
            return nums[0]
        
        if nums[n - 1] != nums[n - 2]:
            return nums[n - 1]

        low, high = 1, n - 2
        while low <= high:
            mid = low + (high - low) // 2

            if nums[mid - 1] != nums[mid] and nums[mid] != nums[mid + 1]:
                return nums[mid]
            
            if mid % 2: # we are at an odd idx
                if nums[mid - 1] == nums[mid]: # we are on the left half of the single elem -> go right
                    low = mid + 1
                else: # we are on the right half of the single elem -> go left
                    high = mid - 1
            else: # we are at an even idx 
                if nums[mid] == nums[mid - 1]: # we are on the right half of single elem -> go left
                    high = mid - 1
                else: # we are on the left half of the single elem -> go right
                    low = mid + 1
```
### Find peak element
https://leetcode.com/problems/find-peak-element/submissions/1289004448/
```python
'''
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
 
'''
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        if nums[0] > nums[1]:
            return 0
        if nums[n - 1] > nums[n - 2]:
            return n - 1

        low, high = 1, n - 2 

        while low <= high:
            mid = low + (high - low) // 2

            if nums[mid - 1] < nums[mid] and nums[mid] > nums[mid + 1]:
                return mid
            
            if nums[mid - 1] < nums[mid]: # we are in an increasing slope -> go right
                low = mid + 1
            else: # we are in a decreasing slope -> go left
                high = mid - 1 
        return -1
```
### Square root of a number
https://www.geeksforgeeks.org/problems/square-root/0?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=square-root
```python
'''
Given an integer x, find the square root of x. If x is not a perfect square, then return floor(√x).

 

Example 1:

Input:
x = 5
Output: 2
Explanation: Since, 5 is not a perfect 
square, floor of square_root of 5 is 2.
Example 2:

Input:
x = 4
Output: 2
Explanation: Since, 4 is a perfect 
square, so its square root is 2.
 

Your Task:
You don't need to read input or print anything. The task is to complete the function floorSqrt() which takes x as the input parameter and return its square root.
Note: Try Solving the question without using the sqrt function. The value of x>=0.

 

Expected Time Complexity: O(log N)
Expected Auxiliary Space: O(1)

 

Constraints:
1 ≤ x ≤ 107
'''
#User function Template for python3


#Complete this function
class Solution:
    def floorSqrt(self, x):
        low, high = 1, x
        ans = x
        while low <= high:
            mid = low + (high - low) // 2
            
            square = mid * mid
            
            if square <= x:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans
            
            
#{ 
 # Driver Code Starts
#Initial Template for Python 3

import math



def main():
        T=int(input())
        while(T>0):
            
            x=int(input())
            
            print(Solution().floorSqrt(x))
            
            T-=1


if __name__ == "__main__":
    main()
# } Driver Code Ends
```
### Find Nth root of M
https://www.geeksforgeeks.org/problems/find-nth-root-of-m5843/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=find-nth-root-of-m
```python
'''
You are given 2 numbers (n , m); the task is to find n√m (nth root of m).
 

Example 1:

Input: n = 2, m = 9
Output: 3
Explanation: 32 = 9
Example 2:

Input: n = 3, m = 9
Output: -1
Explanation: 3rd root of 9 is not
integer.
 

Your Task:
You don't need to read or print anyhting. Your task is to complete the function NthRoot() which takes n and m as input parameter and returns the nth root of m. If the root is not integer then returns -1.
 

Expected Time Complexity: O(n* log(m))
Expected Space Complexity: O(1)
 

Constraints:
1 <= n <= 30
1 <= m <= 109
'''
#User function Template for python3

class Solution:
	def NthRoot(self, n, m):
		low, high = 1, m
		ans = -1
		while low <= high:
		    mid = low + (high - low) // 2
		    
		    nthRoot = mid ** n
		  #  print(low, high, mid, nthRoot)
		    if nthRoot == m:
		        ans = mid
		        break
		    elif nthRoot < m:
		        low = mid + 1
		    else:
		        high = mid - 1
	    return ans


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
	T=int(input())
	for i in range(T):
		n, m = input().split()
		n = int(n); m = int(m);
		ob = Solution()
		ans = ob.NthRoot(n, m)
		print(ans)
# } Driver Code Ends
```
### Koko eating bananas
https://leetcode.com/problems/koko-eating-bananas/description/
```python
'''
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

 

Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4
Example 2:

Input: piles = [30,11,23,4,20], h = 5
Output: 30
Example 3:

Input: piles = [30,11,23,4,20], h = 6
Output: 23
 

Constraints:

1 <= piles.length <= 104
piles.length <= h <= 109
1 <= piles[i] <= 109
'''
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        low, high = 1, max(piles)

        def calculateTime(k):
            time = 0
            for pile in piles:
                time += math.ceil(pile / k)
            return time

        ans = -1
        while low <= high:
            mid = low + (high - low) // 2

            timeTaken = calculateTime(mid)

            if timeTaken <= h:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans
```
### Minimum number of days to make M bouquets
https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/
```python
'''
You are given an integer array bloomDay, an integer m and an integer k.

You want to make m bouquets. To make a bouquet, you need to use k adjacent flowers from the garden.

The garden consists of n flowers, the ith flower will bloom in the bloomDay[i] and then can be used in exactly one bouquet.

Return the minimum number of days you need to wait to be able to make m bouquets from the garden. If it is impossible to make m bouquets return -1.

 

Example 1:

Input: bloomDay = [1,10,3,10,2], m = 3, k = 1
Output: 3
Explanation: Let us see what happened in the first three days. x means flower bloomed and _ means flower did not bloom in the garden.
We need 3 bouquets each should contain 1 flower.
After day 1: [x, _, _, _, _]   // we can only make one bouquet.
After day 2: [x, _, _, _, x]   // we can only make two bouquets.
After day 3: [x, _, x, _, x]   // we can make 3 bouquets. The answer is 3.
Example 2:

Input: bloomDay = [1,10,3,10,2], m = 3, k = 2
Output: -1
Explanation: We need 3 bouquets each has 2 flowers, that means we need 6 flowers. We only have 5 flowers so it is impossible to get the needed bouquets and we return -1.
Example 3:

Input: bloomDay = [7,7,7,7,12,7,7], m = 2, k = 3
Output: 12
Explanation: We need 2 bouquets each should have 3 flowers.
Here is the garden after the 7 and 12 days:
After day 7: [x, x, x, x, _, x, x]
We can make one bouquet of the first three flowers that bloomed. We cannot make another bouquet from the last three flowers that bloomed because they are not adjacent.
After day 12: [x, x, x, x, x, x, x]
It is obvious that we can make two bouquets in different ways.
 

Constraints:

bloomDay.length == n
1 <= n <= 105
1 <= bloomDay[i] <= 109
1 <= m <= 106
1 <= k <= n
'''
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        low, high = 1, max(bloomDay)

        def getBouquets(totalDays):
            numBouquets = 0
            cnt = 0
            for bloomTime in bloomDay:
                if bloomTime <= totalDays:
                    cnt += 1
                else:
                    cnt = 0
                if cnt == k:
                    numBouquets += 1
                    cnt = 0
            return numBouquets


        ans = -1
        while low <= high:
            mid = low + (high - low) // 2

            numBouquets = getBouquets(mid)
            # print(low, high, mid, numBouquets)
            if numBouquets >= m:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

```
### Capacity to ship packages within D days
https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/
```python
'''
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.

 

Example 1:

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.
Example 2:

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4
Example 3:

Input: weights = [1,2,3,1,1], days = 4
Output: 3
Explanation:
1st day: 1
2nd day: 2
3rd day: 3
4th day: 1, 1
 

Constraints:

1 <= days <= weights.length <= 5 * 104
1 <= weights[i] <= 500
'''
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        low, high = max(weights), sum(weights)

        def calculateDays(cap):
            days = 1
            currentCap = cap
            i = 0
            while i < len(weights):
                weight = weights[i]
                # print(currentCap, weight, days)
                if currentCap - weight >= 0:
                    currentCap -= weight
                else:
                    days += 1
                    currentCap = cap
                    i -= 1
                i += 1
            return days
                

        ans = -1
        while low <= high:
            mid = low + (high - low) // 2

            daysTaken = calculateDays(mid)
            # print(low, high, mid, daysTaken)
            if daysTaken <= days:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans
```
### 
```python
'''

'''

```
### 
```python
'''

'''

```
### 
```python
'''

'''

```
### 
```python
'''

'''

```
### 
```python
'''

'''

```
### 
```python
'''

'''

```
### 
```python
'''

'''

```