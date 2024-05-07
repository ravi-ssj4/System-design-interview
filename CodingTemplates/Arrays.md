
## MEDIUM SECTION
### Q - 4 | Maximum Size Subarray Sum Equals k
```python
class Solution:
    # method 1: brute force: gen all subarrays + keep track of running sum of running window
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        n = len(nums)
        maxLen = 0
        for l in range(n):
            runningSum = 0
            for r in range(l, n):
                runningSum += nums[r]
                if runningSum == k:
                    maxLen = max(maxLen, r - l + 1)
        return maxLen

    # method 2: better: presum concept: (x - k) + k = x -> cannot be optimized further(works for +ve, -ve)
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        n = len(nums)
        runningSum = 0
        maxLen = 0
        hashMap = {}

        for i in range(n):
            runningSum += nums[i]

            if runningSum == k:
                maxLen = max(maxLen, i + 1)
            else:
                rem = runningSum - k
                if rem in hashMap:
                    maxLen = max(maxLen, (i - hashMap[rem]))
            
            if runningSum not in hashMap:
                hashMap[runningSum] = i

        return maxLen

    # method 3: optimized but only for positives: sliding window + tp typical
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        l, r = 0, 0
        n = len(nums)
        runningSum = 0
        maxLen = 0

        while r < n:
            runningSum += nums[r]

            while runningSum > k:
                runningSum -= nums[l]
                l += 1

            if runningSum == k:
                maxLen = max(maxLen, r - l + 1)
            
            r += 1

        return maxLen
```
### L5 | Two sum
```python
# brute: consider every combination of sums for i in range(n) for j in range(n) if i != j sum = a[i] + a[j] == target?
# time: O(n**2)
# space: O(1)

# better: hashing
# time: O(N)
# space: O(N)

def read(n: int, book: [int], target: int) -> str:
    hashSet = set()

    for i in range(n):
        rem = target - book[i]
        if rem in hashSet:
            return "YES"
        hashSet.add(book[i])

    return "NO"    

# optimal: 
# time = O(Nlogn)
# space = O(1)
def read(n: int, book: [int], target: int) -> str:
    book.sort()

    i, j = 0, n - 1
    while i < j:
        if book[i] + book[j] < target: # increase sum
            i += 1
        elif book[i] + book[j] > target: # decrease sum
            j -= 1
        else:
            return "YES"
    return "NO"
```

### L6 - Sort an array of 0s, 1s and 2s

```python
# O(2N) - take count of 0s, 1s and 2s and put them in the array
def sortArray(arr, n):
	cnt0, cnt1, cnt2 = 0, 0, 0
	for i in range(n):
		if arr[i] == 0:
			cnt0 += 1
		elif arr[i] == 1:
			cnt1 += 1
		else:
			cnt2 += 1

	for i in range(cnt0):
		arr[i] = 0
	
	for i in range(cnt0, cnt0 + cnt1):
		arr[i] = 1
	
	for i in range(cnt0 + cnt1, n):
		arr[i] = 2

# O(N) - Dutch National Flag Algorithm
def sortArray(arr, n):
	low, mid, high = 0, 0, n - 1

	while mid <= high:
		if arr[mid] == 0:
			arr[mid], arr[low] = arr[low], arr[mid]
			low += 1
			mid += 1
		elif arr[mid] == 1:
			mid += 1
		else: # can only be 2 because there are only 0s, 1s and 2s in the array
			arr[mid], arr[high] = arr[high], arr[mid]
			high -= 1
```

### L7 - Majority Element I
```python
# Brute: consider each element as majority and check if freq > N / 2
# time: O(n ** 2)
# space: O(1)

# better: hashing
# keep track of freq of each element and finally whenever freq of any element becomes > N / 2, return it
# time: O(N)
# space: O(N)
def majorityElement(v: [int]) -> int:
    n = len(v)
    hashMap = {}
    for i in range(n):
        hashMap[v[i]] = 1 + hashMap.get(v[i], 0)
        if hashMap[v[i]] > n / 2:
            return v[i]
    
# Optimal: Moore's voting algorithm + verification
def majorityElement(v: [int]) -> int:
    majElem = None
    cnt = 0
    for num in v:
        if cnt == 0:
            majElem = num
            cnt = 1
        
        if majElem == num:
            cnt += 1
        else:
            cnt -= 1
    
    return majElem
```
### L8 - Kadane's algorithm

```python
from os import *
from sys import *
from collections import *
from math import *

from sys import stdin,setrecursionlimit
setrecursionlimit(10**7)

# just give the maximum subarray sum
def maxSubarraySum(arr, n) :
    Sum = 0
    maxi = float("-inf")

    for i in range(n):
        Sum += arr[i]
    
        if Sum > maxi:
            maxi = Sum
        
        if Sum < 0:
            Sum = 0
    
    return maxi if maxi >= 0 else 0

# Variation: give the start and end indices of the subarray with maxsum
def maxSubarraySum(arr, n) :
    Sum = 0
    maxi = float("-inf")
    startIdx, endIdx = -1, -1
    for i in range(n):
        if Sum == 0:
            start = i
        Sum += arr[i]
    
        if Sum > maxi:
            maxi = Sum
            startIdx = start
            endIdx = i
        
        if Sum < 0:
            Sum = 0

    # return [startIdx, endIdx]   
    return maxi if maxi >= 0 else 0
```
### L9 - Rearrange array elements by sign
```python
from typing import *
# brute: put both pos and neg nos in separate arrays
# put them back in the original array at correct position post rearrangement
# arr[2i] = pos[i]
# arr[2i + 1] = neg[i] 
# time: O(n + n / 2)
# space: O(n/2 + n/2) = O(n)
def alternateNumbers(a : List[int]) -> List[int]:
    pos, neg = [], []

    n = len(a)
    for i in range(n):
        if a[i] > 0:
            pos.append(a[i])
        else:
            neg.append(a[i])
    
    for i in range(n // 2):
        a[2 * i] = pos[i]
        a[2 * i + 1] = neg[i]
    
    return a

# optimal: use pos and neg pointers to index into the answer array
# time: O(n) - single pass
# space: O(n)
def alternateNumbers(a : List[int]) -> List[int]:
    n = len(a)
    pos, neg = 0, 1
    ans = [0] * n
    for i in range(n):
        if a[i] > 0:
            ans[pos] = a[i]
            pos += 2
        else:
            ans[neg] = a[i]
            neg += 2
    return ans
```
### L10 - Best time to buy and sell stock
```python
# brute: go through the array 2 times with i as the buying price and j as the selling price excluding the case when i == j
# for each combination, calculate profit, and update maxProfit
# time: O(n**2)
# space: O(1)

# optimal: for an index i, max profit = sell price at idx i - min from index 0 till i - 1. Keep on doing this for each element from l to r
# time: O(N)
# space: O(1)
def maximumProfit(prices):
    maxProfit = 0
    mini = prices[0]
    n = len(prices)
    for i in range(n):
        profit = prices[i] - mini
        maxProfit = max(maxProfit, profit)

        mini = min(mini, prices[i])
    
    return maxProfit
```

### L11 - Next permutation
```python

```
### L12 - Leaders in an array

```python
from typing import *

# brute: consider each element as superior and check all elements to its right if they are smaller or not
# time: O(n**2)

# optimal: keep track of greatest element seeen so far and move from right to left, for each new element greater, add it to the result
# time: O(n)
# space: O(1)
def superiorElements(a : List[int]) -> List[int]:
    n = len(a)
    greatest = a[n - 1]
    res = [greatest]
    for i in range(n - 2, -1, -1):
        if a[i] > greatest:
            res.append(a[i])
            greatest = a[i]
    
    return res
```
### L13 - Longest consecutive sequence

```python

```
### L17 - Count Subarray Sum Equals K

```python
class Solution:
    # METHOD 1: brute force: generate all subarrays and keep track of runningSum
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cnt = 0

        for l in range(n):
            Sum = 0
            for r in range(l, n):
                Sum += nums[r]
                if Sum == k:
                    cnt += 1
                if Sum > k:
                    break
        return cnt    

    # method 2: optimal - prefixsum: keep track of running sum + (x - k) + k = x trick!
    # a map with prefix sum, cnt will keep track
    # time: O(N)
    # space: O(N)
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cnt = 0
        Sum = 0
        hashMap = {0: 1}

        for i in range(n):
            Sum += nums[i]
            rem = Sum - k
            if rem in hashMap:
                cnt += hashMap[rem]
            hashMap[Sum] = 1 + hashMap.get(Sum, 0)
        
        return cnt

```

### Q -  | Binary Subarrays with Sum


### Q -  | 

```python

```
### Q -  | 

```python

```
### Q -  | 

```python

```

### Q -  | 

```python

```
### Q -  | 

```python

```
### Q -  | 

```python

```
### Q -  | 

```python

```
### Q -  | 

```python

```
