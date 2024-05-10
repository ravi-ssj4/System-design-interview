
## MEDIUM SECTION


### L1 | Check if Array Is Sorted and Rotated
```python
class Solution:
    def check(self, nums: List[int]) -> bool:
        n = len(nums)
        flag = 0
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]: # the % n just takes care of the last index -> an edge case
                if flag == 0:
                    flag = 1
                else:
                    return False
        return True
        
# 0 1 2 3
# 2 1 3 4
# nums[i] = 2 -> 1 -> 3 -> 4
# nums[(i + 1) % n] = 1 -> 3 -> 4 -> 2((3 + 1) % 4 = 0 and nums[0] = 2)
# flag = 0 -> 1

```

### L1 | Remove duplicates from sorted array
```python
class Solution:
    # brute: put everything in a set and return it as a list
    # time: O(n)
    # space: O(n)

    # optimal: Two pointer approach
    # from 0 to pos i, everything is placed at the correct position
    # j is the explorer
    # whenever j sees a new element, increment i and swap both i and j elements
    # time: O(n)
    # space: O(1)
    def removeDuplicates(self, nums: List[int]) -> int:
        i, j = 0, 0
        n = len(nums)
        while j < n:
            if nums[i] != nums[j]: # new num found
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
            j += 1
        return i + 1
```

### L1 | Union of two sorted arrays
```python
# brute: hashing using hashSet
    # time: O(n) + O(m)
    # space: O(m + n)
    def findUnion(self,arr1,arr2,n,m):
        hashSet = set()
        for i in range(n):
            hashSet.add(arr1[i])
        
        for i in range(m):
            hashSet.add(arr2[i])
        
        return sorted(list(hashSet))

    # optimal: merge sort style 2 pointers
    # time: O(m + n)
    # space: O(1)
    def findUnion(self,arr1,arr2,n,m):
        i, j = 0, 0
        ans = []
        while i < n and j < m:
            if arr1[i] <= arr2[j]:
                if len(ans) == 0 or ans[-1] != arr1[i]:
                    ans.append(arr1[i])
                i += 1
            else:
                if len(ans) == 0 or ans[-1] != arr2[j]:
                    ans.append(arr2[j])
                j += 1
        
        while i < n:
            if len(ans) == 0 or ans[-1] != arr1[i]:
                ans.append(arr1[i])
            i += 1
        
        while j < m:
            if len(ans) == 0 or ans[-1] != arr2[j]:
                ans.append(arr2[j])
            j += 1
                
        return ans
```

### L1 | intersection of two sorted arrays
```python

```

### L2 | Left rotate array by D places
```python

```

### L2 | Move zeroes to the end
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        l, r = 0, 0
        while r < len(nums):
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
            r += 1
```

### L3 | Find missing number
```python
class Solution:
    # nums = [3, 0, 1]
    # total nos = 3
    # range = [0, 1, 2, 3]

    # method 1: sum(range) - sum(nums) = missing number

    # time: O(n)
    # space: O(1)

    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        rangeSum = 0
        sumOfNums = 0
        for i in range(n + 1):
            rangeSum += i
            if i < n:
                sumOfNums += nums[i]

        return rangeSum - sumOfNums

    # method 2: range ^ nums = missing number
    # tiem: O(n)
    # space: O(1)
    # what's happening actually?
    # all same nos become 0 by xoring with each other as a ^ a = 0
    # the missing number does not have a pair, so missing num ^ 0 = missing number
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n + 1):
            res ^= i
            if i < n:
                res ^= nums[i] 
        return res 
```

### L3 | Every num appears twice but 1 num appears once. find that num
```python
class Solution:
    # brute: check for every num if it appears twice
    # time: O(n**2)
    # space: O(1)
    def singleNumber(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            cnt = 0
            for j in range(n):
                if i == j:
                    continue
                if nums[i] == nums[j]:
                    cnt += 1
            if cnt == 0:
                return nums[i]
    
    # better: hashing -> hashArray of size = (max elem in nums) + 1 -> max elem could be 10**12 -> too large
    # hashing -> hashMap -> works fine!
    # time: O(n)
    # space: O(n)
    def singleNumber(self, nums: List[int]) -> int:
        hashMap = {}
        n = len(nums)
        for num in nums:
            hashMap[num] = 1 + hashMap.get(num, 0)

        for key, value in hashMap.items():
            if value == 1:
                return key  

    # optimal: XOR
    # time: O(n)
    # space: O(1)
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

### L3 | Maximum consecutive 1s
```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        cnt = 0
        maxi = 0
        for num in nums:
            if num == 1:
                cnt += 1
                maxi = max(maxi, cnt)
            else:
                cnt = 0
        return maxi
```
### L4 | Maximum Size Subarray Sum Equals k
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
                if runningSum > k: # optimization
                    break
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

# 0 to low - 1 -> 0000s
# low to mid - 1 -> 1111s
# mid to high -> unsorted
# high + 1 to n - 1 -> 2222s
# a[mid] can be 0, 1 or 2, based on that differnt things need to be done
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
        
        if Sum < 0: # carrying of negative running sum makes no sense
            Sum = 0
    
    return maxi if maxi >= 0 else 0

# Variation: give the start and end indices of the subarray with maxsum
def maxSubarraySum(arr, n) :
    Sum = 0
    maxi = float("-inf")
    startIdx, endIdx = -1, -1
    for i in range(n):
        if Sum == 0: # whenever sum == 0, we are making a fresh start
            start = i
        Sum += arr[i]
    
        if Sum > maxi:
            maxi = Sum
            startIdx = start # whenever we get a better sum, obv update the start and end indices
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

# variation 2: what if the number of pos and neg numbers are not equal

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
from typing import *

# brute: generate all permutations + iterate through that list and find our given input + return its next
# time: O(N! + N!)
# space: O(N!)

# better: use inbuilt function
# time: ?
# space: ?

# optimal: sexy observations!
# obs1: rav, rax, rbx 
# ra is the longest common between 1 and 2, and deciding factor is the last char
# rax and rbx have longest common as r only and deciding factor = ax, bx
# so we try to go as far right as possible to figure out the boundary and the remaining chars are the deciders
#             0  1  2  3  4  5  6
# arr[] = [2, 1, 5, 4, 3, 0 , 0]
#                ind
# step 1: from idx n - 1 to index 2 we cannot take any combination as at every index, we have the largest number for that series starting at that index 
# -> monotonically increasing -> so we need to find a number which is smaller than the next number -> 1 in this case
# next to find the immediate next permutation starting with the series [2, .... ], we need to find the least number from the right to left that's greater than the num at 'ind'
# ie. 3 in our case
# step 2: swap 1 and 3
#step 3:  arr = [2, 3, 5, 4, 1, 0, 0] -> 5 4 1 0 0 need to be completely reversed to get the least or the first value of this series -> 2 3 0 0 1 4 5 -> next permutation
# time: O(n + n + n)
# space: O(1)
def reverseList(lst):
    return lst[::-1]

def nextGreaterPermutation(A : List[int]) -> List[int]:
    ind = -1
    n = len(A)
    for i in range(n - 2, -1, -1):
        if A[i] < A[i + 1]:
            ind = i
            break

    if ind == -1: # did not find any number to swap, ie. this is the last permutation in the series, so just reverse everything to go to the first sequence
        return reverseList(A)
    
    for i in range(n - 1, ind, -1):
        if A[i] > A[ind]:
            A[i], A[ind] = A[ind], A[i]
            break

    A[ind+1:] = reverseList(A[ind+1:])
    return A

# solution through a different implementation of reversing a list(manually)
def nextGreaterPermutation(A : List[int]) -> List[int]:
    ind = -1
    n = len(A)

    def reverseList(i, j):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
    
    for i in range(n - 2, -1, -1):
        if A[i] < A[i + 1]:
            ind = i
            break

    if ind == -1: # did not find any number to swap, ie. this is the last permutation in the series, so just reverse everything to go to the first sequence
        return reverseList(0, n - 1)
    
    for i in range(n - 1, ind, -1):
        if A[i] > A[ind]:
            A[i], A[ind] = A[ind], A[i]
            break
            
    reverseList(ind + 1, n - 1)
    return A
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
class Solution:
    # brute: consider each num as start of new sequence and traverse entire array to see its length
    # time:O(n**2)
    # space:O(1)

    # optimal: put everything in a set, iterate over it -> if any num's prev is absent, then that num is start of a new seq
    # count the consecutive elements(sequence length) by searching the consecutive elements in the set
    # time: O(n)
    # space: O(n)
    def longestConsecutive(self, nums: List[int]) -> int:
        hashSet = set(nums)
        maxi = 0
        for num in list(hashSet):
            if num - 1 not in hashSet:
                cnt = 1
                while num + cnt in hashSet:
                    cnt += 1
                maxi = max(maxi, cnt)
        return maxi

```

### L14 - Set Matrix Zeroes | O(1) space
```python
class Solution:
    # brute: set everything to -1 and then to 0
    # time: O(n**3)
    # space: O(1)

    # better: using aux arrays row and col to keep track of which cells to mark
    # time: O(n ** 2)
    # space: O(n + m)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        n, m = len(matrix), len(matrix[0])
        row = [0] * n
        col = [0] * m

        # mark the row and col arrays according to the given matrix
        for r in range(n):
            for c in range(m):
                if matrix[r][c] == 0:
                    row[r] = 1
                    col[c] = 1
        # update the given matrix depending on the markers row and col arrays
        for r in range(n):
            for c in range(m):
                if row[r] == 1 or col[c] == 1:
                    matrix[r][c] = 0
        
        return matrix
    
    # optimal: same as better but here we use the 0th row and 0th col + 1 extra variable (to avoid the overlap) for the aux arrays
    # time: O(n ** 2)
    # space: O(1)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        n, m = len(matrix), len(matrix[0])
        # row: matrix[r][0]; r varies from 0 to n - 1
        # col: matrix[0][c]; c varies from 0 to m - 1

        # mark the immaginary row and col arrays
        col0 = 1
        for r in range(n):
            for c in range(m):
                if matrix[r][c] == 0:
                    matrix[r][0] = 0
                    if c == 0:
                        col0 = 0
                    else:
                        matrix[0][c] = 0
                
        # update the inner matrix first
        for r in range(1, n):
            for c in range(1, m):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0
        # update the 0th col first -> depends on matrix[r][0] where r == 0
        if matrix[0][0] == 0: # mark the entire 0th row
            for c in range(m):
                matrix[0][c] = 0

        # finally update the 0th row -> depends on col0 var
        if col0 == 0:  # mark the entire 0th col
            for r in range(n):
                matrix[r][0] = 0

        return matrix 
```
### L15 - Rotate matrix or image by 90 degrees
```python
class Solution:
    # brute: use temp matrix and put elements according to rearrangement
    # time: O(n * m)
    # space: O(n * m)
    def rotate(self, mat: List[List[int]]) -> None:
        ROWS, COLS = len(mat), len(mat[0])
        ans = [[0] * COLS for _ in range(ROWS)]

        for r in range(ROWS):
            for c in range(COLS):
                ans[c][ROWS - 1 - r] = mat[r][c]
        
        for r in range(ROWS):
            for c in range(COLS):
                mat[r][c] = ans[r][c]

    # optimal: transpose + reverse each row
    # time: O(N / 2 * N / 2) + O(N * N / 2)
    # space: O(1)
    def rotate(self, mat: List[List[int]]) -> None:
        ROWS, COLS = len(mat), len(mat[0])

        def transpose():
            for r in range(ROWS - 1):
                for c in range(r + 1, COLS):
                    mat[r][c], mat[c][r] = mat[c][r], mat[r][c]
        
        def reverseRow(lst):
            return lst[::-1]
        
        transpose()
        for r in range(ROWS):
            mat[r] = reverseRow(mat[r])
        
        return mat

```
### L16 - Spiral Traversal of a matrix
```python
class Solution:
    def spiralOrder(self, matrix : List[List[int]]) -> List[int]:
    ROWS, COLS = len(matrix), len(matrix[0])
    top, right, bottom, left = 0, COLS - 1, ROWS - 1, 0

    ans = []
    while left <= right and top <= bottom:
        # for the first 2 loops, the above while condition is sufficient
        for c in range(left, right + 1):
            ans.append(matrix[top][c])
        
        top += 1

        for r in range(top, bottom + 1):
            ans.append(matrix[r][right])
        
        right -= 1
        # for the next 2 loops, we have to check for conditions:
        # if top <= bottom, then only go from right to left
        # if left <= right, then only go from bottom to top
        if top <= bottom:
            for c in range(right, left - 1, -1):
                ans.append(matrix[bottom][c])
            
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                ans.append(matrix[r][left])
            
            left += 1
    return ans
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
                cnt += hashMap[rem] # these are the prefix sums of x - k, so same num of prefix sums of sum == k will also exist, since overall sum = x - k + k = x
            hashMap[Sum] = 1 + hashMap.get(Sum, 0)
        
        return cnt

```

### L18 - Pascal's Triangle
```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:

        def generateRow(row):
            ans = []
            term = 1
            ans.append(term)
            for c in range(1, row):
                term *= row - c
                term //= c
                ans.append(term)
            return ans
        
        res = []
        for r in range(numRows):
            res.append(generateRow(r + 1))
        return res
```

### L19 - Majority Element II
```python

```

### L20 & L21 - 3 sum and 4 sum
```python

```

### L22 - Number of subarrays with XOR k
```python

```
### L23 - Merge overlapping intervals
```python

```
### L24 - Merge sorted arrays without using extra space
```python

```
### L25 - Find the missing and repeating number
```python

```
### L26 - Count inversions
```python

```
### L27 - Reverse pairs
```python

```
### L28 - Maximum product subarray
```python

```



