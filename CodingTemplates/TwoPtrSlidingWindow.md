## Patterns

### Fixed / Constant Window
<br>
Q. Find the max Sum possible in a window of size === k

```python
def solution(nums, k):
    l = 0, r = 0, sum = 0, maxSum = 0
    while r < n:
        sum = sum + nums[r]

        if r - l + 1 >= k:
            maxSum = max(maxSum, sum)
            sum = sum - nums[l]
            l += 1

        r += 1

    return maxSum
```

### Longest Subarray / Substring where "some condition"

Q. Find the longest subarray "sum <= k"

```python
# 1. Brute Force: Generate all subarrays / substrings
def solution(nums, k):
    n = len(nums)
    maxSum = float("-inf")
    for l in range(n):
        summ = 0
        for r in range(l, n):
            summ = summ + nums[r]

            if summ <= k:
                maxLen = max(maxLen, r - l + 1)

            # optimization
            if summ > k:
                break
    return maxLen

# 2. Better solution : Sliding window / Two pointer sol.
# time: O(n) + 2 passes
def solution(nums, k):

    n = len(nums)
    l = 0, r = 0, summ = 0, maxLen = float("-inf")        

    while r < n:
        summ = summ + nums[r]

        while summ > k: # O(n) overall
            summ = summ - nums[l]
            l += 1
        
        if sum <= k:
            maxLen = max(maxLen, r - l + 1)
            # res = [l, r] if indices needed

        r += 1

    return maxLen

# Optimized Sliding Window / Two pointer sol.
# time: O(n) + only 1 pass
def solution(nums, k):
    n = len(nums)
    l = 0, r = 0, summ = 0, maxLen = float("-inf")        

    while r < n: # O(n)
        summ = summ + nums[r]

        if summ > k: # O(1)
            summ = summ - nums[l]
            l += 1
        
        if sum <= k:
            maxLen = max(maxLen, r - l + 1)
            # res = [l, r] if indices needed

        r += 1

    return maxLen

```

### Count Number of subarrays "some condition" - HARD - solved using pattern 2 only
Q. Number of subarrays with "sum == k" -> tough to determine whether to expand or shrink
<br> Solution: [Number of subarrays "sum <= k"] - [number of subarrays "sum <= k - 1"]

```python

```

### Finding the shortest / Minimum window "some condition" -> RARE
Q. Shortest window that contains all distinct chars
<br> str: "aabbccabc"

```python
def solution(nums, k):

```

## Questions:

### Q - 1 | Maximum points you can obtain from cards | EASY

```python
def solution(nums, k):
    '''
    [6 2  3   4  7 2 1 7 1] , k = 4
     0 1  2   3  4 5 6 7 8
         k-2 k-1 k      
    '''
    lsum
    for i in range(k):
        lsum += nums[i]
    
    maxSum = lsum
    rsum = 0
    rptr = len(nums) - 1
    for i in range(k - 1, -1, -1):
        lsum -= nums[i]
        rsum += nums[rptr]
        rptr -= 1
        maxSum = max(maxSum, lsum + rsum)

    return maxSum
```

### Q - 2 | Longest Substring without repeating characters | MEDIUM
```python
# Brute force: Generate all possible substrings and keep on checking
# Time: O(n**2)
def solution(s):
    
    n = len(s)
    maxLen = 0

    for l in range(n): # O(n)
        hashArray = [0] * 256
        for r in range(l, n): # O(n)
            if hashArray[s[r]] == 1:
                break
            maxLen = max(maxLen, r - l + 1)
            hashArray[s[r]] = 1
    
    return maxLen

# Sliding window + Two ptr solution
# Time: O(n)
# Space: O(n)
def solution(s):
    '''
    s = cadbzabcd
        012345678
            l
                  r

    hashArray:
        c -> 0
        a -> 1
        d -> 2
        b -> 3
        z -> 4
    '''
    n = len(s)
    hashArray = [-1] * 256
    l, r = 0, 0
    maxLen = 0

    while r < n: # in last step -> r was incremented -> check the new char first
        # if new char is already in the hashArray ->
            # if l < index of this new char ->
                # move l 1 position ahead of that index
        # update maxLen
        # update the index of this new char to r(latest index)
        # update r = r + 1

        if hash[s[r]] != -1:
            if hash[s[r]] >= l:
                l = hash[s[r]] + 1
        
        maxLen = max(maxLen, r - l + 1)

        hash[s[r]] = r
        r += 1




```


### Q - 3 | Maximum consecutive 1s "allowed to flip atmost k 0s to 1s" | HARD
```python

'''
nums = [1 1 1 0 0 0 1 1 1 1 0], k = 2
        l       r           => len = 5 => subarray of atmost 2 zeroes
                l         r => len = 6 => subarray of atmost 2 zeroes

Hence, question can be changed to: Longest subarray with atmost 2 zeroes - TRICK!

'''
# Brute force: Generate all subarrays and keep on checking
# time: O(2**n), space = O(1)
def solution(nums, k):
    
    n = len(nums)
    maxLen = 0

    for l in range(n)
        zeroes = 0
        for r in range(l, n):
            if nums[r] == 0:
                zeroes += 1
            if zeroes <= k:
                maxLen = max(maxLen, r - l + 1)
            if zeroes > k:
                break
    return maxLen

# SL + TP (non-optimized)
# time = O(n) + O(n) = O(2n), space = O(1)
def solution(nums, k):
    n = len(nums)
    l, r = 0, 0
    zeroes = 0
    maxLen = 0

    while r < n:
        if nums[r] == 0:
            zeroes += 1
        
        while zeroes > K:
            if nums[l] == 0:
                zeroes -= 1
            l += 1
        
        if zeroes <= k:
            maxLen = max(maxLen, r - l + 1)
        
        r += 1
    
    return maxLen

# SL + TP (optimized)
# time = O(n), space = O(1)
# intuition: For each iteration of r, if #zeroes > k, I'll only increase l forward 1 pos. -> that way we are eliminating any chance of maxLen getting updated even though #zeroes > k -> we will only update maxLen once #zeroes <= k -> if that's supposed to happen it'll happen as l moves forward by 1 each time, no need to move l multiple times forward in 1 iteration hence.

def solution(nums, k):
    n = len(nums)
    l, r = 0, 0
    maxLen = 0
    zeroes = 0

    while r < n:
        if nums[r] == 0:
            zeroes += 1
        
        if zeroes > k:
            if nums[l] == 0:
                zeroes -= 1
            l += 1
    
        if zeroes <= k:
            maxLen = max(maxLen, r - l + 1)
        
    return maxLen

```

### Q - 4 |
```python

```

### Q - 5 |
```python

```

### Q - 6 |
```python

```
