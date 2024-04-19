
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

### Q - 17 | 560. Subarray Sum Equals K

```python
class Solution:
    # METHOD 1: brute force: generate all subarrays and keep track of runningSum
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cnt = 0

        for l in range(n):
            runningSum = 0
            for r in range(l, n):
                runningSum += nums[r]
                if runningSum == k:
                    cnt += 1
        return cnt    

    # method 2: better - prefixsum: keep track of running sum + (x - k) + k = x trick!
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cnt = 0
        runningSum = 0
        hashMap = {0: 1}

        for i in range(n):
            runningSum += nums[i]
            rem = runningSum - k
            if rem in hashMap:
                cnt += hashMap[rem]
            hashMap[runningSum] = 1 + hashMap.get(runningSum, 0)
        
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
