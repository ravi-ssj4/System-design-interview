## Patterns

### Type 1: Fixed / Constant Window
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

### Type 2: Longest Subarray / Substring where "some condition"

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

### Type 3: Count Number of subarrays "some condition" - HARD - solved using pattern 2 only
Q. Number of subarrays with "sum == k" -> tough to determine whether to expand or shrink

```python
Solution: [Number of subarrays "sum <= k"] - [number of subarrays "sum <= k - 1"]
```

### Type 4: Finding the shortest / Minimum window "some condition" -> RARE
Q. Shortest window that contains k distinct chars
<br> str: "aabbccabc"

```python
def solution(nums, k):
    n = len(nums)
    hashSet = set()
    l, r = 0, 0
    shortestLen = float("inf")
    while r < n:
        
        hashSet.add(nums[r])
        
        while len(hashSet) < k:
            hashSet.remove(nums[l])
            l += 1
        if len(hashSet) == k:
            shortestLen = min(shortestLen, r - l + 1)
        
        r += 1

    return shortestLen

```

## Questions:

### Q - 1 | Maximum points you can obtain from cards | EASY

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        lsum, rsum = 0, 0
        
        for i in range(k):
            lsum += cardPoints[i]

        maxSum = lsum
        
        j = n - 1

        for i in range(k - 1, -1, -1):
            lsum -= cardPoints[i]
            rsum += cardPoints[j]
            j -= 1
            maxSum = max(maxSum, lsum + rsum)
        
        return maxSum

```

### Q - 2 | Longest Substring without repeating characters | MEDIUM
```python
# Brute force: Generate all possible substrings and keep on checking
# Time: O(n**2), space = O(n)
def solution(s):
    
    n = len(s)
    maxLen = 0

    for l in range(n): # O(n) -> 0 to n - 1
        hashArray = [0] * 256
        for r in range(l, n): # O(n) -> l to n - 1
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


### Q - 3 | Maximum consecutive 1s III -> "allowed to flip atmost k 0s to 1s" | HARD
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

### Q - 4 | Fruit into Baskets
```python
class Solution:
    # method 1: Brute force: generate all subarrays and check if distinct fruits are only 2 types
    def totalFruit(self, fruits: List[int]) -> int:
        n = len(fruits)
        maxFruits = 0
        for l in range(n):
            hashMap = defaultdict(int)
            for r in range(l, n):
                hashMap[fruits[r]] += 1
                if len(hashMap) > 2:
                    break
                
                maxFruits = max(maxFruits, r - l + 1)

        return maxFruits

    # method 2: SL + TP + keep track of distinct fruits and their frequencies
    def totalFruit(self, fruits: List[int]) -> int:
        
        n = len(fruits)
        l, r = 0, 0
        maxFruits = 0
        hashMap = {}

        while r < n:
            hashMap[fruits[r]] = 1 + hashMap.get(fruits[r], 0)

            while len(hashMap) > 2:
                hashMap[fruits[l]] -= 1
                if hashMap[fruits[l]] == 0:
                    del hashMap[fruits[l]]
                l += 1
            
            if len(hashMap) <= 2:
                maxFruits = max(maxFruits, r - l + 1)
            
            r += 1
        
        return maxFruits


    # method 3: Same as method 2 but optimized for 1 pass only
    def totalFruit(self, fruits: List[int]) -> int:
        
        n = len(fruits)
        l, r = 0, 0
        maxFruits = 0
        hashMap = {}

        while r < n:
            hashMap[fruits[r]] = 1 + hashMap.get(fruits[r], 0)

            if len(hashMap) > 2:
                hashMap[fruits[l]] -= 1
                if hashMap[fruits[l]] == 0:
                    del hashMap[fruits[l]]
                l += 1
            
            if len(hashMap) <= 2:
                maxFruits = max(maxFruits, r - l + 1)
            
            r += 1
        
        return maxFruits
        
```

### Q - 5 | Longest Substring with atmost k distinct characters
```python
class Solution:
    # method 1: brute force: Generate all substrings and keep track of distinct chars via hashset
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        
        n = len(s)
        maxLen = 0

        for l in range(n):
            hashSet = set()
            for r in range(l, n):
                hashSet.add(s[r])
                if len(hashSet) > k:
                    break
                
                maxLen = max(maxLen, r - l + 1)

        return maxLen

    # method 2: sl + tp + hashMap to keep track of distinct chars and their frequencies
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        n = len(s)
        l, r = 0, 0
        maxLen = 0
        hashMap = {}

        while r < n:

            hashMap[s[r]] = 1 + hashMap.get(s[r], 0)
            
            while len(hashMap) > k:
                hashMap[s[l]] -= 1
                if hashMap[s[l]] == 0:
                    del hashMap[s[l]]
                l += 1
            
            if len(hashMap) <= k:
                maxLen = max(maxLen, r - l + 1)

            r += 1

        return maxLen

    # method 3: same as above but optimized
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        n = len(s)
        l, r = 0, 0
        maxLen = 0
        hashMap = {}

        while r < n:
            hashMap[s[r]] = 1 + hashMap.get(s[r], 0)
            if len(hashMap) > k:
                hashMap[s[l]] -= 1
                if hashMap[s[l]] == 0:
                    del hashMap[s[l]]
                l += 1
            
            if len(hashMap) <= k:
                maxLen = max(maxLen, r - l + 1)
            
            r += 1
        
        return maxLen
```

### Q - 6 | Number of substrings containing all 3 characters
```python
class Solution:
    # Method 1: Brute force: Based on windows starting at index l -> left to right + Gen. all substrings
    # intuition: counting all valid subarrays starting at a particular index
    # time: O(n**2), space: O(3) = O(1)
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        cnt = 0
        for l in range(n):
            hasSeen = [0] * 3 # stores the truthy values, ie. whether this char has been seen or not yet
            for r in range(l, n):
                # update the mapping data
                hasSeen[ord(s[r]) - ord('a')] = 1
                # check validity of the window
                if hasSeen[0] + hasSeen[1] + hasSeen[2] == 3:
                    # cnt = cnt + 1
                    cnt = cnt + (n - r)
                    break
        return cnt


    # Method 2: Better(SW + TP): Based on windows ending at index r -> left to right
    # intuition: counting all valid subarrays ending at a particular index
    # time: O(n), space: O(1)
    def numberOfSubstrings(self, s: str) -> int:
        
        n = len(s)
        lastSeen = [-1] * 3 # to store the index where that particular char was last seen
        cnt = 0

        for i in range(n):
            lastSeen[ord(s[i]) - ord('a')] = i
            # if lastSeen[0] != -1 and lastSeen[1] != -1 and lastSeen[2] != -1: # valid window -> not needed (when window is invalid the min() term will give -1 -> cnt += (1 - 1))
            cnt = cnt + (1 + min(lastSeen[0], lastSeen[1], lastSeen[2]))

        return cnt

```

### Q - 7 | Longest Repeating Character Replacement - (MEDIUM - HARD)

```python
class Solution:
    # method 1: brute force: generate all substrings + formula: {valid window length - maxFreq <= k}
    # time: O(n**2), space: O(26) == O(1)
    def characterReplacement(self, s: str, k: int) -> int:
        
        n = len(s)
        maxLen = 0

        for l in range(n):
            hashArray = [0] * 26
            maxFreq = 0
            for r in range(l, n):
                # update the map and maxFreq
                hashArray[ord(s[r]) - ord('A')] += 1
                maxFreq = max(maxFreq, hashArray[ord(s[r]) - ord('A')])

                # check for validity of the window
                if (r - l + 1) - maxFreq <= k:
                    maxLen = max(maxLen, r - l + 1)
                else:
                    break
        return maxLen        
                
    
    # method 2: Better: (SW + TP + Formula)
    # time: O(n**2) in worst case, space: O(n)
    def characterReplacement(self, s: str, k: int) -> int:

        n = len(s)
        l, r = 0, 0
        hashMap = {}
        maxLen = 0
        maxFreq = 0

        for r in range(n): # O(n)
            # update the map and maxFreq
            hashMap[s[r]] = 1 + hashMap.get(s[r], 0)
            # update max freq with every map update -> because current element can only increase it
            maxFreq = max(maxFreq, hashMap[s[r]])

            # make the window valid by shrinking
            # because current window size - current max freq = the num of chars we can replace to make the entire string equal to the char having current max freq
            while (r - l + 1) - maxFreq > k: 
                hashMap[s[r]] -= 1
                tempMaxFreq = 0
                for value in hashMap.values(): # O(n)
                    tempMaxFreq = max(tempMaxFreq, value) # while shrinking we are re-calculating the max freq which is not needed
                maxFreq = tempMaxFreq
                l += 1

            # check for validity of window
            if (r - l + 1) - maxFreq <= k:
                maxLen = max(maxLen, r - l + 1)
        
        return maxLen

    # method 3: Better: (SW + TP + Formula + ignoring to decrease maxFreq as it dosen't affect answer)
    # time: O(n) in worst case, space: O(n)
    def characterReplacement(self, s: str, k: int) -> int:

        n = len(s)
        l, r = 0, 0
        hashMap = {}
        maxLen = 0
        maxFreq = 0

        for r in range(n): # O(n)
            # update the map and maxFreq
            hashMap[ord(s[r]) - ord('A')] = 1 + hashMap.get(ord(s[r]) - ord('A'), 0)
            maxFreq = max(maxFreq, hashMap[ord(s[r]) - ord('A')])

            # make the window valid by shrinking -> ignore to decrease maxFreq -> dosen't affect ans -> answer will change only when maxFreq increases
            while (r - l + 1) - maxFreq > k:
                hashMap[ord(s[l]) - ord('A')] -= 1
                l += 1

            # check for validity of window
            if (r - l + 1) - maxFreq <= k:
                maxLen = max(maxLen, r - l + 1)
        
        return maxLen
```

### NOTE: Checkout Q-4 and Q-17 of Arrays.md first
### Q - 8 | Binary Subarrays with sum 

```python
class Solution:
    # METHOD 1: BRUTE FORCE: GEN ALL SUBARRAYS + running Sum
    # intuition: for every subarray, if current sum > goal, break else if its equal to goal, add to count
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        cnt = 0
        n = len(nums)

        for l in range(n):
            Sum = 0
            for r in range(l, n):
                Sum += nums[r]
                if Sum == goal:
                    cnt += 1
                if Sum > goal:
                    break
        return cnt


    # method 2: better: presum + hashmap -> we can optimize this further as there are no -ve nos
    # intuition: if there are z num of subarrays with a sum of x - k, there would be z num of subarrays with a sum of k
    # works for all cases
    # time: O(n), space: O(n)
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        n = len(nums)
        runningSum = 0
        hashMap = {0: 1}
        cnt = 0

        for i in range(n):
            runningSum += nums[i]

            rem = runningSum - goal
            if rem in hashMap:
                cnt += hashMap[rem]
            
            hashMap[runningSum] = 1 + hashMap.get(runningSum, 0)

        return cnt


    # method 3: optimal: sw + tp -> as there are no -ve nos
    # pattern 3: count subarrays when sum == k (only positives)
    # time: O(2 * 2 *n) = O(n) but slower than method 2, space: O(1)
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        
        n = len(nums)
        
        def numSubarraysSumLessThan(k):
            cnt = 0
            l, r = 0, 0
            runningSum = 0
            while r < n:
                runningSum += nums[r]
                while l <= r and runningSum > k:
                    runningSum -= nums[l]
                    l += 1
                
                if runningSum <= k:
                    cnt += (r - l + 1)
                
                r += 1
            
            return cnt

        return numSubarraysSumLessThan(goal) - numSubarraysSumLessThan(goal - 1)

```
### Q - 9 | 1248. Count Number of Nice Subarrays -> same question as above but just twisted a bit

```python
class Solution:
    # METHOD 1: BRUTE FORCE: GEN ALL SUBARRAYS + running Sum
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        cnt = 0
        n = len(nums)

        for l in range(n):
            Sum = 0
            for r in range(l, n):
                Sum += nums[r] % 2
                if Sum == k:
                    cnt += 1
                if Sum > k:
                    break
        return cnt

    # method 2: better: presum + hashmap -> we can optimize this further as there are no -ve nos
    # time: O(n), space: O(n)
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        Sum = 0
        hashMap = {0: 1}
        cnt = 0

        for i in range(n):
            Sum += nums[i] % 2

            rem = Sum - k
            if rem in hashMap:
                cnt += hashMap[rem]
            
            hashMap[Sum] = 1 + hashMap.get(Sum, 0)

        return cnt

    # method 3: optimal: sw + tp -> as there are no -ve nos
    # time: O(2 * 2 *n) = O(n) but slower than method 2, space: O(1)
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        
        n = len(nums)
        
        def numSubarraysSumLessThan(k):
            cnt = 0
            l, r = 0, 0
            Sum = 0
            while r < n:
                Sum += nums[r] % 2
                while l <= r and Sum > k:
                    Sum -= nums[l] % 2
                    l += 1
                
                if Sum <= k:
                    cnt += (r - l + 1)
                
                r += 1
            
            return cnt

        return numSubarraysSumLessThan(k) - numSubarraysSumLessThan(k - 1)

```

### Q - 10 | 992. Subarrays with K Different Integers
```python 
class Solution:
    # method 1: brute force: gen. all subarrays + hashMap to keep track of freq of each element
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cnt = 0
        for l in range(n):
            hashMap = {}
            for r in range(l, n):
                # update the hashmap
                hashMap[nums[r]] = 1 + hashMap.get(nums[r], 0)
                if len(hashMap) == k:
                    cnt += 1
                if len(hashMap) > k:
                    break
        return cnt        

    # method 2: better & optimal: sw + tp + hashMap as above + ((<= k) - (<= (k - 1))) = k trick
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        n = len(nums)

        def helper(K): # gives the count of subarrays with sum <= K
            cnt = 0
            l, r = 0, 0
            hashMap = {}
            while r < n:
                hashMap[nums[r]] = 1 + hashMap.get(nums[r], 0)
                while len(hashMap) > K:
                    hashMap[nums[l]] -= 1
                    if hashMap[nums[l]] == 0:
                        del hashMap[nums[l]]
                    l += 1
                if len(hashMap) <= K:
                    cnt += (r - l + 1)      
                r += 1
            return cnt

        return helper(k) - helper(k - 1)
```

### Q - 11 | 76. Minimum Window Substring
```python 
class Solution:
    # method 1: Brute force: generate all substrings + 1 hashArray of size 256 only
    # time: O(n**2), space: O(n)
    def minWindow(self, s: str, t: str) -> str:
        n = len(s)
        m = len(t)
        if n < m:
            return ""
        startIdx = -1
        minLength = float("inf")
        for i in range(n):
            hashMap = defaultdict(int)
            for j in range(m):
                hashMap[t[j]] += 1
            cnt = 0
            for j in range(i, n):
                if hashMap[s[j]] > 0:
                    cnt += 1
                hashMap[s[j]] -= 1
                if cnt == m:
                    if (j - i + 1) < minLength:
                        minLength = (j - i + 1)
                        startIdx = i
                    break
        print(startIdx, minLength)
        if startIdx != -1:
            return s[startIdx:startIdx + minLength]
        else:
            return ""

    # method 2: Better & optimal: SW + TP + 2 hashMaps + smart way to figure out valid window condition
    # time: O(n) 
    def minWindow(self, s: str, t: str) -> str:
        n, m = len(s), len(t)
        hashMap = defaultdict(int)
        for c in t:
            hashMap[c] += 1
        
        startIdx = -1
        minLength = float("inf")
        
        cnt = 0
        l, r = 0, 0
        while r < n:
            if hashMap[s[r]] > 0:
                cnt += 1
            
            hashMap[s[r]] -= 1

            # while cnt == length of t:
            # we can update the minLength, startIdx and shrink the window and check and update again
            while cnt == m: 
                if (r - l + 1) < minLength:
                    minLength = (r - l + 1)
                    startIdx = l
                hashMap[s[l]] += 1
                if hashMap[s[l]] > 0:
                    cnt -= 1
                l += 1
            
            r += 1
        
        if startIdx != -1:
            return s[startIdx: startIdx + minLength]
        else:
            return ""
```

### Leetcode: Subarray product less than k
```python
class Solution:
    # def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    #     n = len(nums)
    #     cnt = 0
    #     for i in range(n):
    #         prod = 1
    #         for j in range(i, n):
    #             prod *= nums[j]
    #             if prod < k:
    #                 cnt += 1
    #             if prod >= k:
    #                 break
    #     return cnt

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        l, r = 0, 0
        cnt = 0
        prod = 1
        while r < n:
            prod *= nums[r]

            while l <= r and prod >= k:
                prod /= nums[l]
                l += 1
            
            if prod < k:
                cnt += (r - l + 1)
            
            r += 1
        return cnt
```

### Leetcode: https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/

```python
'''
You are given an integer array nums and an integer k.

The frequency of an element x is the number of times it occurs in an array.

An array is called good if the frequency of each element in this array is less than or equal to k.

Return the length of the longest good subarray of nums.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,2,3,1,2,3,1,2], k = 2
Output: 6
Explanation: The longest possible good subarray is [1,2,3,1,2,3] since the values 1, 2, and 3 occur at most twice in this subarray. Note that the subarrays [2,3,1,2,3,1] and [3,1,2,3,1,2] are also good.
It can be shown that there are no good subarrays with length more than 6.
Example 2:

Input: nums = [1,2,1,2,1,2,1,2], k = 1
Output: 2
Explanation: The longest possible good subarray is [1,2] since the values 1 and 2 occur at most once in this subarray. Note that the subarray [2,1] is also good.
It can be shown that there are no good subarrays with length more than 2.
Example 3:

Input: nums = [5,5,5,5,5,5,5], k = 4
Output: 4
Explanation: The longest possible good subarray is [5,5,5,5] since the value 5 occurs 4 times in this subarray.
It can be shown that there are no good subarrays with length more than 4.
 

Constraints:

1 <= nums.length <= 105
1 <= nums[i] <= 109
1 <= k <= nums.length
'''
class Solution:
    # def maxSubarrayLength(self, nums: List[int], k: int) -> int:
    #     # brute force

    #     maxi = 0
    #     n = len(nums)
    #     for i in range(n):
    #         hashMap = defaultdict(int)
    #         for j in range(i, n):
    #             hashMap[nums[j]] += 1
    #             if hashMap[nums[j]] > k:
    #                 break
                
    #             maxi = max(maxi, j - i + 1)
    #     return maxi

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        l, r = 0, 0
        hashMap = defaultdict(int)
        maxi = 0
        while r < n:
            hashMap[nums[r]] += 1
            
            while l <= r and hashMap[nums[r]] > k:
                hashMap[nums[l]] -= 1
                l += 1
            
            maxi = max(maxi, r - l + 1)

            r += 1
        
        return maxi
                    
```

### Leetcode HARD: https://leetcode.com/problems/length-of-the-longest-valid-substring/description/

```python
'''
You are given a string word and an array of strings forbidden.

A string is called valid if none of its substrings are present in forbidden.

Return the length of the longest valid substring of the string word.

A substring is a contiguous sequence of characters in a string, possibly empty.

 

Example 1:

Input: word = "cbaaaabc", forbidden = ["aaa","cb"]
Output: 4
Explanation: There are 11 valid substrings in word: "c", "b", "a", "ba", "aa", "bc", "baa", "aab", "ab", "abc" and "aabc". The length of the longest valid substring is 4. 
It can be shown that all other substrings contain either "aaa" or "cb" as a substring. 
Example 2:

Input: word = "leetcode", forbidden = ["de","le","e"]
Output: 4
Explanation: There are 11 valid substrings in word: "l", "t", "c", "o", "d", "tc", "co", "od", "tco", "cod", and "tcod". The length of the longest valid substring is 4.
It can be shown that all other substrings contain either "de", "le", or "e" as a substring. 
 

Constraints:

1 <= word.length <= 105
word consists only of lowercase English letters.
1 <= forbidden.length <= 105
1 <= forbidden[i].length <= 10
forbidden[i] consists only of lowercase English letters.
'''
class Solution:
    # def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    #     n = len(word)
    #     maxi = float("-inf")
    #     for i in range(n): # O(n)
    #         for j in range(i, n): # O(n)
    #             # current substr
    #             curWord = word[i:j + 1]
    #             print(curWord)
    #             present = False
    #             for w in forbidden: # O(m)
    #                 if w in curWord: # O(10)
    #                     present = True
    #                     break
    #             if present == True:
    #                 break
    #             else:
    #                 maxi = max(maxi, j - i + 1)
    #     return maxi
    
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # we will find all the substrings that are forbidden
        n = len(word)
        cnt = 0
        l, r = 0, 0
        curStr = collections.deque()
        forbiddenSet = set(forbidden)
        maxi = float("-inf")
        
        def invalid():
            curS = ""
            for i in range(len(curStr) - 1, -1, -1):
                curS = curStr[i] + curS
                if curS in forbiddenSet:
                    return True
            return False
                
        while r < n:
            curStr.append(word[r])

            
            while invalid():
                curStr.popleft()
                l += 1
            
            maxi = max(maxi, r - l + 1)

            r += 1

        return maxi



```

### Leetcode Medium: https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/
```python
'''
You are given an integer array nums and a positive integer k.

Return the number of subarrays where the maximum element of nums appears at least k times in that subarray.

A subarray is a contiguous sequence of elements within an array.

 

Example 1:

Input: nums = [1,3,2,3,3], k = 2
Output: 6
Explanation: The subarrays that contain the element 3 at least 2 times are: [1,3,2,3], [1,3,2,3,3], [3,2,3], [3,2,3,3], [2,3,3] and [3,3].
Example 2:

Input: nums = [1,4,2,1], k = 3
Output: 0
Explanation: No subarray contains the element 4 at least 3 times.
 

Constraints:

1 <= nums.length <= 105
1 <= nums[i] <= 106
1 <= k <= 105
'''
class Solution:
    # def countSubarrays(self, nums: List[int], k: int) -> int:
    #     # brute force
    #     n = len(nums)
    #     maxi = max(nums)
    #     res = 0
    #     for i in range(n):
    #         countMaxi = 0
    #         for j in range(i, n):
    #             if nums[j] == maxi:
    #                 countMaxi += 1
                
    #             if countMaxi >= k:
    #                 res += (n - j)
    #                 break
    #     return res


    def countSubarrays(self, nums: List[int], k: int) -> int:
        # two pointer sliding window
        n = len(nums)
        l, r = 0, 0
        maxi = max(nums)
        countMaxi = 0
        res = 0
        while r < n:
            if nums[r] == maxi:
                countMaxi += 1
            
            while l <= r and countMaxi >= k:
                res += (n - r)
                if nums[l] == maxi:
                    countMaxi -= 1
                l += 1
            r += 1
        
        return res

```
### Leetcode: https://leetcode.com/problems/repeated-dna-sequences/description/
```python
'''
The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.

For example, "ACGAATTCCG" is a DNA sequence.
When studying DNA, it is useful to identify repeated sequences within the DNA.

Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.

 

Example 1:

Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]
Example 2:

Input: s = "AAAAAAAAAAAAA"
Output: ["AAAAAAAAAA"]
 

Constraints:

1 <= s.length <= 105
s[i] is either 'A', 'C', 'G', or 'T'.
'''
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        if len(s) < 10:
            return []
        
        l, r = 0, 9
        n = len(s)
        dnaSet = set()
        res = []
        while r < n:
            if s[l:r + 1] in dnaSet:
                res.append(s[l:r + 1])
            else:
                dnaSet.add(s[l:r + 1])
            l += 1
            r += 1

        return set(res)
```
### Leetcode: https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/description/
```python
class Solution:
    # def longestSubarray(self, nums: List[int], limit: int) -> int:
    #     n = len(nums)
    #     maxi = 0

    #     for i in range(n):
    #         count = 0
    #         maxx = nums[i]
    #         minn = nums[i]
    #         for j in range(i, n):
    #             if abs(nums[j] - maxx) <= limit and abs(nums[j] - minn) <= limit:
    #                 count += 1
    #                 maxi = max(maxi, count)
    #                 maxx = max(maxx, nums[j])
    #                 minn = min(minn, nums[j])
    #             else:
    #                 break
    #     return maxi
    
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        l, r = 0, 0
        minQ = deque()
        maxQ = deque()
        maxi = 0
        while r < n:
            while minQ and nums[r] < nums[minQ[0]]:
                minQ.popleft()
            minQ.append(r)
            
            while maxQ and nums[r] > nums[maxQ[0]]:
                maxQ.popleft()
            maxQ.append(r)

            print(l, r, nums[l], nums[r])
            if minQ and maxQ:
                print(minQ[0], maxQ[0])

            while l <= r and minQ and maxQ and abs(nums[minQ[0]] - nums[maxQ[0]]) > limit:
                l += 1
                if l <= r:
                    while minQ and minQ[0] <= l:
                        minQ.popleft()
                    while maxQ and maxQ[0] <= l:
                        maxQ.popleft()
            # print(minQ[0], maxQ[0])
            print(l, r, nums[l], nums[r])
            maxi = max(maxi, r - l + 1)

            print(l, r, maxi)
            r += 1
        return maxi
```
### Leetcode
```python

```
### Leetcode
```python

```
### Leetcode
```python

```
### Leetcode
```python

```