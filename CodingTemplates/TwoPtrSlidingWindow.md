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
    # time: O(n), space: O(1)
    def numberOfSubstrings(self, s: str) -> int:
        
        n = len(s)
        lastSeen = [-1] * 3 # to store the index where that particular char was last seen
        cnt = 0

        for i in range(n):
            lastSeen[ord(s[i]) - ord('a')] = i
            if lastSeen[0] != -1 and lastSeen[1] != -1 and lastSeen[2] != -1: # valid window
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
            hashMap[ord(s[r]) - ord('A')] = 1 + hashMap.get(ord(s[r]) - ord('A'), 0)
            maxFreq = max(maxFreq, hashMap[ord(s[r]) - ord('A')])

            # make the window valid by shrinking
            while (r - l + 1) - maxFreq > k:
                hashMap[ord(s[l]) - ord('A')] -= 1
                tempMaxFreq = 0
                for value in hashMap.values(): # O(n)
                    tempMaxFreq = max(tempMaxFreq, value) 
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

            # make the window valid by shrinking -> ignore to decrease maxFreq -> dosen't affect ans
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
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        cnt = 0
        n = len(nums)

        for l in range(n):
            runningSum = 0
            for r in range(l, n):
                runningSum += nums[r]
                if runningSum == goal:
                    cnt += 1
                if runningSum > goal:
                    break
        return cnt


    # method 2: better: presum + hashmap -> we can optimize this further as there are no -ve nos
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
            runningSum = 0
            for r in range(l, n):
                runningSum += nums[r] % 2
                if runningSum == k:
                    cnt += 1
                if runningSum > k:
                    break
        return cnt

    # method 2: better: presum + hashmap -> we can optimize this further as there are no -ve nos
    # time: O(n), space: O(n)
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        runningSum = 0
        hashMap = {0: 1}
        cnt = 0

        for i in range(n):
            runningSum += nums[i] % 2

            rem = runningSum - k
            if rem in hashMap:
                cnt += hashMap[rem]
            
            hashMap[runningSum] = 1 + hashMap.get(runningSum, 0)

        return cnt

    # method 3: optimal: sw + tp -> as there are no -ve nos
    # time: O(2 * 2 *n) = O(n) but slower than method 2, space: O(1)
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        
        n = len(nums)
        
        def numSubarraysSumLessThan(k):
            cnt = 0
            l, r = 0, 0
            runningSum = 0
            while r < n:
                runningSum += nums[r] % 2
                while l <= r and runningSum > k:
                    runningSum -= nums[l] % 2
                    l += 1
                
                if runningSum <= k:
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

        def helper(K):
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
        sIndex = -1
        minLen = float("inf")

        for l in range(n):
            cnt = 0
            hashArray = [0] * 256
            for i in range(m):
                hashArray[ord(t[i]) - ord('A')] += 1
            for r in range(l, n):
                if hashArray[ord(s[r]) - ord('A')] > 0:
                    cnt += 1
                hashArray[ord(s[r]) - ord('A')] -= 1

                if cnt == m:
                    if (r - l + 1) < minLen:
                        minLen = r - l + 1
                        sIndex = l
                        break
        print(sIndex, minLen)
        return s[sIndex:sIndex+minLen] if sIndex != -1 else ""




    # method 2: Better & optimal: SW + TP + 2 hashMaps + smart way to figure out valid window condition
    # time: O(n) 
    def minWindow(self, s: str, t: str) -> str:
        n = len(s)
        m = len(t)
        hashArray = [0] * 256
        cnt = 0
        sIndex = -1
        minLen = float("inf")

        for i in range(m):
            hashArray[ord(t[i]) - ord('A')] += 1
        
        l, r = 0, 0

        while r < n:
            # check if there was a previous entry for this char
            if hashArray[ord(s[r]) - ord('A')] > 0:
                cnt += 1
            
            # decrease the frequency of this char
            hashArray[ord(s[r]) - ord('A')] -= 1

            # until window is valid, keep on calculating minLen and sIndex and updating the hashArray
            while l <= r and cnt == m:
                if (r - l + 1) < minLen:
                    minLen = r - l + 1
                    sIndex = l
                # shrink window for search of a shorter one -> add to the char's freq
                if hashArray[ord(s[l]) - ord('A')] == 0:
                    cnt -= 1
                hashArray[ord(s[l]) - ord('A')] += 1
                l += 1
            
            r += 1
        print(sIndex, minLen)
        return s[sIndex:sIndex+minLen] if sIndex != -1 else ""



```
