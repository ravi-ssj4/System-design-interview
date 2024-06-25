### Subarray product less than k | Sliding window
https://leetcode.com/problems/subarray-product-less-than-k/
```python
'''
Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.

 

Example 1:

Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
Example 2:

Input: nums = [1,2,3], k = 0
Output: 0
 

Constraints:

1 <= nums.length <= 3 * 104
1 <= nums[i] <= 1000
0 <= k <= 106
'''
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        cnt = 0
        prod = 1
        l, r = 0, 0
        n = len(nums)
        while r < n:
            prod = prod * nums[r]
            while l <= r and prod >= k:
                prod = prod // nums[l]
                l += 1
            cnt += (r - l + 1)
            r += 1
        return cnt
```
### Daily Temperatures | Stack
https://leetcode.com/problems/daily-temperatures/description/
```python
'''
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

 

Example 1:

Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
Example 2:

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
Example 3:

Input: temperatures = [30,60,90]
Output: [1,1,0]
 

Constraints:

1 <= temperatures.length <= 105
30 <= temperatures[i] <= 100
'''
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        stack = []
        res = [0] * n
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                idx = stack.pop()
                res[idx] = i - idx
            stack.append(i)
        return res

```
### The kth factor of n
https://leetcode.com/problems/the-kth-factor-of-n/description/
```python
'''
You are given two positive integers n and k. A factor of an integer n is defined as an integer i where n % i == 0.

Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.

 

Example 1:

Input: n = 12, k = 3
Output: 3
Explanation: Factors list is [1, 2, 3, 4, 6, 12], the 3rd factor is 3.
Example 2:

Input: n = 7, k = 2
Output: 7
Explanation: Factors list is [1, 7], the 2nd factor is 7.
Example 3:

Input: n = 4, k = 4
Output: -1
Explanation: Factors list is [1, 2, 4], there is only 3 factors. We should return -1.
 

Constraints:

1 <= k <= n <= 1000
 
'''
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        def getFactors():
            factors = []
            for i in range(1, n // 2 + 1):
                if n % i == 0:
                    factors.append(i)
            return factors + [n]

        factors = getFactors()

        if len(factors) < k:
            return -1
        else:
            return factors[k - 1]
```
### LRU Cache | Doubly linked list + HashMap
```python
'''
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
 

Constraints:

1 <= capacity <= 3000
0 <= key <= 104
0 <= value <= 105
At most 2 * 105 calls will be made to get and put.
'''
class ListNode:
    def __init__(self, val=(-1, -1), prev = None, nxt = None):
        self.val = val
        self.prev = prev
        self.next = nxt

class LRUCache:

    def __init__(self, capacity: int):
        self.hashMap = {}
        self.capacity = capacity
        self.left = ListNode()
        self.right = ListNode()
        self.left.next = self.right
        self.right.prev = self.left
        
    # removal can be a generic one
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
    
    # insertion is to the right always
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.prev = prev
        node.next = nxt

    def get(self, key: int) -> int:
        if key in self.hashMap:
            # get this node in a variable
            queryNode = self.hashMap[key]
            # remove this node from left and insert in the right
            self.remove(queryNode) # lru
            self.insert(queryNode) # mru
            return queryNode.val[1]
        return -1

        

    def put(self, key: int, value: int) -> None:
        if key in self.hashMap:
            self.remove(self.hashMap[key])
        self.hashMap[key] = ListNode((key, value)) # create new node for this key
        self.insert(self.hashMap[key]) # insert at mru

        if len(self.hashMap) > self.capacity:
            lruNode = self.left.next
            key = lruNode.val[0]
            self.remove(lruNode)
            del self.hashMap[key]

        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```
### Maximal Square | DP Tabulation | Filling of table trick!
https://leetcode.com/problems/maximal-square/description/
```python
'''
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

 

Example 1:


Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
Example 2:


Input: matrix = [["0","1"],["1","0"]]
Output: 1
Example 3:

Input: matrix = [["0"]]
Output: 0
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 300
matrix[i][j] is '0' or '1'.
'''
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        ROWS, COLS = len(matrix), len(matrix[0])
        dp = [[0] * COLS for _ in range(ROWS)]

        for r in range(ROWS):
            dp[r][0] = int(matrix[r][0])
        for c in range(COLS):
            dp[0][c] = int(matrix[0][c])
        
        print(matrix)
        print(dp)
        
        for r in range(1, ROWS):
            for c in range(1, COLS):
                if matrix[r][c] == "0":
                    dp[r][c] = 0
                else:
                    dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c-1], dp[r - 1][c - 1])
        
        print(dp)
        maxi = 0
        for r in range(ROWS):
            maxi = max(maxi, max(dp[r]))
        
        return maxi**2
```
### Insert Delete GetRandom O(1) | List + Hashmap (key: number, value: index of that number in the list)
https://leetcode.com/problems/insert-delete-getrandom-o1/description/
```python
'''
Implement the RandomizedSet class:

RandomizedSet() Initializes the RandomizedSet object.
bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.
You must implement the functions of the class such that each function works in average O(1) time complexity.

 

Example 1:

Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
 

Constraints:

-231 <= val <= 231 - 1
At most 2 * 105 calls will be made to insert, remove, and getRandom.
There will be at least one element in the data structure when getRandom is called.
'''
class RandomizedSet:

    def __init__(self):
        self.hashMap = {}
        self.list = []
        

    def insert(self, val: int) -> bool:
        if val in self.hashMap:
            return False
        self.list.append(val)
        self.hashMap[val] = len(self.list) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.hashMap:
            return False
        
        idxInList = self.hashMap[val]
        lastValue = self.list[-1]
        self.list[idxInList] = lastValue
        self.list.pop()
        self.hashMap[lastValue] = idxInList
        del self.hashMap[val] # this should be after the previous statement always. Why?****
        return True
        # ****: the val can be equal to lastVal, in that case, if 23 and 24 were reversed,
        #       after deleting again 1 value which is the same value would be inserted wrongly

    def getRandom(self) -> int:
        return random.choice(self.list)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```
### Stone Game | MinMax game theory
https://leetcode.com/problems/stone-game/
```python
'''

'''
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        n = len(piles)
        winnerScoreNeeded = sum(piles) // 2 + 1
        
        def f(left, right):
            if left > right:
                return 0
            
            if dp[left][right] != -1:
                return dp[left][right]

            pickupLeft = piles[left] - f(left + 1, right)
            pickupRight = piles[right] - f(left, right - 1) 
            
            dp[left][right] = max(pickupLeft, pickupRight)
            return dp[left][right]
        
        dp = [[-1] * (n + 1) for _ in range(n + 1)]
        if f(0, n - 1) > 0:
            return True
        else:
            return False
```
### Task Scheduler | MaxHeap + queue(keep track of tasks gone to sleep and puts them back in the priority queue(maxHeap))
```python
'''

'''
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freqMap = {}
        for task in tasks:
            freqMap[task] = 1 + freqMap.get(task, 0)
        
        maxHeap = []

        for freq in freqMap.values():
            heapq.heappush(maxHeap, -1 * freq)
        
        q = deque()

        time = 0

        while maxHeap or q:
            time += 1

            if maxHeap:
                task = heapq.heappop(maxHeap) * -1
                task = task - 1 # task left after execution
                if task:
                    q.append([task, time + n])
                
            if q:
                if time >= q[0][1]:
                    taskFromQueue = q.popleft()[0]
                    heapq.heappush(maxHeap, -1 * taskFromQueue)
        return time

                
                
```
### Maximize the Greatness of an array | Sort + Two pointer | Try to put just greater number in place of any number
https://leetcode.com/problems/maximize-greatness-of-an-array/description/
```python
'''

'''
class Solution:
    def maximizeGreatness(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        cnt = 0
        l, r = 0, 0
        while r < n:
            if nums[l] < nums[r]:
                cnt += 1
                r += 1
                l += 1
            else:
                r += 1
        return cnt
```
### Search Suggestion System | Trie not needed | Sort the product list + Two pointers on left and right extremes + converge inside(depending on matches with search words' chars)
https://leetcode.com/problems/search-suggestions-system/
```python
'''

'''
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        n = len(products)
        products.sort()
        l, r = 0, n - 1
        m = len(searchWord)
        res = []
        for i in range(m):
            # searchword = "mouse"
            #               i
            c = searchWord[i]
            while l <= r and (i >= len(products[l]) or products[l][i] != c):
                l += 1
            while l <= r and (i >= len(products[r]) or products[r][i] != c):
                r -= 1

            tempList = []
            for j in range(l, r + 1): # l to r inclusive
                if len(tempList) == 3:
                    break
                tempList.append(products[j])
                
            res.append(tempList)
            
        return res
```
### Group Anagrams
https://leetcode.com/problems/group-anagrams/
```python
'''

'''
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashMap = defaultdict(list)
        for word in strs:
            key = [0] * 26
            for c in word:
                key[ord(c) - ord('a')] += 1

            immutableKey = tuple(key)
            hashMap[immutableKey].append(word)
        
        return list(hashMap.values())
        
```
### Integer to Roman | Add exceptions to the input hashMap itself
```python
'''

'''
class Solution:
    def intToRoman(self, num: int) -> str:
        romanList = [['I', 1], ['IV', 4], ['V', 5], ['IX', 9], ['X', 10],
                    ['XL', 40], ['L', 50], ['XC', 90], ['C', 100], 
                    ['CD', 400], ['D', 500], ['CM', 900], ['M', 1000]]
        
        res = ""
        for i in range(len(romanList) - 1, -1, -1):
            sym, val = romanList[i]
            if num // val:
                count = num // val
                res += (sym * count)
                num = num % val
        return res
```
### Smallest Subtree with all the deepest nodes | Simple postorder traversal + every node returns 2 values (height, probableAns)
https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/
```python
'''

'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        
        def f(node):
            if node == None:
                return [0, None]
            
            leftH, ansFromLeft = f(node.left)
            rightH, ansFromRight = f(node.right)
            
            if leftH == rightH:
                return [1 + leftH, node]
            else:
                if leftH > rightH:
                    return [1 + leftH, ansFromLeft]
                else:
                    return [1 + rightH, ansFromRight]
            
        return f(root)[1]
```
### Range Addition
https://leetcode.com/problems/range-addition/
```python
'''

'''
class Solution:

    # def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
    #     arr = [0] * length
    #     for i, j, inc in updates:
    #         for k in range(i, j + 1):
    #             arr[k] += inc
    #     return arr

    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        preSum = [0] * (length + 1)

        for i, j, inc in updates:
            preSum[i] += inc
            preSum[j + 1] += inc * (-1)
        
        # print(preSum)
        for i in range(1, len(preSum)):
            preSum[i] = preSum[i - 1] + preSum[i]
    
        return preSum[:-1]

```
### Beautiful Arrangement
```python
'''

'''
class Solution:
    def countArrangement(self, n: int) -> int:

        def swap(i, j):
            nums[i], nums[j] = nums[j], nums[i]
        
        def permute(ind):
            nonlocal count
            if ind == n + 1:
                count += 1        
            for i in range(ind, n + 1):
                swap(ind, i)
                # only permute(go forward) if condition is not broken
                # prunes the search space by a lot !
                if nums[ind] % ind == 0 or ind % nums[ind] == 0:
                    permute(ind + 1)
                swap(ind, i)

        nums = [0] * (n + 1)
        for i in range(1, n + 1):
            nums[i] = i

        count = 0

        permute(1)

        return count
```
### Maximum difference between increasing elements | easy | Stock - I
```python
'''

'''
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        n = len(nums)
        minOnLeft = nums[0]
        maxi = -1
        for i in range(1, n):
            if nums[i] - minOnLeft > 0:
                maxi = max(maxi, nums[i] - minOnLeft)
            minOnLeft = min(minOnLeft, nums[i])
        return maxi
```
### Predict the winner | minimax game theory | same as stone game
https://leetcode.com/problems/predict-the-winner/description/
```python
'''

'''
class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        N = len(nums)

        dp = [[-1] * (N + 1) for _ in range(N + 1)]

        def f(left, right):
            if left > right:
                return 0

            if dp[left][right] != -1:
                return dp[left][right]
                
            # minmax game theory
            scoreFront = nums[left] - f(left + 1, right)
            scoreBack = nums[right] - f(left, right - 1)
            
            dp[left][right] = max(scoreFront, scoreBack)
            return dp[left][right]
        
        return f(0, N - 1) >= 0

```
### Degree of an array | Sliding Window
```python
'''

'''
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        freqMap = Counter(nums)
        degree = max(freqMap.values())
        mini = float("inf")
        l, r = 0, 0
        currentMap = defaultdict(int)
        while r < n:
            currentMap[nums[r]] += 1
            while currentMap[nums[r]] == degree:
                mini = min(mini, r - l + 1)
                currentMap[nums[l]] -= 1
                l += 1
            r += 1
        return mini
```
### Graph Valid Tree | Cycle Detection in undirected Graph | If no cycle -> valid tree
```python
'''

'''
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        adjList = {i: [] for i in range(n)}

        for a, b in edges:
            adjList[a].append(b)
            adjList[b].append(a)

        visited = set()

        def dfs(node, parent):
            visited.add(node)

            for neighbor in adjList[node]:
                if neighbor in visited:
                    if neighbor != parent:
                        return True
                else:
                    if dfs(neighbor, node) == True:
                        return True
            
            return False

        print(visited)
        return not dfs(0, -1) and len(visited) == n

```
### Subarray Product Less Than K | Sliding Window
```python
'''

'''
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        cnt = 0
        prod = 1
        l, r = 0, 0
        n = len(nums)
        while r < n:
            prod = prod * nums[r]
            while l <= r and prod >= k:
                prod = prod // nums[l]
                l += 1
            cnt += (r - l + 1)
            r += 1
        return cnt
```
### Merge In Between Linked Lists | Simple Linked List | prev A, A, B, next B
https://leetcode.com/problems/merge-in-between-linked-lists/
```python
'''

'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        '''
        1. get prevA and A and B and nextB
        2. prevA.next = list2
        3. traverse list2 till last node
        4. last node.next = nextB
        '''
        
        p = list1
        for _ in range(a - 1): # go only till position a - 1
            p = p.next
        
        q = p.next
        for _ in range(b - a + 1): # go 1 pos beyond b
            q = q.next
        
        p.next = list2

        cur = list2
        while list2.next:
            list2 = list2.next
        
        list2.next = q
    
        return list1
        


```
### Reconstruct Original Digits from English
https://leetcode.com/problems/reconstruct-original-digits-from-english/description/
```python
'''

'''
class Solution:
    # def originalDigits(self, s: str) -> str:
    #     originalMap = defaultdict(int)
    #     for c in s:
    #         originalMap[c] += 1

    #     digitToStr = list(pair.split(' ') for pair in "0 zero,2 two,4 four,6 six,8 eight,1 one,3 three,5 five,7 seven,9 nine".split(','))
    #     # print(digitToStr)
    
    #     i = 0
    #     ans = []
    #     while i < len(digitToStr):
    #         num, chars = digitToStr[i]
    #         digitMap = defaultdict(int)
    #         for c in chars:
    #             digitMap[c] += 1
    #         isContained = True    
    #         for key in digitMap:
    #             if key not in originalMap or originalMap[key] < digitMap[key]:
    #                 isContained = False
    #                 i += 1
    #                 break
    #         # print(isContained)
    #         if isContained:
    #             for key in digitMap:
    #                 originalMap[key] -= digitMap[key]
    #             ans.append(int(num))
    #     ans = [str(a) for a in sorted(ans)]

    #     return  "".join(ans)

    def originalDigits(self, s: str) -> str:
        num = [0] * 10
        freqMap = collections.Counter(s)
        
        num[0] = freqMap['z']
        num[2] = freqMap['w']
        num[6] = freqMap['x']
        num[8] = freqMap['g']
        num[4] = freqMap['u']

        num[1] = freqMap['o'] - num[0] - num[2] - num[4]
        num[3] = freqMap['h'] - num[8]
        num[5] = freqMap['f'] - num[4]
        num[7] = freqMap['s'] - num[6]

        num[9] = freqMap['i'] - num[5] - num[6] - num[8]

        res = ""
        for i in range(10):
            res += (str(i) * num[i])
        return res
```
### 
https://leetcode.com/problems/number-of-matching-subsequences/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/different-ways-to-add-parentheses/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/largest-number/
```python
'''

'''

```
### 
https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/max-area-of-island/
```python
'''

'''

```
### Populating Next Right Pointers in Each Node
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/
```python
'''
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 

Example 1:


Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 212 - 1].
-1000 <= Node.val <= 1000
 

Follow-up:

You may only use constant extra space.
The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.
'''
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    # # # method 1: DFS (O(n) auxillary stack space!)
    # def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':

    #     def dfs(node):
    #         if node == None:
    #             return
            
    #         if node.left:
    #             node.left.next = node.right
    #         if node.right and node.next:
    #             node.right.next = node.next.left
    #         dfs(node.left)
    #         dfs(node.right)

    #     dfs(root)
    #     return root


    # # method 2: BFS (O(n) space!)
    # def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
    #     q = deque()
    #     q.append(root)

    #     while q:
    #         n = len(q)
    #         if n > 0:
    #             node1 = q.popleft()
    #             if node1:
    #                 q.append(node1.left)
    #                 q.append(node1.right)
    #                 for _ in range(1, n):
    #                     node2 = q.popleft()
    #                     if node2:
    #                         q.append(node2.left)
    #                         q.append(node2.right)

    #                         node1.next = node2
    #                         node1 = node2
    #     return root


    # method 3: BFS logic only (O(1) space!)
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        cur, nxt = root, root.left if root else None

        while cur and nxt:
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
            
            cur = cur.next

            if cur == None:
                cur = nxt
                nxt = cur.left
        return root
```
### 
https://leetcode.com/problems/asteroid-collision/
```python
'''

'''

```
### 
https://leetcode.com/problems/kth-largest-element-in-an-array/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/majority-element-ii/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
```python
'''

'''

```
### 
https://leetcode.com/problems/combination-sum/description/
```python
'''

'''

```
### Trapping Rain water | keep track of max at left and max at right -> min of that is the bottleneck
https://leetcode.com/problems/trapping-rain-water/
```python
'''

'''
class Solution:
    # def trap(self, height: List[int]) -> int:
    #     n = len(height)
    #     maxLeft, maxRight = [0] * n, [0] * n
    #     # populate the maxLeft and maxRight lists
    #     for i in range(n):
    #         j = n - 1 - i
    #         if i == 0:
    #             maxLeft[i] = height[i]
    #         else:
    #             maxLeft[i] = max(maxLeft[i - 1], height[i])
    #         if j == n - 1:
    #             maxRight[j] = height[j]
    #         else:
    #             maxRight[j] = max(maxRight[j + 1], height[j])
             
    #     # calculate the amount of water now!
    #     water = 0
    #     for i in range(n):
    #         minLeftRight = min(maxLeft[i], maxRight[i])
    #         if height[i] < minLeftRight:
    #             water += minLeftRight - height[i]
    #     return water

    def trap(self, height: List[int]) -> int:
        n = len(height)
        maxLeft, maxRight = height[0], height[n - 1]
        l, r = 1, n - 2
        water = 0
        while l <= r:
            if maxLeft < maxRight:
                if maxLeft > height[l]:
                    water += maxLeft - height[l]
                maxLeft = max(maxLeft, height[l])
                l += 1
            else:
                if maxRight > height[r]:
                    water += maxRight - height[r]
                maxRight = max(maxRight, height[r])
                r -= 1
        return water

```
### Triangle | DP Grid | easy
https://leetcode.com/problems/triangle/description/
```python
'''

'''
class Solution:
    # def minimumTotal(self, triangle: List[List[int]]) -> int:
    #     n = len(triangle)
    #     def f(i, j):
    #         if i == n - 1:
    #             return triangle[i][j]
    #         down = triangle[i][j] + f(i + 1, j)
    #         downDiagRight = triangle[i][j] + f(i + 1, j + 1)
    #         return min(down, downDiagRight)
    #     return f(0, 0)
    
    # def minimumTotal(self, triangle: List[List[int]]) -> int:
    #     n = len(triangle)
    #     def f(i, j):
    #         if i == n - 1:
    #             return triangle[i][j]
    #         if dp[i][j] != -1:
    #             return dp[i][j]
    #         down = triangle[i][j] + f(i + 1, j)
    #         downDiagRight = triangle[i][j] + f(i + 1, j + 1)
    #         dp[i][j] = min(down, downDiagRight)
    #         return dp[i][j]
    #     dp = [[-1] * (n) for _ in range(n)]
    #     return f(0, 0)
        

    

```
### Decode Ways | DP on 
https://leetcode.com/problems/decode-ways/description/
```python
'''

'''
class Solution:
    # def numDecodings(self, s: str) -> int:
    #     '''
    #     11106
    #     f(11106) -> 1 + f(1106) or 11 + f(106)
    #     f(1106) -> 1 + f(106) or 11 + f(06), 
        
    #     0th idx can be 1 ->  11, 12, 13.. 19
    #     0th idx can be 2 -> 20, 21, 22, 23, 24, 25, 26 -> only if 1th idx is in "0123456"
    #     f(106) -> 1 + f(06) or 10 + f(6)
    #     f(6) -> 1
    #     '''

    #     n = len(s)
        
    #     def f(i):
    #         # base cases
    #         if i == n:
    #             return 1
    #         if s[i] == '0':
    #             return 0

    #         # gen case
    #         takeOne = f(i + 1)
    #         takeTwo = 0
    #         if i + 1 < n:
    #             if s[i] == '1' or s[i] == '2' and s[i + 1] in "0123456":
    #                 takeTwo = f(i + 2)
        
    #         return takeOne + takeTwo

    #     return f(0)

    def numDecodings(self, s: str) -> int:
        '''
        11106
        f(11106) -> 1 + f(1106) or 11 + f(106)
        f(1106) -> 1 + f(106) or 11 + f(06), 
        
        0th idx can be 1 ->  11, 12, 13.. 19
        0th idx can be 2 -> 20, 21, 22, 23, 24, 25, 26 -> only if 1th idx is in "0123456"
        f(106) -> 1 + f(06) or 10 + f(6)
        f(6) -> 1
        '''

        n = len(s)
        
        def f(i):
            # base cases
            if i == n:
                return 1
            if s[i] == '0':
                return 0
            if dp[i] != -1:
                return dp[i]
            # gen case
            takeOne = f(i + 1)
            takeTwo = 0
            if i + 1 < n:
                if s[i] == '1' or s[i] == '2' and s[i + 1] in "0123456":
                    takeTwo = f(i + 2)
        
            dp[i] = takeOne + takeTwo
            return dp[i]

        dp = [-1] * n
        return f(0)
```
### Next Greater Element I | Stack + HashMap
https://leetcode.com/problems/next-greater-element-i/
```python
'''

The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

 

Example 1:

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
Example 2:

Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
 

Constraints:

1 <= nums1.length <= nums2.length <= 1000
0 <= nums1[i], nums2[i] <= 104
All integers in nums1 and nums2 are unique.
All the integers of nums1 also appear in nums2.
 

Follow up: Could you find an O(nums1.length + nums2.length) solution?
'''
# Time: O(n * m)
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        n, m = len(nums1), len(nums2)
        nextGreater = [-1] * m
        for i in range(m):
            while stack and nums2[i] > nums2[stack[-1]]:
                idx  = stack.pop()
                nextGreater[idx] = nums2[i]
            stack.append(i)

        res = [-1] * n
        for i in range(n):
            idx = nums2.index(nums1[i])
            res[i] = nextGreater[idx]
        return res

# time: O(n + m)
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        n, m = len(nums1), len(nums2)
        nextGreater = [-1] * m
        idxTracker = {}
        for i in range(m):
            while stack and nums2[i] > nums2[stack[-1]]:
                idx  = stack.pop()
                nextGreater[idx] = nums2[i]
            stack.append(i)
            idxTracker[nums2[i]] = i

        res = [-1] * n
        for i in range(n):
            idx = idxTracker[nums1[i]]
            res[i] = nextGreater[idx]
        return res
```
### Next Greater Element II | Stack + run loop for 2n times
https://leetcode.com/problems/next-greater-element-ii/description/
```python
'''

'''
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        n = len(nums)
        res = [-1] * n
        for i in range(2 * n):
            i = i % n
            while stack and nums[i] > nums[stack[-1]]:
                idx = stack.pop()
                res[idx] = nums[i]
            stack.append(i)
        return res 
```
### Next Greater Element III | Brute: Generate all perms and keep checking the minimum greater than n | Optimal: Next Permutation pattern
```python
'''

'''
class Solution:
    # def nextGreaterElement(self, n: int) -> int:
    #     # collect digits in a list
    #     num = n
    #     nums = []
    #     while num:
    #         digit = num % 10
    #         nums = [str(digit)] + nums
    #         num = num // 10
    #     # print(nums)
    #     # get all permutations of this list
    #     ans = float("inf")
    #     m = len(nums)
    #     # for each permutation, check if its greater than n and smaller than prev ans
    #     def permute(i):
    #         nonlocal ans
    #         if i == m:
    #             res = int("".join(nums))
    #             # print(res, n, ans)
    #             if res > n and res < ans:
    #                 ans = res
    #                 # print("ans ", ans)
    #                 return
            
    #         for j in range(i, m):
    #             # print(i, j)
    #             nums[i], nums[j] = nums[j], nums[i]
    #             permute(i + 1)
    #             nums[i], nums[j] = nums[j], nums[i]
    #     # print(nums)
    #     permute(0)
    #     return ans if ans != float("inf") and ans <= (2**31 - 1) else -1
    
    def nextGreaterElement(self, n: int) -> int:
        # collect the digits in a list
        num = n
        nums = []
        while num:
            digit = num % 10
            nums = [str(digit)] + nums
            num = num // 10
        m = len(nums)
        
        # scan from right until we find condition a[i - 1] < a[i]
        # (elements to a[i - 1]'s left don't matter)
        # (elements to the right of i - 1 are in decending order for sure)
        i = m - 1
        k = -1
        while i >= 1:
            if nums[i - 1] < nums[i]:
                k = i - 1
                break
            i -= 1
        if k == -1:
            return -1 # not possible
        
        # to get the smallest next greater, swap a[i - 1] with a[j], where a[j] is next greater of a[i - 1]
        # find next greater element of "k" th idx element
        j = m - 1
        while j >= 0 and nums[j] <= nums[k]:
            j -= 1
        nums[k], nums[j] = nums[j], nums[k]
        
        # reverse from ith idx till the end (as previously it was in decending 
        # -> reversing gives smallest possible arrangement)
        def reverse(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        
        reverse(k + 1, m - 1)
        
        ans = int("".join([str(n) for n in nums]))
        return ans if ans <= (2**31 - 1) else -1
```
