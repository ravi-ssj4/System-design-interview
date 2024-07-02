### 01. Logger Rate Limiter
```python
class Logger:

    def __init__(self):
        self.hashMap = {}

    def shouldPrintMessage(self, timestamp: int, msg: str) -> bool:
        if msg not in self.hashMap:
            self.hashMap[msg] = timestamp + 10
            return True
        else:
            if timestamp < self.hashMap[msg]:
                return False
            else:
                self.hashMap[msg] = timestamp + 10
                return True


# Your Logger object will be instantiated and called as such:
# obj = Logger()
# param_1 = obj.shouldPrintMessage(timestamp,message)
```
### 02. Closest Binary Search Tree 
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        cur = root
        if cur == None:
            return -1
        ans = [float("inf"), None]
        # jab hum left ja rahe hain aur difference equal aa raha hai prev
        # kisi answer ke, to hum left jayenge but current node ko consider nahi karenge
        # kyuki left me aur choti value wali node milegi
        # par jab right jayenge, to pehle choti value milegi aur baad me saari badi values
        # milengi, so, need to capture in this case the current node's value
        while cur:
            if target < cur.val:
                if abs(cur.val - target) < ans[0]:
                    ans[0] = abs(cur.val - target) 
                    ans[1] = cur.val
                cur = cur.left
            else:
                if abs(cur.val - target) <= ans[0]:
                    ans[0] = abs(cur.val - target) 
                    ans[1] = cur.val
                cur = cur.right
        return ans[1]
```
### 03. Merge Two Sorted lists
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    # def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    #     if list1 == None:
    #         return list2
    #     if list2 == None:
    #         return list1
    #     if list1.val <= list2.val:
    #         head = list1
    #     else:
    #         head = list2
    #     p, q = list1, list2
    #     while p and q:
    #         if p.val <= q.val:
    #             nextP = p.next
    #             p.next = q
    #             nextQ = q.next
    #             q.next = nextP
    #             q = nextQ
    #         else:
    #             nextP = p.next
    #             while q.val < nextP.val:
    #                 q = q.next
    #             nextQ = q.next
    #             q.next = nextP
        
    #     if p:
    #         if q:
    #             q.next = p
    #     if q:
    #         if p:
    #             p.next = q
        
    #     return head

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = ListNode()
        dummy = cur
        while list1 and list2:
            if list1.val <= list2.val:
                dummy.next = list1
                list1 = list1.next
            else:
                dummy.next = list2
                list2 = list2.next
            
            dummy = dummy.next
        
        if list1:
            dummy.next = list1
        if list2:
            dummy.next = list2
        
        return cur.next

```
### 04. Implement Queue using Stacks 
```python
class MyQueue:

    def __init__(self):
        self.tempStack = []
        self.mainStack = []

    def push(self, x: int) -> None:
        if not self.mainStack:
            self.mainStack.append(x)
        else:
            while self.mainStack:
                self.tempStack.append(self.mainStack.pop())
            self.mainStack.append(x)
            while self.tempStack:
                self.mainStack.append(self.tempStack.pop())

    def pop(self) -> int:
        return self.mainStack.pop()

    def peek(self) -> int:
        return self.mainStack[-1]

    def empty(self) -> bool:
        return len(self.mainStack) == 0
```
### 05. Isomorphic Strings
```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mappingST = {}
        mappingTS = {}
        n = len(s)
        for i in range(n):
            c1 = s[i]
            c2 = t[i]
            if c1 in mappingST and mappingST[c1] != c2:
                return False
            if c2 in mappingTS and mappingTS[c2] != c1:
                return False
            mappingST[c1] = c2
            mappingTS[c2] = c1
        return True
```
### 06. Roman To Integer
```python
class Solution:
    def romanToInt(self, s: str) -> int:
        romanToInt = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        n = len(s)
        Sum = 0
        for i in range(n):
            if i + 1 < n and romanToInt[s[i]] < romanToInt[s[i + 1]]:
                Sum -= romanToInt[s[i]]
            else:
                Sum += romanToInt[s[i]]
        return Sum
```
### 07. Design Authentication Manager | Medium 
```python
class AuthenticationManager:

    def __init__(self, timeToLive: int):
        self.tokenMap = {}
        self.timeToLive = timeToLive
        

    def generate(self, tokenId: str, currentTime: int) -> None:
        self.tokenMap[tokenId] = currentTime + self.timeToLive

    def renew(self, tokenId: str, currentTime: int) -> None:
        if tokenId in self.tokenMap and currentTime < self.tokenMap[tokenId]:
            self.tokenMap[tokenId] = currentTime + self.timeToLive

    def countUnexpiredTokens(self, currentTime: int) -> int:
        count = 0
        for tokenId, expTime in self.tokenMap.items():
            if currentTime < expTime:
                count += 1
        return count
        
# Your AuthenticationManager object will be instantiated and called as such:
# obj = AuthenticationManager(timeToLive)
# obj.generate(tokenId,currentTime)
# obj.renew(tokenId,currentTime)
# param_3 = obj.countUnexpiredTokens(currentTime)
```
### 08. Word Pattern 1 | Isomorphic String logic
```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        mappingPS = {}
        mappingSP = {}
        s = s.split()
        if len(pattern) != len(s):
            return False
        n = len(pattern)
        for i in range(n):
            char = pattern[i]
            word = s[i]
            if char in mappingPS and mappingPS[char] != word:
                return False
            if word in mappingSP and mappingSP[word] != char:
                return False
            mappingPS[char] = word
            mappingSP[word] = char
        return True


```
### 09. Word Pattern 2 | Backtracking
```python

```
### 10. Pairs of Songs With Total Durations Divisible by 60 | Lookup in hashMap logic of two sum
```python
class Solution:
    # def numPairsDivisibleBy60(self, time: List[int]) -> int:
    #     n = len(time)
    #     count = 0
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             if (time[i] + time[j]) % 60 == 0:
    #                 count += 1
    #     return count

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        # (a + b) % 60 == 0
        # ((a % 60) + (b % 60)) % 60 == 0
        # either a % 60 == 0 and b % 60 == 0
        # or (a % 60) + (b % 60) == 60
        # 1st condition -> if a % 60 == 0, then b % 60 == 0 (has to be)
        # 2nd condition -> if a % 60 != 0, then b % 60 = 60 - (a % 60)
        # and hence, we store b % 60 in the map, to check dynamically
        n = len(time)
        count = 0
        hashMap = defaultdict(list) # key: b % 60, value: list of numbers of remainder = b % 60
        for a in time:
            if a % 60 == 0: # b % 60 == 0, should be the other pair
                if 0 in hashMap:
                    count += len(hashMap[0])
            else:
                other = 60 - (a % 60)
                if other in hashMap:
                    count += len(hashMap[other])
            hashMap[a % 60].append(a)
        return count
```
### 11. Break a Palindrome
```python
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        n = len(palindrome)

        if n == 1:
            return ""
        
        res = "z" * n
        
        def getSmallest(word1, word2):
            # print(word1, word2)
            for i in range(n):
                c1, c2 = word1[i], word2[i]
                # print(c1, c2)
                if c1 > c2:
                    return word2
                elif c1 < c2:
                    return word1
                else:
                    continue
        i = 0
        while i < n:
            if n % 2 and i == n // 2:
                i += 1
                continue
            if palindrome[i] == 'a':
                word = palindrome[:i] + 'b' + palindrome[i + 1:]
            else:
                word = palindrome[:i] + 'a' + palindrome[i + 1:]
            res = getSmallest(word, res)
            
            i += 1
        
        return res

    def breakPalindrome(self, palindrome: str) -> str:
        n = len(palindrome)

        if n == 1:
            return ""
        
        for i in range(n):
            if i == n - 1 - i: # ie. the middle element
                continue

            if palindrome[i] != 'a':
                return palindrome[:i] + 'a' + palindrome[i + 1:]
            
        return palindrome[:-1] + 'b'
        
            


```
### 12. 
```python

```
### 13. 
```python

```
### 14. 
```python

```
### 15. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```
### 01. 
```python

```

