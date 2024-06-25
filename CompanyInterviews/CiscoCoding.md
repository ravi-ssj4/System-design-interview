### Gas Station
```python
'''

'''
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        n = len(gas)
        currentGas = 0
        start = 0
        for i in range(n):
            currentGas += (gas[i] - cost[i])
            if currentGas < 0:
                currentGas = 0
                start = i + 1
        return start
```
### Rotate Matrix
```python
'''

'''
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # transpose and then reverse row by row
        ROWS, COLS = len(matrix), len(matrix[0])

        # transpose
        for r in range(ROWS):
            for c in range(r + 1, COLS):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        
        for r in range(ROWS):
            left, right = 0, COLS - 1
            while left < right:
                matrix[r][left], matrix[r][right] = matrix[r][right], matrix[r][left]
                left += 1
                right -= 1
        
```
### Spiral Traverse
```python
'''

'''
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ROWS, COLS = len(matrix), len(matrix[0])

        top, bottom, left, right = 0, ROWS - 1, 0, COLS - 1
        res = []
        while left <= right and top <= bottom:

            # traverse first row
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1
            # traverse rightmost col
            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1

            # traverse bottom row in reverse
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    res.append(matrix[bottom][i])
                
                bottom -= 1

            # traverse first col in reverse
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                
                left += 1
        return res
```
### Predict the winner
```python
'''
You are given an integer array nums. Two players are playing a game with this array: player 1 and player 2.

Player 1 and player 2 take turns, with player 1 starting first. Both players start the game with a score of 0. At each turn, the player takes one of the numbers from either end of the array (i.e., nums[0] or nums[nums.length - 1]) which reduces the size of the array by 1. The player adds the chosen number to their score. The game ends when there are no more elements in the array.

Return true if Player 1 can win the game. If the scores of both players are equal, then player 1 is still the winner, and you should also return true. You may assume that both players are playing optimally.

 

Example 1:

Input: nums = [1,5,2]
Output: false
Explanation: Initially, player 1 can choose between 1 and 2. 
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2). 
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5. 
Hence, player 1 will never be the winner and you need to return false.
Example 2:

Input: nums = [1,5,233,7]
Output: true
Explanation: Player 1 first chooses 1. Then player 2 has to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.
Finally, player 1 has more score (234) than player 2 (12), so you need to return True representing player1 can win.
 

Constraints:

1 <= nums.length <= 20
0 <= nums[i] <= 107
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
### Lucky Numbers in a matrix
https://leetcode.com/problems/lucky-numbers-in-a-matrix/
```python
'''
Given an m x n matrix of distinct numbers, return all lucky numbers in the matrix in any order.

A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.

 

Example 1:

Input: matrix = [[3,7,8],[9,11,13],[15,16,17]]
Output: [15]
Explanation: 15 is the only lucky number since it is the minimum in its row and the maximum in its column.
Example 2:

Input: matrix = [[1,10,4,2],[9,3,8,7],[15,16,17,12]]
Output: [12]
Explanation: 12 is the only lucky number since it is the minimum in its row and the maximum in its column.
Example 3:

Input: matrix = [[7,8],[1,2]]
Output: [7]
Explanation: 7 is the only lucky number since it is the minimum in its row and the maximum in its column.
 

Constraints:

m == mat.length
n == mat[i].length
1 <= n, m <= 50
1 <= matrix[i][j] <= 105.
All elements in the matrix are distinct.
'''
class Solution:
    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        ROWS, COLS = len(matrix), len(matrix[0])
        hashMap = {} # key: min/max in row/col respectively, value: num of times this key appeared
        # for an element to be both min in a row and max in a col, it must appear 2 times in the hashMap

        for r in range(ROWS):
            mini = float("inf")
            for c in range(COLS):
                mini = min(mini, matrix[r][c])
            hashMap[mini] = 1 + hashMap.get(mini, 0)
    
        for c in range(COLS):
            maxi = float("-inf")
            for r in range(ROWS):
                maxi = max(maxi, matrix[r][c])
            hashMap[maxi] = 1 + hashMap.get(maxi, 0)
        
        res = []
        for key in hashMap:
            if hashMap[key] == 2:
                res.append(key)
        return res
        

        
```
### Strange Printer I || Matrix Chain Multiplication
https://leetcode.com/problems/strange-printer/
```python
'''
There is a strange printer with the following two special properties:

The printer can only print a sequence of the same character each time.
At each turn, the printer can print new characters starting from and ending at any place and will cover the original existing characters.
Given a string s, return the minimum number of turns the printer needed to print it.

 

Example 1:

Input: s = "aaabbb"
Output: 2
Explanation: Print "aaa" first and then print "bbb".
Example 2:

Input: s = "aba"
Output: 2
Explanation: Print "aaa" first and then print "b" from the second place of the string, which will cover the existing character 'a'.
 

Constraints:

1 <= s.length <= 100
s consists of lowercase English letters.
'''
class Solution:
    def strangePrinter(self, s: str) -> int:
        
        n = len(s)

        def f(i, j): # i...j is the part of the string under consideration
            # base case
            if i == j:
                return 1

            if dp[i][j] != -1:
                return dp[i][j]

            # gen case
            minTurns = float("inf")
            for k in range(i, j):
                minTurns = min(minTurns, f(i, k) + f(k + 1, j))

            # edge case
            if s[i] == s[j]:
                minTurns -= 1
            
            dp[i][j] = minTurns
            return dp[i][j]
        
        dp = [[-1] * (n) for _ in range(n)]
        return f(0, n - 1)
```
### Stone Game I
https://leetcode.com/problems/stone-game/description/
```python
'''
Alice and Bob play a game with piles of stones. There are an even number of piles arranged in a row, and each pile has a positive integer number of stones piles[i].

The objective of the game is to end with the most stones. The total number of stones across all the piles is odd, so there are no ties.

Alice and Bob take turns, with Alice starting first. Each turn, a player takes the entire pile of stones either from the beginning or from the end of the row. This continues until there are no more piles left, at which point the person with the most stones wins.

Assuming Alice and Bob play optimally, return true if Alice wins the game, or false if Bob wins.

 

Example 1:

Input: piles = [5,3,4,5]
Output: true
Explanation: 
Alice starts first, and can only take the first 5 or the last 5.
Say she takes the first 5, so that the row becomes [3, 4, 5].
If Bob takes 3, then the board is [4, 5], and Alice takes 5 to win with 10 points.
If Bob takes the last 5, then the board is [3, 4], and Alice takes 4 to win with 9 points.
This demonstrated that taking the first 5 was a winning move for Alice, so we return true.
Example 2:

Input: piles = [3,7,2,3]
Output: true
 

Constraints:

2 <= piles.length <= 500
piles.length is even.
1 <= piles[i] <= 500
sum(piles[i]) is odd.
'''

```
### Maximum Difference Between Increasing Elements | EASY
```python
'''
Given a 0-indexed integer array nums of size n, find the maximum difference between nums[i] and nums[j] (i.e., nums[j] - nums[i]), such that 0 <= i < j < n and nums[i] < nums[j].

Return the maximum difference. If no such i and j exists, return -1.

 

Example 1:

Input: nums = [7,1,5,4]
Output: 4
Explanation:
The maximum difference occurs with i = 1 and j = 2, nums[j] - nums[i] = 5 - 1 = 4.
Note that with i = 1 and j = 0, the difference nums[j] - nums[i] = 7 - 1 = 6, but i > j, so it is not valid.
Example 2:

Input: nums = [9,4,3,2]
Output: -1
Explanation:
There is no i and j such that i < j and nums[i] < nums[j].
Example 3:

Input: nums = [1,5,2,10]
Output: 9
Explanation:
The maximum difference occurs with i = 0 and j = 3, nums[j] - nums[i] = 10 - 1 = 9.
 

Constraints:

n == nums.length
2 <= n <= 1000
1 <= nums[i] <= 109
'''

```
### LRU Cache
https://leetcode.com/problems/lru-cache/description/
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

```
### Roman to Integer
https://leetcode.com/problems/roman-to-integer/description/
```python
'''
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].
'''
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
        res = 0
        for i in range(n):
            if i + 1 < n and romanToInt[s[i]] < romanToInt[s[i + 1]]:
                res -= romanToInt[s[i]]
            else:
                res += romanToInt[s[i]]
        return res
```
### Design Browser History
```python
'''
You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:

BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
void visit(string url) Visits url from the current page. It clears up all the forward history.
string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.
 

Example:

Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"
 

Constraints:

1 <= homepage.length <= 20
1 <= url.length <= 20
1 <= steps <= 100
homepage and url consist of  '.' or lower case English letters.
At most 5000 calls will be made to visit, back, and forward.
'''
class ListNode:
    def __init__(self, data="", prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

class BrowserHistory:

    def __init__(self, homepage: str):
        self.current = ListNode(homepage)

    def visit(self, url: str) -> None:
        newNode = ListNode(url)
        self.current.next = newNode
        newNode.prev = self.current
        self.current = newNode

    def back(self, steps: int) -> str:
        while steps:
            if self.current.prev:
                self.current = self.current.prev
            else:
                break
            steps -= 1
        return self.current.data

    def forward(self, steps: int) -> str:
        while steps:
            if self.current.next:
                self.current = self.current.next
            else:
                break
            steps -= 1
        return self.current.data

# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```
### Merge K sorted Lists
https://leetcode.com/problems/merge-k-sorted-lists/
```python
'''
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:

Input: lists = []
Output: []
Example 3:

Input: lists = [[]]
Output: []
 

Constraints:

k == lists.length
0 <= k <= 104
0 <= lists[i].length <= 500
-104 <= lists[i][j] <= 104
lists[i] is sorted in ascending order.
The sum of lists[i].length will not exceed 104.
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        def merge2SortedLists(l1, l2):
            dummy = ListNode()
            tail = dummy
            while l1 and l2:
                if l1.val < l2.val:
                    tail.next = ListNode(l1.val)
                    l1 = l1.next
                else:
                    tail.next = ListNode(l2.val)
                    l2 = l2.next
                tail = tail.next
            if l1:
                tail.next = l1
            if l2:
                tail.next = l2

            return dummy.next

        while len(lists) > 1:
            mergedLists = []
            for i in range(0,len(lists), 2):
                list1 = lists[i]
                list2 = lists[i + 1] if (i + 1) < len(lists) else None
            
                mergedLists.append(merge2SortedLists(list1, list2))
        
            lists = mergedLists.copy()
        
        return lists[0] if len(lists) == 1 else None
```
### Longest Palindromic Substring | Without DP | Outwards expansion check
https://leetcode.com/problems/longest-palindromic-substring/description/
```python
'''
Given a string s, return the longest 
palindromic
 
substring
 in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
 

Constraints:

1 <= s.length <= 1000
s consist of only digits and English letters.
'''
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def isPalin(l, r):
            length = 0
            resStr = ""
            # expanding outwards
            while l >= 0 and r < len(s) and s[l] == s[r]:
                length = r - l + 1
                resStr = s[l: r + 1]
                l -= 1
                r += 1
            return (length, resStr)
        
        longest = 0
        res = ""
        for i in range(len(s)):
            oddLength, oddStr = isPalin(i, i)
            if oddLength > longest:
                longest = oddLength
                res = oddStr
                
            evenLength, evenStr = isPalin(i, i + 1)
            if evenLength > longest:
                longest = evenLength
                res = evenStr
        
        return res
```
### Maximum product of 2 elements in an array | easy
https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array/description/
```python
'''
Given the array of integers nums, you will choose two different indices i and j of that array. Return the maximum value of (nums[i]-1)*(nums[j]-1).
 

Example 1:

Input: nums = [3,4,5,2]
Output: 12 
Explanation: If you choose the indices i=1 and j=2 (indexed from 0), you will get the maximum value, that is, (nums[1]-1)*(nums[2]-1) = (4-1)*(5-1) = 3*4 = 12. 
Example 2:

Input: nums = [1,5,4,5]
Output: 16
Explanation: Choosing the indices i=1 and j=3 (indexed from 0), you will get the maximum value of (5-1)*(5-1) = 16.
Example 3:

Input: nums = [3,7]
Output: 12
 

Constraints:

2 <= nums.length <= 500
1 <= nums[i] <= 10^3
'''
class Solution:
    def maxProduct(self, nums: List[int]) -> int:

        n = len(nums)
        largest = float("-inf")
        firstIdx, secondIdx = -1, -1
        secondLargest = float("-inf")
        for i in range(n):
            if nums[i] > largest:
                secondLargest = largest
                secondIdx = firstIdx
                largest = nums[i]
                firstIdx = i
            elif nums[i] > secondLargest:
                secondLargest = nums[i]
                secondIdx = i
        return (nums[firstIdx] - 1) * (nums[secondIdx] - 1)
```
### Happy Number
```python
'''

'''
class Solution:
    def isHappy(self, n: int) -> bool:
        def giveSumOfSquaresOfDigits(n):
            summation = 0
            while n > 0:
                digit = n % 10
                n = n // 10
                summation += (digit ** 2)
            return summation

        tracker = set()
        while n:
            if n in tracker:
                    return False
            else:
                tracker.add(n)
            n = giveSumOfSquaresOfDigits(n)
            if n == 1:
                return True
        return False
```
### Fizz Buzz
```python
'''

'''
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        ans = []
        for num in range(1, n + 1):
            substr = ""
            if num % 3 == 0:
                substr += "Fizz"
            
            if num % 5 == 0:
                substr += "Buzz"
            
            if len(substr) == 0:
                substr += str(num)

            ans.append(substr)
        return ans
```
### Maximum difference between increasing elements | easy
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
### Decode String
https://leetcode.com/problems/decode-string/
```python
'''
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.

 

Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
Example 2:

Input: s = "3[a2[c]]"
Output: "accaccacc"
Example 3:

Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
 

Constraints:

1 <= s.length <= 30
s consists of lowercase English letters, digits, and square brackets '[]'.
s is guaranteed to be a valid input.
All the integers in s are in the range [1, 300].
'''
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for c in s:
            if c != ']':
                stack.append(c)
            else:
                substr = ""
                while stack[-1] != "[":
                    substr = stack.pop() + substr
                
                stack.pop() # for the '['

                k = ""
                while stack and stack[-1].isdigit():
                    k = stack.pop() + k
                
                substr = int(k) * substr

                for char in substr:
                    stack.append(char)
        return "".join(stack)
```
### Can Place Flowers
```python
'''
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.

 

Example 1:

Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
Example 2:

Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
 

Constraints:

1 <= flowerbed.length <= 2 * 104
flowerbed[i] is 0 or 1.
There are no two adjacent flowers in flowerbed.
0 <= n <= flowerbed.length
'''
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], m: int) -> bool:
        n = len(flowerbed)
        if len(flowerbed) == 1:
            if flowerbed[0] == 0:
                if m == 0 or m == 1:
                    return True
                else:
                    return False
            else:
                if m == 0:
                    return True
                else:
                    return False

        if flowerbed[0] == 0 and flowerbed[1] == 0:
            flowerbed[0] = 1
            m -= 1
        
        if flowerbed[n - 1] == 0 and flowerbed[n - 2] == 0:
            flowerbed[n - 1] = 1
            m -= 1
        
        for i in range(1, n - 1):
            if flowerbed[i] == 0 and flowerbed[i - 1] == 0 and flowerbed[i + 1] == 0:
                flowerbed[i] = 1
                m -= 1
        
        return True if m <= 0 else False
```