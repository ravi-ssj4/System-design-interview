### 1. Inorder Successor in BST II
```python
"""
Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Optional[Node]':
        def getLeftmost(node):
            while node.left:
                node = node.left
            return node

        successor = None
        # two cases
        # if node has right child
        if node.right:
            successor = getLeftmost(node.right)
        else: # if node dosen't have right child
            while node.parent:
                if node.parent.left == node:
                    successor = node.parent
                    break
                else:
                    node = node.parent
        return successor
```
### 2. Inorder Successor in BST
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        successor = None
        while root:
            if root.val > p.val: # have to go left -> root is a probable answer
                successor = root
                root = root.left
            else:
                root = root.right
        return successor

```	
### 3. Rotated Digits	
```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        invalidSet = set([3, 4, 7])
        validSet = set([2, 5, 6, 9])

        cnt = 0
        for i in range(1, n + 1):
            num = i
            valid = False
            while num:
                digit = num % 10
                if digit in validSet:
                    valid = True
                if digit in invalidSet:
                    valid = False
                    break
                num = num // 10
            
            if valid:
                cnt += 1

        return cnt
```
### 4. Missing Element in Sorted Array
```python

```	
### 5. Restore IP Addresses		
```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        n = len(s)
        if n > 12:
            return []
        
        def backtrack(i, dots, curIp):
            # base case
            if i == n and dots == 4:
                res.append(curIp[:-1])
                return
            # gen case
            for j in range(i, min(i + 3, n)):
                if int(s[i: j + 1]) < 256 and (i == j or s[i] != "0"):
                    backtrack(j + 1, dots + 1, curIp + s[i: j + 1] + ".")
            
        backtrack(0, 0, "")
        return res
```
### 6. Moving Average from Data Stream
```python
class MovingAverage:

    def __init__(self, size: int):
        self.l, self.r = 0, 0
        self.size = size
        self.slidingWindow = []
        self.sum = 0


    def next(self, val: int) -> float:
        self.slidingWindow.append(val)
        self.sum += self.slidingWindow[self.r]
        
        if self.r - self.l + 1 > self.size: # shrink the window
            self.sum -= self.slidingWindow[self.l]
            self.l += 1
        
        movingAvg = self.sum / (self.r - self.l + 1)

        self.r += 1
        
        return movingAvg


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)

```	
### 7. Reverse Linked List II
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        first, second = None, None
        dummy = ListNode()
        dummy.next = head
        cur = dummy
        cnt = 1

        while cur.next:
            if cnt == left:
                firstPrev = cur
                first = cur.next
            if cnt == right:
                second = cur.next
                break
            cur = cur.next
            cnt += 1

        secondNext = second.next

        prev = None
        cur = first
        while cur != secondNext:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        firstPrev.next = prev
        first.next = secondNext

        return dummy.next 

```
### 8. Reorder List
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        dummy = ListNode()
        dummy.next = head

        # locate middle / bifurcate into two halves
        fast = slow = dummy
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        second = slow.next
        slow.next = None

        # reverse second list
        prev = None
        cur = second
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        
        # re-arrange list
        second = prev
        first = head
        while first and second:
            tmp1 = first.next
            tmp2 = second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2
        
        return dummy.next

```
### 9. Add Two Numbers
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy
        carry = 0
        while l1 or l2 or carry:
            num1 = l1.val if l1 else 0
            num2 = l2.val if l2 else 0
            Sum = num1 + num2 + carry
            digit = Sum % 10
            carry = Sum // 10
            tail.next = ListNode(digit)
            tail = tail.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
```
### 10. Reverse Words in a String
```python
class Solution:
    # def reverseWords(self, s: str) -> str:
    #     s = " " + s
    #     res = []
    #     i = len(s) - 1
    #     while i >= 0:
    #         if s[i] != " ":
    #             j = i
    #             while s[j] != " ":
    #                 j -= 1
    #             res.append(s[j + 1: i + 1])
                 
    #             i = j
    #         else:
    #             i -= 1
            
    #     return " ".join(res)
    
    def reverseWords(self, s: str) -> str:
        s = " " + s
        res = ""
        i = len(s) - 1
        while i >= 0:
            if s[i] != " ":
                j = i
                while s[j] != " ":
                    j -= 1
                # res.append(s[j + 1: i + 1])
                res += s[j + 1: i + 1] + " "
                i = j
            else:
                i -= 1
            
        return res[:-1]
```
### 11 Binary Tree Maximum Path Sum
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # without split: max sum via one path only (from the root to the
        # leaf of that tree in one direction either left/right)
        # with split: max sum including both sides left and right including the root
        maxSum = float("-inf") # to calculate the maxSum with split and overall
        def dfs(node): # returns the maxSum without split
            nonlocal maxSum
            if node == None:
                return 0
            '''
                    15
                  /    \
                20       30
                       /   \
                    -10     -20
            leftMaxPathSumWithoutSplit = -10
            rightMaxPathSumWithoutSplit = -20
            so, better to not carry them forward(kadane logic) -> make them 0s in the  beginning itself
            '''
            leftMaxSumWithoutSplit = max(dfs(node.left), 0)
            rightMaxSumWithoutSplit = max(dfs(node.right), 0)

            maxSumWithSplit = node.val + leftMaxSumWithoutSplit + rightMaxSumWithoutSplit
            maxSumWithoutSplit = node.val + max(leftMaxSumWithoutSplit, rightMaxSumWithoutSplit)

            maxSum = max(maxSum, maxSumWithSplit, maxSumWithoutSplit)

            return maxSumWithoutSplit
        
        dfs(root)

        return maxSum

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        maxPathSumWithSplit = [root.val]

        def helper(node): # returns mps without split
            # base case
            if node == None:
                return 0

            # gen case
            leftMax = helper(node.left)
            rightMax = helper(node.right)
            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)

            # calc. of mps with split
            maxPathSumWithSplit[0] = max(maxPathSumWithSplit[0], node.val + leftMax + rightMax)

            # return of mps without split
            return node.val + max(leftMax, rightMax)
        
        helper(root)
        return maxPathSumWithSplit[0]


```	
### 12 Greatest Common Divisor of Strings
```python
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        
        def helper(n1, n2):
            hcf = 1
            for i in range(min(n1, n2), 0, -1):
                if n1 % i == 0 and n2 % i == 0:
                    hcf = i
                    break
            return hcf

        n1, n2 = len(str1), len(str2)
        gcd = helper(n1, n2)

        div1 = n1 // gcd
        div2 = n2 // gcd
        
        substr = str1[:gcd]
        
        if substr * div1 == str1 and substr * div2 == str2:
            return substr
        else:
            return ""
```
### 13 Reverse Nodes in k-Group
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:

        def getKthNode(node, k):
            k -= 1
            while node and k:
                node = node.next
                k -= 1
            return node
        
        def reverse(node):
            cur = node
            prev = None
            while cur:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            return prev

        temp = head
        prevNode = None
        while temp:
            kthNode = getKthNode(temp, k)
            if kthNode == None:
                if prevNode: # prevList = None if first group itself is having #nodes < k
                    prevNode.next = temp # connecting the leftover list(ie. #nodes < k)
                break
            nextNode = kthNode.next
            kthNode.next = None # the sublist is ready to be reversed

            _ = reverse(temp)
            # kth node points to the head of reversed list
            # temp points to the tail of the reversed list
            if temp == head: # first group
                head = kthNode
            else: # for later groups
                prevNode.next = kthNode # connecting the prev list with the newly reversed list(whose head = kthNode)
            prevNode = temp
            temp = nextNode
        return head
```
### 14 Find Minimum in Rotated Sorted Array
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # note: we can only decide something in the sorted half, unsorted half is always uncertain
        n = len(nums)
        low, high = 0, n - 1
        mini = float("inf")
        while low <= high:
            mid = (low + high) // 2
            if nums[low] <= nums[mid]: # left half sorted
                mini = min(mini, nums[low])
                low = mid + 1
            else:
                mini = min(mini, nums[mid])
                high = mid - 1
        return mini
```
### 15 Reverse String
```python
# too easy
```
### 16 Find the Town Judge
```python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        indegrees = [0] * (n + 1)
        outdegrees = [0] * (n + 1)

        for a, b in trust:
            outdegrees[a] += 1
            indegrees[b] += 1
        
        for i in range(1, n + 1):
            if outdegrees[i] == 0 and indegrees[i] == n - 1:
                return i
        return -1
```	
### 17 LCA in BST
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
```
### 18 Construct String With Repeat Limit
```python

```
### 19 Valid parenthesis
```python
class Solution:
    def isValid(self, s: str) -> bool:
        closeToOpen = {")": "(", "}": "{", "]": "["}
        stack = []
        for c in s:
            if c in closeToOpen:
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        return len(stack) == 0
        '''
        * closeToOpen karke ek mapping banayenge (key: close bracket, value: open bracket)
        * saare chars par iterate karenge
        * agar current char closing bracket hai, to fir stack check karenge
        * agar stack top par uska hi opening bracket hai, to fir usko pop kar denge, otherwise return False
        * agar current char open bracket hai, toh sidha push kar denge stack me
        * example stack: [ '(' , '[' ... ]
        '''
```
### 20 Maximum depth of n-ary tree
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if root == None:
            return 0

        maxChildDepth = 0
        for child in root.children:
            childDepth = self.maxDepth(child)
            maxChildDepth = max(maxChildDepth, childDepth)
        
        return maxChildDepth + 1
```
### 21 How many numbers are smaller than the current number
```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        temp = sorted(nums)
        hashMap = {}
        for i in range(len(temp)):
            if temp[i] not in hashMap:
                hashMap[temp[i]] = i
            
        res = []
        for i in nums:
            res.append(hashMap[i])
        return res
```
### 22 Delete all occurrences of a given key in a doubly linked list
```python
class Solution:
    #Function to delete all the occurances of a key from the linked list.
    def deleteAllOccurOfX(self, head, x):
        # code here
        # edit the linked list
        temp = head
        while temp:
            if temp.data == x:
                if temp == head:
                    head = head.next
                
                prevNode = temp.prev
                nextNode = temp.next
                
                if prevNode:
                    prevNode.next = nextNode
                if nextNode:
                    nextNode.prev = prevNode
                
                del temp
                
                temp = nextNode
            
            else:
                
                temp = temp.next
        return head
```
### 23 Iterative inorder
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node = root
        stack = []
        res = []
        while node or stack:
            if node != None:
                stack.append(node)
                node = node.left
            else:
                if stack == None:
                    break
                node = stack.pop()
                res.append(node.val)
                node = node.right
        return res
```
### 24 Kth Smallest Element in a BST
https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        node = root
        stack = []
        cnt = 0
        kthSmallest = -1
        while node or stack:
            if node != None:
                stack.append(node)
                node = node.left
            else:
                if stack == None:
                    break
                node = stack.pop()
                cnt += 1
                if cnt == k:
                    kthSmallest = node.val
                    break
                node = node.right
        return kthSmallest
```
### 25 BST Iterator
```python
'''
Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
int next() Moves the pointer to the right, then returns the number at the pointer.
Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.

 

Example 1:


Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]

Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False
 

Constraints:

The number of nodes in the tree is in the range [1, 105].
0 <= Node.val <= 106
At most 105 calls will be made to hasNext, and next.
 

Follow up:

Could you implement next() and hasNext() to run in average O(1) time and use O(h) memory, where h is the height of the tree?
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        node = root
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        nextNode = self.stack.pop()
        cur = nextNode.right
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return nextNode.val

    def hasNext(self) -> bool:
        if self.stack:
            return True
        else:
            return False


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
```
### 26 Children Sum Property in Binary Tree | simple preorder traversal
```python
'''
Given a binary tree having n nodes. Check whether all of its nodes have the value equal to the sum of their child nodes. Return 1 if all the nodes in the tree satisfy the given properties, else it return 0.

For every node, data value must be equal to the sum of data values in left and right children. Consider data value as 0 for NULL child.  Also, leaves are considered to follow the property.

Example 1:

Input:
Binary tree
       35
      /   \
     20  15
    /  \  /  \
   15 5 10 5
Output: 
1
Explanation: 
Here, every node is sum of its left and right child.
Example 2:

Input:
Binary tree
       1
     /   \
    4    3
   /  
  5    
Output: 
0
Explanation: 
Here, 1 is the root node and 4, 3 are its child nodes. 4 + 3 = 7 which is not equal to the value of root node. Hence, this tree does not satisfy the given condition.
Your Task:
You don't need to read input or print anything. Your task is to complete the function isSumProperty() that takes the root Node of the binary tree as input and returns 1 if all the nodes in the tree satisfy the following properties, else it returns 0.

Expected Time Complexiy: O(n).
Expected Auxiliary Space: O(Height of the Tree).

Constraints:
1 <= n <= 105
1 <= Data on nodes <= 105
'''

class Solution:
    #Function to check whether all nodes of a tree have the value 
    #equal to the sum of their child nodes.
    def isSumProperty(self, root):

        def isLeaf(node):
            return node.left == None and node.right == None
            
        if root == None or isLeaf(root):
            return 1
        
        leftVal, rightVal = 0, 0
        
        if root.left:
            leftVal = root.left.data
        if root.right:
            rightVal = root.right.data
        
        if leftVal + rightVal != root.data:
            return 0
        
        left = self.isSumProperty(root.left)
        right = self.isSumProperty(root.right)
        
        return left and right
```
### 27 Root to Node Path | Common path list + backtracking
```python
'''
Can't find link -> just find root to target node path
'''
class Solution:
    def Paths(self, root : Optional['Node'], targetNode : 'Node') -> List[List[int]]:
        path = []
        def f(node, target):
            if node == None:
                return False
            if node.data == target.data:
                path.append(node.data)
                return True
            path.append(node.data)
            left = f(node.left, target)
            if left == False:
                right = f(node.right, target)
            if left == False and right == False:
                path.pop()
                return False
            else:
                return True
        return path
```
### 28 Root to leaf paths | Modified above logic
If we are at a node and from both sides we are getting a false(ie. from left and right subtrees)
and if the current node is a leaf node, then the path list that we have has the path till that node(leaf)
at this point of time -> append it to the result -> remove the last element(previous logic)
```python
'''
Given a Binary Tree of nodes, you need to find all the possible paths from the root node to all the leaf nodes of the binary tree.

Example 1:

Input:
       1
    /     \
   2       3
Output: 
1 2 
1 3 
Explanation: 
All possible paths:
1->2
1->3
Example 2:

Input:
         10
       /    \
      20    30
     /  \
    40   60
Output: 
10 20 40 
10 20 60 
10 30 
Your Task:
Your task is to complete the function Paths() which takes the root node as an argument and returns all the possible paths. (All the paths are printed in new lines by the driver's code.)

Expected Time Complexity: O(n)
Expected Auxiliary Space: O(height of the tree)

Constraints:
1<=n<=104
'''
class Solution:
    def Paths(self, root : Optional['Node']) -> List[List[int]]:
        path = []
        res = []
        def f(node):
            if node == None:
                return False
                
            path.append(node.data)
            left = f(node.left)
            if left == False:
                right = f(node.right)
            if left == False and right == False:
                if node.left == None and node.right == None:
                    res.append(path.copy())
                path.pop()
                return False
            else:
                return True

        f(root)
        return res
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