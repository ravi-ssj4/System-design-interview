### L08 | Level Order Traversal
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            n = len(q)
            tempList = []
            for _ in range(n):
                node = q.popleft()
                if node:
                    tempList.append(node.val)
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            if tempList:
                res.append(tempList)
        return res

```
### L09 | Iterative Preorder Traversal
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        stack.append(root)
        res = []
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        return res
```
### L10 | Iterative inorder traversal
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        res = []
        node = root
        while True:
            if node: # go as left as possible
                stack.append(node) # filling the aux stack space(memory)
                node = node.left
            else: # node == None
                if not stack:
                    break
                node = stack.pop()
                res.append(node.val) # process the node
                node = node.right # go right
        return res
```
### L11 | Iterative postorder traversal (1 stack)
```python

```
### L14 | Max Depth of Binary tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# postorder: calculate the maxDepth of left and right children (recursively), then calculate the depth of current node
class Solution:
    # gives me the max depth of tree rooted at node = root
    def maxDepth(self, root: Optional[TreeNode]) -> int: 
        if root == None:
            return 0
        
        # postorder dfs traversal
        # get depth of lst
        # get depth of rst
        # calculate the depth of current node
        maxDepthOfLeft = self.maxDepth(root.left)
        maxDepthOfRight = self.maxDepth(root.right)
        maxDepthCurrent = 1 + max(maxDepthOfLeft, maxDepthOfRight)
        return maxDepthCurrent
        # return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```
### L15 | Check for Balanced Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # returns 2 values for tree rooted @node: [if its balanced, its height]
        def dfs(node):
            if node == None:
                return [True, -1]
            
            left = dfs(node.left)
            right = dfs(node.right)
            heightCurr = 1 + max(left[1], right[1])
            isBalancedCurr = left[0] and right[0] and abs(left[1] - right[1]) <= 1
            return [isBalancedCurr, heightCurr]
        
        return dfs(root)[0]
```
### L16 | Diameter of Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # returns 2 values for tree rooted @node: [diameter, height]
        # here diameter = max(diam via node, diam of left subtree, diam of right subtree)
        # diam via node = 2 + leftheight + rightHeight
        def dfs(node):
            if node == None:
                return [0, -1]
            
            leftDiam, leftHeight = dfs(node.left)
            rightDiam, rightHeight = dfs(node.right)
            
            currHeight = 1 + max(leftHeight, rightHeight)
            
            currDiamViaRoot = 2 + leftHeight + rightHeight
            currDiam = max(currDiamViaRoot, leftDiam, rightDiam)

            return [currDiam, currHeight]
        
        return dfs(root)[0]

```
### L17 | Maximum path sum in binary tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # without split: max sum from the root to the leaf of that tree in one direction either left/right
        # with split: max sum including both sides left and right, ie. leaf of lst to leaf of rst passing via root
        maxSum = [float("-inf")] # to calculate the maxSum with split and overall
        def dfs(node): # returns the maxSum without split
            if node == None:
                return 0
            
            leftMaxSumWithoutSplit = dfs(node.left)
            rightMaxSumWithoutSplit = dfs(node.right)

            maxSumWithSplit = max((node.val + leftMaxSumWithoutSplit + rightMaxSumWithoutSplit), node.val)
            maxSumWithoutSplit = max(max(leftMaxSumWithoutSplit, rightMaxSumWithoutSplit) + node.val, node.val)

            maxSum[0] = max(maxSum[0], maxSumWithSplit, maxSumWithoutSplit)

            return maxSumWithoutSplit
        
        dfs(root)

        return maxSum[0]

```
### L18 | Check if two trees are identical or not
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None and q == None:
            return True
        if p == None or q == None:
            return False
        if p.val != q.val:
            return False
        
        # we are already sure that p.val == q.val here
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```
### L19 | Zig-zag or spiral traversal in Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# [3, 2]
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        rightToLeft = True
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            n = len(q)
            tempList = []
            for _ in range(n):
                node = q.popleft()
                if node:
                    q.append(node.left)
                    q.append(node.right)
                    tempList.append(node.val)
            if tempList:
                if rightToLeft:
                    res.append(tempList)
                else:
                    res.append(tempList[::-1])
            rightToLeft = not rightToLeft
        return res
                
```
### L20 | Boundary Traversal in binary tree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root == None:
            return -1

        def isLeaf(node):
            return node.left == None and node.right == None

        def leftBoundary(node):
            cur = node.left
            while cur:
                if not isLeaf(cur):
                    res.append(cur.val)
                if cur.left:
                    cur = cur.left
                else:
                    cur = cur.right
        
        def leaves(node):
            if node == None:
                return
            leaves(node.left)
            if isLeaf(node):
                res.append(node.val)
            leaves(node.right)
    
        def rightBoundary(node):
            stack = []
            cur = node.right
            while cur:
                if not isLeaf(cur):
                    stack.append(cur.val)
                if cur.right:
                    cur = cur.right
                else:
                    cur = cur.left
            while stack:
                res.append(stack.pop())
        
        res.append(root.val)
        leftBoundary(root)
        if not isLeaf(root):
            leaves(root)
        rightBoundary(root)
        return res
```
### L21 | Vertical order traversal
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    '''
    step 1: create a queue data structure which will traverse the tree
    step 2: create another data structure to store the information: 
            sorted node list for every vertical and horizontal level (hashmap of hashmap of heaps)
    step 3: iterate through the hashmap in sorted fashion(according to the vertical level)
            for each vertical, iterate through its horizontal level's heap and fill the temp List
            add the temp list to the result after every horizonttal level's iteration is done
    '''
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = collections.deque()
        q.append([root, 0, 0])
        # {verticalLevel: {horizontalLevel: [Priority Queue]}}
        hashMap = defaultdict(lambda: defaultdict(list))
        while q:
            node, vertical, level = q.popleft()
            
            heapq.heappush(hashMap[vertical][level], node.val)

            if node.left:
                q.append([node.left, vertical - 1, level + 1])
            if node.right:
                q.append([node.right, vertical + 1, level + 1])
        
        sortedHashMap = dict(sorted(hashMap.items()))

        res = []
        for vertical, levelDict in sortedHashMap.items():
            tempList = []
            for level, heap in levelDict.items():
                while heap:
                    tempList.append(heapq.heappop(heap))
            res.append(tempList)

        return res
```
### L22 | Top view
```python
from collections import deque

class Solution:
    
    #Function to return a list of nodes visible from the top view 
    #from left to right in Binary Tree.
    def topView(self,root):
        q = deque()
        q.append([root, 0])
        hashMap = {}
        while q:
            node, verticalLevel = q.popleft()
            if verticalLevel not in hashMap:
                hashMap[verticalLevel] = node.data
            if node.left:
                q.append([node.left, verticalLevel - 1])
            if node.right:
                q.append([node.right, verticalLevel + 1])
        
        sortedDict = dict(sorted(hashMap.items()))
        
        res = []
        for key, value in sortedDict.items():
            res.append(value)
        return res
```
### L23 | Bottom view
```python
def bottomView(self, root):
        q = deque()
        q.append([root, 0])
        hashMap = {}
        while q:
            node, verticalLevel = q.popleft()
            hashMap[verticalLevel] = node.data
            if node.left:
                q.append([node.left, verticalLevel - 1])
            if node.right:
                q.append([node.right, verticalLevel + 1])
        
        sortedDict = dict(sorted(hashMap.items()))
        
        res = []
        for key, value in sortedDict.items():
            res.append(value)
        return res
```
### L24 | Left/Right view
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            n = len(q)
            rightmost = None
            for _ in range(n):
                node = q.popleft()
                if node:
                    rightmost = node.val
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            if rightmost != None:
                res.append(rightmost)
        return res

    def leftSideView(self, root: Optional[TreeNode]) -> List[int]:
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            n = len(q)
            leftmost = None
            for _ in range(n):
                node = q.popleft()
                if node:
                    if leftmost == None:
                        leftmost = node.val
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
            if leftmost != None:
                res.append(leftmost)
        return res
```
### L25 | Check for symmetrical Binary Trees
```python

```
### L26 - A | Print root to node path in Binary Tree
```python
class Solution:
    def getPath(root: Optional['Node'], x: int) -> List[int]:

        res = []
        
        def dfs(node):
            if node == None:
                return False
            
            nodeList.append(node.val)

            # check the node itself
            if node.val == x:
                return True
            
            # check the subtrees
            if dfs(node.left) == True or dfs(node.right) == True:
                return True
            
            # backtrack
            nodeList.pop()

            return False
        
        dfs(root)
        return res

```
### L26 - B | Find all paths from root to all leaves
```python
"""

definition of binary tree node.
class Node:
    def _init_(self,val):
        self.data = val
        self.left = None
        self.right = None
"""
class Solution:
    def Paths(self, root : Optional['Node']) -> List[List[int]]:
        
        res = []
        
        def rootToLeaf(node, nodeList):
            nodeList.append(node.data)
            if node.left == None and node.right == None:
                res.append(nodeList.copy())
                return
            
            if node.left:
                rootToLeaf(node.left, nodeList)
                nodeList.pop()
                    
            if node.right:
                rootToLeaf(node.right, nodeList)
                nodeList.pop()
            
        rootToLeaf(root, [])
        return res
```
### L27 | Lowest Common ancestor in Binary Tree
```python

```

### L28 | Maximum width of binary tree
```python

```

### L29 | Children sum property in binary tree
```python

```

### L30 | Print all nodes at a distance of k
```python

```

### L31 | Maximum time taken to burn the binary tree
```python

```

### L32 | Count total nodes in a COMPLETE binary tree
```python

```

### L |
```python

```

### L |
```python

```

### L |
```python

```

### L |
```python

```

