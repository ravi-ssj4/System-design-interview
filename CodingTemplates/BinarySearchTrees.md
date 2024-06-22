### Search in BST
https://leetcode.com/problems/search-in-a-binary-search-tree/
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        while root and root.val != val:
            root = root.left if val < root.val else root.right
        return root
```
### Find min/max in BST
```python
#User function Template for python3

"""
class Node:
    def __init__(self, val):
        self.right = None
        self.data = val
        self.left = None
"""

class Solution:
    #Function to find the minimum element in the given BST.
    def minValue(self, root):
        while root:
            val = root.data
            root = root.left
        return val

#{ 
 # Driver Code Starts

#Initial Template for Python 3

from collections import deque
# Tree Node
class Node:
    def __init__(self, val):
        self.right = None
        self.data = val
        self.left = None

# Function to Build Tree   
def buildTree(s):
    #Corner Case
    if(len(s)==0 or s[0]=="N"):           
        return None
        
    # Creating list of strings from input 
    # string after spliting by space
    ip=list(map(str,s.split()))
    
    # Create the root of the tree
    root=Node(int(ip[0]))                     
    size=0
    q=deque()
    
    # Push the root to the queue
    q.append(root)                            
    size=size+1 
    
    # Starting from the second element
    i=1                                       
    while(size>0 and i<len(ip)):
        # Get and remove the front of the queue
        currNode=q[0]
        q.popleft()
        size=size-1
        
        # Get the current node's value from the string
        currVal=ip[i]
        
        # If the left child is not null
        if(currVal!="N"):
            
            # Create the left child for the current node
            currNode.left=Node(int(currVal))
            
            # Push it to the queue
            q.append(currNode.left)
            size=size+1
        # For the right child
        i=i+1
        if(i>=len(ip)):
            break
        currVal=ip[i]
        
        # If the right child is not null
        if(currVal!="N"):
            
            # Create the right child for the current node
            currNode.right=Node(int(currVal))
            
            # Push it to the queue
            q.append(currNode.right)
            size=size+1
        i=i+1
    return root
    
    
if __name__=="__main__":
    t=int(input())
    for _ in range(0,t):
        s=input()
        root=buildTree(s)
        ob = Solution()
        print(ob.minValue(root))
# } Driver Code Ends
```
### Ceil / Floor in a BST
https://www.geeksforgeeks.org/problems/implementing-ceil-in-bst/1
```python
#Function to return the ceil of given number in BST.

class Solution:
    def findCeil(self,root, target):
        ans = -1
        while root:
            if root.key >= target:
                ans = root.key # probable answer
                root = root.left # in search for a better ans(even smaller value -> obv on the left)
            else:
                root = root.right # in search of an answer on the right
        return ans

    def floor(self, root, target):
        ans = -1
        while root:
            if root.data <= target:
                ans = root.data
                root = root.right
            else:
                root = root.left
        return ans


#{ 
 # Driver Code Starts
#Initial Template for Python 3

from collections import deque
# Tree Node
class Node:
    def __init__(self, val):
        self.right = None
        self.key = val
        self.left = None

# Function to Build Tree  

def buildTree(s):
        #Corner Case
        if(len(s)==0 or s[0]=="N"):           
            return None
            
        # Creating list of strings from input 
        # string after spliting by space
        ip=list(map(str,s.split()))
        
        # Create the root of the tree
        root=Node(int(ip[0]))                     
        size=0
        q=deque()
        
        # Push the root to the queue
        q.append(root)                            
        size=size+1 
        
        # Starting from the second element
        i=1                                       
        while(size>0 and i<len(ip)):
            # Get and remove the front of the queue
            currNode=q[0]
            q.popleft()
            size=size-1
            
            # Get the current node's value from the string
            currVal=ip[i]
            
            # If the left child is not null
            if(currVal!="N"):
                
                # Create the left child for the current node
                currNode.left=Node(int(currVal))
                
                # Push it to the queue
                q.append(currNode.left)
                size=size+1
            # For the right child
            i=i+1
            if(i>=len(ip)):
                break
            currVal=ip[i]
            
            # If the right child is not null
            if(currVal!="N"):
                
                # Create the right child for the current node
                currNode.right=Node(int(currVal))
                
                # Push it to the queue
                q.append(currNode.right)
                size=size+1
            i=i+1
        return root
    
    
if __name__=="__main__":
    t=int(input())
    for _ in range(0,t):
        s=input()
        n=int(input())
        root=buildTree(s)
        obj=Solution()
        print(obj.findCeil(root,n))
# } Driver Code Ends
```
### Insert a given node in a BST
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        '''
        either target < current node or greater than current node
        wherever the target is first check if that node exists or not
        if it exists go to that direction, else insert
        '''
        if root == None:
            return TreeNode(target)
        cur = root
        while cur:
            if target < cur.val:
                if cur.left:
                    cur = cur.left
                else:
                    cur.left = TreeNode(target)
                    break
            else:
                if cur.right:
                    cur = cur.right
                else:
                    cur.right = TreeNode(target)
                    break
        return root

```
### Delete a given node in a BST
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        
        def getRightmost(node):
            while node.right:
                node = node.right
            return node

        def helper(node):
            if node.right == None:
                return node.left
            if node.left == None:
                return node.right

            rightChild = node.right
            rightmostInLeft = getRightmost(node.left)
            rightmostInLeft.right = rightChild
            return node.left

        # in case of empty tree
        if root == None:
            return root
            
        # the search part
        temp = root
        # if root itself needs to be deleted
        if root.val == key:
            return helper(root)
        
        # we are sure its not the root node that's to be deleted
        while root:
            if key < root.val:
                if root.left and root.left.val == key:
                    root.left = helper(root.left)
                    break
                else:
                    root = root.left
            else:
                if root.right and root.right.val == key:
                    root.right = helper(root.right)
                    break
                else:
                    root = root.right
        return temp

```
### Find kth smallest/largest element in a BST
```python

```
### Check if BT is a BST or not
```python

```
### LCA in BST
```python

```
### Construct BST using preorder traversal
```python

```
### Inorder Successor / Predecessor in BST
```python

```
### Merge 2 BSTs
```python

```