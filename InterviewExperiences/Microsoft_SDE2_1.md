https://leetcode.com/discuss/interview-experience/4920820/Microsoft-or-SDE2-L61-or-India-or-March-Offer



```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for asteroid in asteroids:
            if asteroid < 0:
                while stack and stack[-1] > 0 and stack[-1] < abs(asteroid):
                    stack.pop()
                if stack and stack[-1] > 0:
                    if stack[-1] == abs(asteroid):
                        stack.pop()
                else:
                    stack.append(asteroid)
                continue
            else:
                stack.append(asteroid)
        return stack
```

```python
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