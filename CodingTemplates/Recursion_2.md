### Permutations
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        def backtrack(i, arr):
            # base case
            if i == n:
                res.append(arr.copy())
                return
            
            # gen case
            for j in range(i, n):
                arr[i], arr[j] = arr[j], arr[i]
                backtrack(i + 1, arr)
                arr[i], arr[j] = arr[j], arr[i]
        
        backtrack(0, nums)
        return res
```
