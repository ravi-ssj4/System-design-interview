
### General Code Structure - Backtracking

```python
# time: O((2**n) * n) -> every idx has 2 options(take/not take) & deep copy = O(n)
# space: O(n) -> max rec depth
def solution(nums):
    res = []
    N = len(nums)
    def backtrack(i, someList):
        # base case
        if i >= N:
            res.append(someList.copy()) # O(n)
            return

        # take condition
        someList.append(nums[i])
        backtrack(i + 1, someList)

        # not take condition
        someList.remove(nums[i])
        backtrack(i + 1, someList)
    
    backtrack(0, []) # O(2**n)
    return res

```

### Print all Subsequences
What's a subsequence? <br> A sequence ( can be contiguous or non-contiguous ) which follows the original ordering of elements

#### Diagram for Vertical recursion: (later)

#### Diagram for recursion tree: (later)

```python
def solution():
    nums = [3, 1, 2]
    N = len(nums)
    def backtrack(i, subseq):
        if i >= N:
            print(subseq)
            return
        
        # take condition
        subseq.append(nums[i])
        backtrack(i + 1, subseq)

        # not take condition
        subseq.pop()
        backtrack(i + 1, subseq)

```

### Backtracking with constraint -> all results
Q. Print all subsequences when their sum == k
```python
def solution(nums, k):
    N = len(nums)
    def backtrack(i, subseq, summ):
        # base case
        if i == N:
            if summ == k:
                print(subseq)
                return

        # take condition
        subseq.append(nums[i])
        summ += nums[i]
        backtrack(i + 1, subseq, summ)

        # not take condition
        subseq.pop()
        summ -= nums[i]
        backtrack(i + 1, subseq, summ)
    
    backtrack(0, [], 0)

```

### Backtracking with constraint -> only 1st result
Q. Print any subsequence with sum == k
```python
def solution(nums, k):
    N = len(nums)

    def backtrack(i, subseq, summ):
        # base case
        if i == N:
            if summ == k:
                print(subseq)
                return True
            return False

        # take condition
        subseq.append(nums[i])
        summ += nums[i]
        if backtrack(i + 1, subseq, summ) == True:
            return True

        # not take condition
        subseq.pop()
        summ -= nums[i]
        if backtrack(i + 1, subseq, summ) == True:
            return True
        
        return False

    backtrack(0, [], 0)

```

### Backtracking with constraint -> count of results
Q. Find the count of all the subsequences when sum == k
```python
    def solution(nums, k):
        N = len(nums)
            
        def backtrack(i, subseq, summ):
            # base case
            if i == N:
                if summ == k:
                    return 1
                else:
                    return 0

            # take condition
            subseq.append(nums[i])
            summ += nums[i]
            k = backtrack(i + 1, subseq, summ)

            # not take condition
            subseq.pop()
            summ -= nums[i]
            l = backtrack(i + 1, subseq, summ)

            return k + l
        
        return backtrack(0, [], 0)
        
```







