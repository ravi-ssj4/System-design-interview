
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

### Q. Print all Subsequences
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
    
    backtrack(0, [])

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

### General Backtracking pattern with loop
Q. Find all permutations -> using extra space!
```python
# Solution with extra space: freq-counter to keep track of which elements are available -> O(n) extra space!
# Overall Time = O(n! * n)
def solution(nums):
    N = len(nums)

    res = []
    freq = [0] * N

    def backtrack(perm, freq):
        # base case
        if len(perm) == N:
            res.append(perm.copy())
            return
        
        for i in range(N):
            if freq[i] == 0:
                freq[i] = 1
                perm.append(nums[i])
                backtrack(perm, freq)
                freq[i] = 0
                perm.pop()

    backtrack([], freq)
    return res

```

#### Backtracking Swapping technique
Q. Find all permutations -> without using extra space!

```python
# Solution without extra space -> The swapping technique! -> Another pattern
def solution(nums):
    N = len(nums)
    res = []

    def swap(i, j, array):
        array[i], array[j] = array[j], array[i]

    def backtrack(idx, perm):
        # base case
        if idx == N:
            res.append(perm.copy())

        # general case loop
        for i in range(idx, N): # for each idx from idx to N - 1
            swap(idx, i, perm) # swap i with idx
            backtrack(i + 1, perm)
            swap(i, idx, perm) # swap back i with idx -> backtrack
    
    perm = nums.copy() # to not modify the original input array
    backtrack(perm)
    return res
```

### N - Queens
```python

```
### Palindrome Partitioning
```python

```
### Rat in a Maze
```python

```
### kth permutation sequence
```python

```

