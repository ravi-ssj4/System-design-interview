### Insert Interval
```python
class Solution:
    # * non-overlapping intervals already
    # * already sorted in ascending order


    # intervals = [[1, 3], [6, 9]]
    # newInterval = [2, 5]

    # case 1: newInterval before current interval -> add newInterval to res and the rest of the intervals as well

    # case 2: newInterval after current interval -> add current interval to result

    # case 3: newInterval overlaps with the current interval 
    #     -> merge current and new interval
    #     -> since the updated interval can again overlap with the remaining,
    #         -> we make newInterval = updatedInveral and continue
    #         -> as now for this newInterval and the remaining, intervals,
    #         -> we are back at the same state, hence any of the cases 1, 2 or 3 can re-occur

    # * clarifying question: does [1, 2] and [2, 3] overlap?
    # -> for this question: yes

    # intervals = [[1, 3], [6, 9]]
    # newInterval = [2, 5]

    def insert(self, intervals, newInterval):
        n = len(intervals)
        res = []
        for i in range(n):
            curStart = intervals[i][0]
            curEnd = intervals[i][1]
            newStart = newInterval[0]
            newEnd = newInterval[1]
            # dosen't overlap
            # before
            if newEnd < curStart:
                res.append(newInterval)
                return res + intervals[i:]
            # after
            elif curEnd < newStart:
                res.append(intervals[i])
            # overlap
            else:
                newInterval[0] = min(curStart, newStart)
                newInterval[1] = max(curEnd, newEnd)
        
        res.append(newInterval)

        return res
        
```

### Merge Intervals
```python
class Solution:
    # eg 1: intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]

    # eg 2: intervals = [[1, 4], [4, 5]] -> considered overlapping this means!

    # Note: here its not given that the intervals are sorted, but in the eg. 1, they are sorted according to start times
    # -> gives us the hint that these intervals need to be sorted to solve the problem

    # * again building into cases:

    # prevInterval = intervals[0]
    # start iteration from intervals[1]

    # case 1: prevInterval before the currentInterval -> put prevInterval in the res and make prevInterval = curInterval

    # case 2: prevInterval after the current interval -> cannot happen as intervals are sorted by start times!

    # case 3: prevInterval and curInterval overlap! -> merge them and make this mergedInterval as prevInterval

    # dry run:

    # [[1, 3], [2, 6], [8, 10], [15, 18]]


    # res = [[1, 6], [8, 10], [15, 18]]
    # prevInterval = [1, 3]
    # curInterval = [2, 6]

    # prevInterval = [1, 6]
    # curInterval = [8, 10]

    # prevInterval = [8, 10]
    # curInterval = [15, 18]

    # prevInterval = [15, 18]
    # curInterval = Nothing

    def merge(self, intervals):
        intervals.sort(key = lambda x: x[0])
        prevInterval = intervals[0]
        res = []
        for i in range(1, len(intervals)):
            curInterval = intervals[i]
            if prevInterval[1] < curInterval[0]:
                res.append(prevInterval)
                prevInterval = curInterval
            else:
                prevInterval[1] = max(prevInterval[1], curInterval[1])
        res.append(prevInterval)
        return res

'''

test cases:

Formula to write test cases :-

5 test cases I can easily find:
1. basic positive
2. basic negative
3. empty input
4. only 1 item in input
5. negative nos in input
6. All items same in input

extra:
6. edge case
7. mixed big and small inputs in magnitude
8. sorted inputs (ascending or descending)
9. portion of positive and portion of negative inputs mixed

test cases for this problem :

1. Basic Overlapping Intervals:

Input: [[1,3],[2,6],[8,10],[15,18]]
Expected Output: [[1,6],[8,10],[15,18]]

2. Non-overlapping Intervals:

Input: [[1,2],[3,4],[5,6]]
Expected Output: [[1,2],[3,4],[5,6]]

3. Empty List of Intervals (Edge Case):

Input: []
Expected Output: []

4. Single Interval (Edge Case):

Input: [[1,3]]
Expected Output: [[1,3]]

5. Intervals with Negative Numbers:

Input: [[-3,-1],[1,2],[0,4]]
Expected Output: [[-3,-1],[0,4]]

6. Intervals Fully Contained Within Other Intervals:

Input: [[1,5],[2,3],[4,6]]
Expected Output: [[1,6]]

7. Intervals with Same Start and End Points:

Input: [[1,4],[4,5]]
Expected Output: [[1,5]]

8. Intervals in Descending Order:

Input: [[4,5],[2,3],[0,1]]
Expected Output: [[0,1],[2,3],[4,5]]

9. Large Intervals with Small Overlaps:

Input: [[1,100],[2,3],[99,200],[150,300]]
Expected Output: [[1,300]]

10. Overlapping and Contiguous Intervals:

Input: [[1,2],[2,3],[3,4],[4,5]]
Expected Output: [[1,5]]
        
'''
```

### Non-overlapping intervals
```python
class Solution:
    # intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
    # output = 1 
    # explanation: [1, 3] can be removed

    # conclusion: [1, 2] and [2, 3] are non-overlapping

    # so which interval to remove?

    # cases:
    # sorted intervals:

    # 0        1      2        3
    # [[1, 2], [1, 3], [2, 3], [3, 4]]

    # intervals 0, 1 and 2 all overlap

    # prevInterval = 1----2
    # currInterval = 1---------3
    # nextInterval =      2----3

    # if we remove prevInterval ie. [1, 2], then [1, 3] still overlaps with [2, 3]
    # if we remove currInterval ie. [1, 3], then [1, 2] does not overlap with [2, 3]

    # case 1: no overlap between prev and curr intervals:
    #     add prevInterval to the result list
    #     make prevInterval = currInterval and continue

    # case 2: overlap exists between prev and curr intervals:
    #     if prevInterval[1] > currInterval[1]:
    #         # remove prevInterval
    #         prevInterval = currInterval
    #     else:
    #         # remove the current interval
    #         # nothing to do
                

    # so, since prevInterval is has to start before current, we just need to remove the interval having the
    # largest end time, so that overlap chances with future intervals is minimized -> hence, less no. of intervals
    # would need removal

    # dry run for code:

    # 0        1      2        3
    # [[1, 2], [1, 3], [2, 3], [3, 4]]

    # prevInterval = [1, 2]
    # currInterval = [1, 3]

    # prevInterval = [3, 4]
    # currInterval = []

    # res = 1


    # eg. 2:
    # [[1, 2], [1, 2], [1, 2]]

    # prevInterval = [1, 2] @idx 0
    # currInterval = [1, 2] @idx 1

    # prevInterval = [1, 2] @idx 0
    # currInterval = [1, 2] @idx 2

    # prevInterval = [1, 2] @idx 0
    # currInterval = None -> loop terminates

    # res = 2

    def eraseOverlapIntervals(self, intervals):
        intervals.sort(key = lambda x: x[0])
        prevInterval = intervals[0]
        res = 0
        for i in range(1, len(intervals)):
            curInterval = intervals[i]
            if prevInterval[1] <= curInterval[0]: # # if no overlap
                prevInterval = curInterval
            else: # else overlap
                prevInterval[1] = min(prevInterval[1], curInterval[1])
                res += 1
        return res


'''
test cases:
Basic Overlapping Intervals:

Input: [[1,3],[2,4],[3,5]]
Expected Output: 1
Explanation: Removing any one of the intervals will make the rest non-overlapping.
Non-overlapping Intervals:

Input: [[1,2],[3,4],[5,6]]
Expected Output: 0
Explanation: All intervals are already non-overlapping, so no need to remove any.
Intervals Fully Contained Within Other Intervals:

Input: [[1,10],[2,3],[4,5],[6,7]]
Expected Output: 1
Explanation: Removing the first interval will make the rest non-overlapping.
Single Interval (Edge Case):

Input: [[1,3]]
Expected Output: 0
Explanation: Only one interval is present, so it's already non-overlapping.
Intervals with Same Start Point:

Input: [[1,3],[1,4],[1,5]]
Expected Output: 2
Explanation: You need to remove two intervals to have only one remaining.
Empty List of Intervals (Edge Case):

Input: []
Expected Output: 0
Explanation: No intervals, so nothing to remove.
Intervals with the Same End Point:

Input: [[1,4],[2,4],[3,4]]
Expected Output: 2
Explanation: You need to remove two intervals to have only one remaining.
Multiple Overlapping Intervals:

Input: [[1,2],[1,3],[1,4],[1,5]]
Expected Output: 3
Explanation: You need to remove three intervals to have only one remaining.
Intervals with Negative Numbers:

Input: [[-3,1],[0,2],[1,3],[2,4]]
Expected Output: 2
Explanation: Removing the first two intervals or the last two will make the rest non-overlapping.
Long Overlapping Intervals:

Input: [[1,10],[2,3],[4,5],[6,7]]
Expected Output: 1
Explanation: Removing the first interval makes all other intervals non-overlapping.


'''
        
```

### Meeting Rooms
```python
class Solution:
    # intervals = [[0, 30], [5, 10], [15, 20]]
    # output = false

    # intervals = [[7, 10], [2, 4]]
    # output = true

    # * to attend all meetings, overlap should not occur between intervals(meetings)
    # -> sort the input intervals according to start times
    # -> check of overlap, that's it !

    def canAttendMeetings(self, intervals):
        intervals.sort(key = lambda x: x[0])
        if len(intervals) == 0:
            return True
        prevInterval = intervals[0]

        for i in range(1, len(intervals)):
            currInterval = intervals[i]
            if prevInterval[1] > currInterval[0]:
                return False
            prevInterval = currInterval
        return True

'''
test cases:

1. basic overlapping:

2. non-overlapping

3. empty list

4. 1 interval only

5. meetings with the same start times

6. meetings with the same end times

7. back to back meetings

8. meetings ordered in descending order

'''

'''
ai generated test cases:

No Overlapping Meetings:

Input: [[0, 30],[35, 40],[45, 50]]
Expected Output: True
Explanation: No meetings overlap, so all can be attended.
Overlapping Meetings:

Input: [[5, 10],[10, 15],[15, 20]]
Expected Output: False
Explanation: Meetings overlap, so not all can be attended.
Single Meeting (Edge Case):

Input: [[10, 20]]
Expected Output: True
Explanation: Only one meeting, so it can be attended.
Empty List of Meetings (Edge Case):

Input: []
Expected Output: True
Explanation: No meetings scheduled, so the person can attend all (none).
Back-to-Back Meetings:

Input: [[1,3],[3,6]]
Expected Output: True
Explanation: Meetings are back-to-back without overlapping.
Long Meeting Followed by Short Meeting:

Input: [[1,8],[2,3]]
Expected Output: False
Explanation: The second meeting overlaps with the first one.
Meetings with the Same Start Time:

Input: [[1,4],[1,5]]
Expected Output: False
Explanation: Both meetings start at the same time.
Meetings with the Same End Time:

Input: [[1,5],[4,5]]
Expected Output: False
Explanation: The first meeting ends when the second one ends.
Meetings Ordered in Descending Start Time:

Input: [[20,30],[15,25],[10,15]]
Expected Output: False
Explanation: The second meeting overlaps with the first and third meetings.
Meetings with Negative Times:

Input: [[-5,0],[0,5]]
Expected Output: True
Explanation: Meetings are back-to-back without overlapping.

pytest:

# Assuming the function to test is named 'canAttendMeetings'
# and is located in a file named 'meeting_rooms.py'

import pytest
from meeting_rooms import canAttendMeetings

# Test cases
@pytest.mark.parametrize("intervals, expected", [
    ([[0, 30],[35, 40],[45, 50]], True),  # No Overlapping Meetings
    ([[5, 10],[10, 15],[15, 20]], False),  # Overlapping Meetings
    ([[10, 20]], True),                    # Single Meeting
    ([], True),                            # Empty List of Meetings
    ([[1,3],[3,6]], True),                 # Back-to-Back Meetings
    ([[1,8],[2,3]], False),                # Long Meeting Followed by Short Meeting
    ([[1,4],[1,5]], False),                # Meetings with Same Start Time
    ([[1,5],[4,5]], False),                # Meetings with Same End Time
    ([[20,30],[15,25],[10,15]], False),    # Meetings in Descending Start Time
    ([[-5,0],[0,5]], True)                 # Meetings with Negative Times
])
def test_canAttendMeetings(intervals, expected):
    assert canAttendMeetings(intervals) == expected
'''
```

### Meeting Rooms II
```python
class Solution:
    # intervals = [[0, 30], [5, 10], [15, 20]]

    # output = 2

    #     0--------------------------------------30
    #                 5-------10
    #                             15----20

    # rooms req:
    # 0       1        2        1       2       1         0   


    # meeting starts: room += 1
    # meeting ends : room -= 1

    #                     i
    # start times = 0, 5, 15
    # end times   = 30, 10, 20
    #         j
    # rooms = 2

    # issue with above: j cannot move further as its pointing to the longest meeting, but we need to decrement no. of rooms when we know a meeting has ended, hence the meeting end times also are needed sorted separately

    #                     i 
    # start times = 0, 5, 15
    # end times   = 10, 20, 30
    #                 j

    # rooms = 2 : no need to iterate j further once i goes out of bounds as i is the one that was starting a meeting
    #         -> for which more rooms were needed, and that's what we are counting

    # maxRooms = 2

    # dry run for code:

    #                         i 
    # start times = 0, 5, 15
    # end times   = 10, 20, 30
    #                 j

    # rooms = 2
    # maxRooms = 2
    
    def minMeetingRooms(self, intervals):
        startTimes = sorted([interval[0] for interval in intervals])
        endTimes = sorted([interval[1] for interval in intervals])

        rooms = 0
        maxi = 0
        n = len(intervals)
        i, j = 0, 0
        while i < n:
            if startTimes[i] < endTimes[j]:
                i += 1
                rooms += 1
            else:
                j += 1
                rooms -= 1
            maxi = max(maxi, rooms)
        return maxi
```

### Minimum Platforms
https://www.geeksforgeeks.org/problems/minimum-platforms-1587115620/1
```python
class Solution:    
    #Function to find the minimum number of platforms required at the
    #railway station such that no train waits.
    # def minimumPlatform(self,n,arr,dep):
    #     startTimes = sorted(arr)
    #     endTimes = sorted(dep)

    #     platforms = 0
    #     maxi = 0
    #     i, j = 0, 0
    #     while i < n:
    #         if startTimes[i] <= endTimes[j]:
    #             i += 1
    #             platforms += 1
    #         else:
    #             j += 1
    #             platforms -= 1
    #         maxi = max(maxi, platforms)
    #     return maxi
        
    # line sweep algo -> time: O(n)
    def minimumPlatform(self,n,arr,dep):
        platform = [0] * 2361 # values = 0 to 2360 (because if departure time = 2359, we need 2359 + 1 = 2360 index to decrement the platform count)
        
        for i in range(n):
            platform[arr[i]] += 1
            platform[dep[i] + 1] -= 1
        
        maxi = 0
        for i in range(1, 2361):
            platform[i] = platform[i] + platform[i - 1]
            maxi = max(maxi, platform[i])
        return maxi
```