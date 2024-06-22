### Next Closest Time | Google
```python
'''
Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.

 

Example 1:

Input: time = "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.
It is not 19:33, because this occurs 23 hours and 59 minutes later.
Example 2:

Input: time = "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22.
It may be assumed that the returned time is next day's time since it is smaller than the input time numerically.
 

Constraints:

time.length == 5
time is a valid time in the form "HH:MM".
0 <= HH < 24
0 <= MM < 60
'''

class Solution:
    def nextClosestTime(self, time: str) -> str:
        hashSet = set()
        curTime = 0
        mul = 10
        for c in time:
            if c != ":":
                hashSet.add(int(c))
                curTime = curTime * mul + int(c)

        minutes = curTime // 100 * 60 + curTime % 100

        nextTime = minutes
        while True:
            nextTime = (nextTime + 1) % (24 * 60)
            

            isValid = True
            newDigits = [nextTime // 60 // 10, nextTime // 60 % 10, nextTime % 60 // 10, nextTime % 60 % 10]
            for digit in newDigits:
                if digit not in hashSet:
                    isValid = False
            
            if isValid == True:
                break

        newTime = ""
        for digit in newDigits:
            if len(newTime) == 2:
                newTime += ":"
            newTime += str(digit)
        return newTime

```