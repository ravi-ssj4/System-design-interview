## Race Condition:

```python
# It is a bug in multithreading.
# Two or more threads try to access the same variable/shared memory
# at the same time resulting in unreliable outputs 

# toy example
from threading import *
import time

def snippet(myLock, msg):
    
    # myLock.acquire()
    
    for _ in range(5):
        print(msg)
    time.sleep(3)

    # myLock.release()

myLock = Lock()

t1 = Thread(target=snippet, args=(myLock, "Hello world"))
t2 = Thread(target=snippet, args=(myLock, "Welcome world"))
t1.start()
t2.start()

# PS D:\Ravi\2023\Interview> python toy.py
# Hello world
# Welcome world
# Welcome world
# Welcome world
# Welcome world
# Welcome world
# Hello world
# Hello world
# Hello world
# Hello world

from threading import *
import time

def snippet(myLock, msg):
    
    myLock.acquire()
    
    for _ in range(5):
        print(msg)
    time.sleep(3)

    myLock.release()

myLock = Lock()

t1 = Thread(target=snippet, args=(myLock, "Hello world"))
t2 = Thread(target=snippet, args=(myLock, "Welcome world"))
t1.start()
t2.start()

# PS D:\Ravi\2023\Interview> python toy.py
# Hello world
# Hello world
# Hello world
# Hello world
# Hello world
# Welcome world
# Welcome world
# Welcome world
# Welcome world
# Welcome world

# Bus Ticketing System
from threading import *

class Bus:
    def __init__(self, name, available_seats):
        self.available_seats = available_seats
    
    def reserve(self, seats_needed):
        print("Available seats are: ", self.available_seats)
        if seats_needed < self.available_seats:
            nm = current_thread().name
            print(seats_needed, "are allocated to ", nm)
            self.available_seats -= seats_needed
        else:
            print("Sorry, enough seats are not available!")


b1 = Bus("Ravi's Bus company", 2)

t1 = Thread(target=b1.reserve, args=(1,), name="Rohan")
t2 = Thread(target=b1.reserve, args=(1,), name="Jakaria")

t1.start()
t2.start()

# Issue: 
# The two threads t1 and t2 are accessing self.available_seats variable at the same time! -> Race Condition

# PS D:\Ravi\2023\Interview> python .\main.py
# Available seats are:  2
# 1 are allocated to  Rohan
# Available seats are:  2
# Sorry, enough seats are not available!

# How to avoid race condition?

# Thread Synchronization technique:
# Protect the critical section of the code (logic where the value of self.available_seats is updated)
# How?
# 1. Locks
# 2. R-locks
# 3. Semaphores

from threading import *
myLock = Lock()

class Bus:
    def __init__(self, name, available_seats, lock):
        self.available_seats = available_seats
        self.name = name
        self.lock = lock

    def reserve(self, seats_needed):
        self.lock.acquire()
        print("Available seats are: ", self.available_seats)
        if seats_needed <= self.available_seats:
            nm = current_thread().name
            print(seats_needed, "are allocated to ", nm)
            self.available_seats -= seats_needed
        else:
            print("Sorry, enough seats are not available!")
        self.lock.release()



b1 = Bus("Ravi's Bus company", 2, myLock)
t1 = Thread(target=b1.reserve, args=(2,), name="Rohan")
t2 = Thread(target=b1.reserve, args=(1,), name="Jakaria")

t1.start()
t2.start()


```

### Issue with using Locks for thread sync.
```python
from threading import *
myLock = Lock()

class Bus:
    def __init__(self, name, available_seats, lock):
        self.available_seats = available_seats
        self.name = name
        self.lock = lock

    def reserve(self, seats_needed):
        self.lock.acquire()
        self.lock.acquire()
        print("Available seats are: ", self.available_seats)
        if seats_needed <= self.available_seats:
            nm = current_thread().name
            print(seats_needed, "are allocated to ", nm)
            self.available_seats -= seats_needed
        else:
            print("Sorry, enough seats are not available!")
        self.lock.release()
        self.lock.release()


b1 = Bus("Ravi's Bus company", 2, myLock)
t1 = Thread(target=b1.reserve, args=(2,), name="Rohan")
t2 = Thread(target=b1.reserve, args=(1,), name="Jakaria")

t1.start()
t2.start()

```

### Multiple locks can be acquired via R-Lock:
```python
from threading import *
myLock = RLock()


class Bus:
    def __init__(self, name, available_seats, rlock):
        self.available_seats = available_seats
        self.name = name
        self.rlock = rlock

    def reserve(self, seats_needed):
        self.rlock.acquire()
        self.rlock.acquire()
        print("Available seats are: ", self.available_seats)
        if seats_needed <= self.available_seats:
            nm = current_thread().name
            print(seats_needed, "are allocated to ", nm)
            self.available_seats -= seats_needed
        else:
            print("Sorry, enough seats are not available!")
        self.rlock.release()
        self.rlock.release()


b1 = Bus("Ravi's Bus company", 2, myLock)
t1 = Thread(target=b1.reserve, args=(2,), name="Rohan")
t2 = Thread(target=b1.reserve, args=(1,), name="Jakaria")

t1.start()
t2.start()
```

### Practical use case of RLock:
```python
from threading import *
l = RLock()

def get_first_line():
    l.acquire()
    print("Fetching first line from file")
    l.release()

def get_second_line():
    l.acquire()
    print("Fetching second line from file")
    l.release()

def main():
    l.acquire()
    get_first_line()
    get_second_line()
    l.release()

t1 = Thread(target=main)
t2 = Thread(target=main)
t1.start()
t2.start()

```

### Semaphores:
```python
from threading import *
import time
obj = Semaphore(3)

def display(name):
    
    obj.acquire() #acquire: decrement count

    for _ in range(5):
        print("Hello")
        print(name)
        time.sleep(3)
    
    obj.release() # release: increment count

t1 = Thread(target=display, args=('Thread-1',))
t2 = Thread(target=display, args=('Thread-2',))
t3 = Thread(target=display, args=('Thread-3',))
t4 = Thread(target=display, args=('Thread-4',))
t5 = Thread(target=display, args=('Thread-5',))
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

# Issue: if we have multiple acquire or release locks, 
# value of semaphore can become anything

# To be safe from this, use BoundedSemaphore

```
### Simple example of threading:
```python
from threading import *
import time
obj = BoundedSemaphore(3)

def display(name):
    
    obj.acquire() #acquire: decrement count

    for _ in range(5):
        print("Hello")
        print(name)
        time.sleep(3)
    
    obj.release() # release: increment count

t1 = Thread(target=display, args=('Thread-1',))
t2 = Thread(target=display, args=('Thread-2',))
t3 = Thread(target=display, args=('Thread-3',))
t4 = Thread(target=display, args=('Thread-4',))
t5 = Thread(target=display, args=('Thread-5',))
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
```