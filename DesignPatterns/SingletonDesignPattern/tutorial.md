### Use cases:
1. Database connection class
2. ConfigManager class
3. Logger class

### Logger example in detail
```python

class Logger:
    def __init__(self):
        print("New instance created!")

    def Log(self, msg: str) -> None:
        print(msg)

if __name__ == "__main__":
    logger1 = Logger()
    logger1.Log('This message is from user 1')

    logger2 = Logger()
    logger2.Log('This message is from user 2')

# New instance created!
# This message is from user 1
# New instance created!
# This message is from user 2

# Q. how to count the number of objects/instances created?
# A.


class Logger:
    # no need to mention static here! -> just access it using the Class name
    ctr = 0

    def __init__(self):
        Logger.ctr += 1 # since its truly a class variable(acts as static)
        print("New instance created! No. of instances: ", Logger.ctr)

    def Log(self, msg: str) -> None:
        print(msg)

if __name__ == "__main__":
    logger1 = Logger()
    logger1.Log('This message is from user 1')

    logger2 = Logger()
    logger2.Log('This message is from user 2')
```

### How to restrict users from creating multiple instances?
```python

# 1. Make the constructor private

class Logger:
    
    ctr = 0
    __loggerInstance = None

    # constructor cannot be made private, but still we avoid creation of instance
    # through the constructor in the code below
    def __init__(self): 
        if Logger.__loggerInstance != None:
            raise Exception("This class is singleton!")
        else:
            # If Logger.__loggerInstance is None, it assigns self 
            # (the new instance being created) to Logger.__loggerInstance. 
            # This ensures that the new instance is stored in the class-level 
            # variable, and future attempts to create new instances will be prevented.
            Logger.__loggerInstance = self
            Logger.ctr += 1
            print("New instance created! No. of instances: ", Logger.ctr)
    
    @staticmethod
    def getInstance():
        if Logger.__loggerInstance == None:
            return Logger()
        return Logger.__loggerInstance
        

    def Log(self, msg: str) -> None:
        print(msg)

if __name__ == "__main__":
    logger1 = Logger.getInstance()
    logger1.Log('This message is from user 1')

    logger2 = Logger.getInstance()
    logger2.Log('This message is from user 2')

# Issue with above code: 
# Not thread - safe -> In a multithreaded environment, line 77 can be exected by 2 threads in succession -> let's say, t1 executed 77 and before it could execute 78, t2 also executed 77 -> t2 also enters the if construct -> t1 and t2 both create instances!

# Demonstration:

import threading
import time

class Logger:
    
    ctr = 0
    __loggerInstance = None

    # constructor cannot be made private, but still we avoid creation of instance
    # through the constructor in the code below
    def __init__(self): 
        if Logger.__loggerInstance != None:
            raise Exception("This class is singleton!")
        else:
            # If Logger.__loggerInstance is None, it assigns self 
            # (the new instance being created) to Logger.__loggerInstance. 
            # This ensures that the new instance is stored in the class-level 
            # variable, and future attempts to create new instances will be prevented.
            Logger.__loggerInstance = self
            Logger.ctr += 1
            print("New instance created! No. of instances: ", Logger.ctr)
    
    @staticmethod
    def getInstance():
        # Introduce a deliberate delay to simulate a race condition
        time.sleep(0.5)
        if Logger.__loggerInstance == None:
            return Logger()
        return Logger.__loggerInstance
        

    def Log(self, msg: str) -> None:
        print(msg)

def user1Logs():
    logger1 = Logger.getInstance()
    logger1.Log('This message is from user 1')

def user2Logs():
    logger2 = Logger.getInstance()
    logger2.Log('This message is from user 2')


if __name__ == "__main__":
    t1 = threading.Thread(target=user1Logs)
    t2 = threading.Thread(target=user2Logs)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

### Fixing the issue of thread-safety
```python

@staticmethod
def getInstance():
    with Logger.__lock:  # Ensure that only one thread can execute this block at a time
        if Logger.__loggerInstance is None:
            Logger()
    return Logger.__loggerInstance

# Still there is an issue -> Locks are expensive!
# We don't need to lock unnecessarily after the instance is created once

```

### Fixing the all the time locking -> Double checked lock!
```python
@staticmethod
def getInstance():
    if Logger.__loggerInstance is None:
        with Logger.__lock:  # Ensure that only one thread can execute this block at a time
            if Logger.__loggerInstance is None:
                Logger()
    return Logger.__loggerInstance
```

