# Create only 1 instance/object of a class for the entire app/product
## Examples of Singleton design pattern:
1. Db Connection - only 1 db object should be there
2. Config Manager - should be only 1 config manager
3. Logger - only 1 logger should be there as all the logs should be stored at one place

# Example we talk about: Logger

## Initial setup
logger.hpp:
```cpp
#ifndef logger_h
#define logger_h

#include <string>
using namespace std;

class Logger {
public:
    Logger();
    void Log(string msg);
};


# endif
```
logger.cpp:
```cpp
#include <iostream>
#include "logger.hpp"
using namespace std;


Logger::Logger() {
    cout << "Logger instance created!" << endl;
}

void Logger::Log(string msg) {
    cout << msg << endl;
}
```
user.cpp
```cpp
#include <iostream>
#include "logger.hpp"

using namespace std;

int main() {

    Logger* logger1 = new Logger();
    logger1->Log("this msg is from user 1");

    Logger* logger2 = new Logger();
    logger2->Log("this msg is from user 2");
    
    return 0;
}
```
output:
```
Logger instance created!
this msg is from user 1
Logger instance created!
this msg is from user 2
```
Now, we will put logic to count the number of instances of the Logger:
<br>=> Using a static variable in Logger class

logger.hpp:
```cpp
#ifndef logger_h
#define logger_h

#include <string>
using namespace std;

class Logger {
    static int ctr;
public:
    Logger();
    void Log(string msg);
};


# endif
```
logger.cpp
```cpp
#include <iostream>
#include "logger.hpp"
using namespace std;

int Logger::ctr = 0;

Logger::Logger() {
    ctr++;
    cout << "Logger instance created! No. of instances: " << ctr << endl;
}

void Logger::Log(string msg) {
    cout << msg << endl;
}
```
output:
```
Logger instance created! No. of instances: 1
this msg is from user 1
Logger instance created! No. of instances: 2
this msg is from user 2
```

Now, we want to take away the power to create instances of the Logger away from the user. How? <br> => make the constructor of Logger private!
logger.hpp
```cpp
#ifndef logger_h
#define logger_h

#include <string>
using namespace std;

class Logger {
    static int ctr;
    static Logger *loggerInstance;
    Logger();
public:
    void Log(string msg);
    static Logger *getLogger();
};


# endif
```
logger.cpp
```cpp
#include <iostream>
#include "logger.hpp"
using namespace std;

int Logger::ctr = 0;

Logger *Logger::loggerInstance = nullptr;

Logger::Logger() {
    ctr++;
    cout << "Logger instance created! No. of instances: " << ctr << endl;
}

void Logger::Log(string msg) {
    cout << msg << endl;
}

Logger *Logger::getLogger() {
    
    if (loggerInstance == nullptr) 
    {
        loggerInstance = new Logger();
    }
    
    return loggerInstance;
}
```
user.cpp
```cpp
#include <iostream>
#include "logger.hpp"

using namespace std;

int main() {

    Logger* logger1 = Logger::getLogger();
    logger1->Log("this msg is from user 1");

    Logger* logger2 = Logger::getLogger();
    logger2->Log("this msg is from user 2");
    
    return 0;
}


```
output:
```
Logger instance created! No. of instances: 1
this msg is from user 1
this msg is from user 2
```

Q. Is this code thread safe? <br>
A. No

```cpp
#include <iostream>
#include "logger.hpp"
#include <thread>

using namespace std;

void user1Logs() {
    Logger* logger1 = Logger::getLogger();
    logger1->Log("this msg is from user 1");
}

void user2Logs() {
    Logger* logger2 = Logger::getLogger();
    logger2->Log("this msg is from user 2");
}

int main() {
    thread t1(user1Logs);
    thread t2(user2Logs);

    t1.join();
    t2.join();
    return 0;
}


```
Issue: both threads are pretty fast and hit the function Logger::getLogger()'s first line one after the other <br> => if (loggerInstance == nullptr) => hence, both create the object!

This is typical example of race condition for a piece of code snippet!
<br> Solution: Lock that piece of code!, so that only 1 thread can access it at a time!

```cpp
#ifndef logger_h
#define logger_h

#include <string>
#include <mutex>

using namespace std;

class Logger {
    static int ctr;
    static Logger *loggerInstance;
    static mutex mtx;
    Logger();
public:
    void Log(string msg);
    static Logger *getLogger();
};


# endif
```

```cpp
#include <iostream>
#include "logger.hpp"
using namespace std;

int Logger::ctr = 0;

Logger *Logger::loggerInstance = nullptr;

mutex Logger::mtx;

Logger::Logger() {
    ctr++;
    cout << "Logger instance created! No. of instances: " << ctr << endl;
}

void Logger::Log(string msg) {
    cout << msg << endl;
}

Logger *Logger::getLogger() {
    
    mtx.lock();
    if (loggerInstance == nullptr) 
    {
        loggerInstance = new Logger();
    }
    mtx.unlock();
    
    return loggerInstance;
}
```

Q. Is everything fine now?
<br>A. No, because:
<br>-> everytime getLogger is called by any user, locks are being created!
<br>-> locks are expensive

Solution: Double Checked logging

```cpp
#include <iostream>
#include "logger.hpp"
using namespace std;

int Logger::ctr = 0;

Logger *Logger::loggerInstance = nullptr;

mutex Logger::mtx;

Logger::Logger() {
    ctr++;
    cout << "Logger instance created! No. of instances: " << ctr << endl;
}

void Logger::Log(string msg) {
    cout << msg << endl;
}

Logger *Logger::getLogger() {
    // double checked logging
    if (loggerInstance == nullptr) {
        mtx.lock();
        if (loggerInstance == nullptr) 
        {
            loggerInstance = new Logger();
        }
        mtx.unlock();
    }
    
    return loggerInstance;
}
```

Q. Is everything fine now?
<br>A. No, still there is an issue:
1. The default constructor is not the only way we can call constructors!
    -> Need to make copy constructor as private as well
2. We need to make the = operator overloading also private!

```cpp
#ifndef logger_h
#define logger_h

#include <string>
#include <mutex>

using namespace std;

class Logger {
    static int ctr;
    static Logger *loggerInstance;
    static mutex mtx;
    Logger();
    Logger(const Logger &);
    Logger operator=(const Logger &);
public:
    void Log(string msg);
    static Logger *getLogger();
};


# endif


```