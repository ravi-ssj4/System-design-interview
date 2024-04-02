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