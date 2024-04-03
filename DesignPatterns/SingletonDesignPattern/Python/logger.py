import threading

class Logger:
    _instance = None
    _lock = threading.Lock()
    ctr = 0

    def __new__(self):
        if self._instance == None:
            with self._lock:
                if self._instance == None:
                    self._instance = super(Logger, self).__new__(self)
                    self.ctr += 1
                    print(f"Logger instance created. No. of instances = {self.ctr}")
        return self._instance
    
    def log(self, msg):
        print(msg)
