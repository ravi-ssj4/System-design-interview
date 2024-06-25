import threading

class Logger:
    ctr = 0
    __loggerInstance = None
    __lock = threading.Lock()  # Lock for thread safety

    def __init__(self):
        if Logger.__loggerInstance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__loggerInstance = self
            Logger.ctr += 1
            print("New instance created! No. of instances: ", Logger.ctr)

    @staticmethod
    def getInstance():
        if Logger.__loggerInstance is None:
            with Logger.__lock:  # Ensure that only one thread can execute this block at a time
                if Logger.__loggerInstance is None:
                    Logger()
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