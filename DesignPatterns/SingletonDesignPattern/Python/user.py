from logger import Logger
import threading

def user1Logs():
    logger1 = Logger()
    logger1.log("this msg is from user 1")

def user2Logs():
    logger2 = Logger()
    logger2.log("this msg is from user 2")

if __name__ == "__main__":

    t1 = threading.Thread(target=user1Logs)
    t2 = threading.Thread(target=user2Logs)

    t1.start()
    t2.start()

    t1.join()
    t2.join()