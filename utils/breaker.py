import signal
import sys

def dummy_handle():
    pass

handle = dummy_handle
breaks = 0

def setBreakHandle(func_handle):
    global handle
    handle = func_handle

def signal_handler(sig, frame):
    print('Break triggered:', sig, frame)
    handle()

    global breaks

    breaks += 1
    if breaks == 3:
        print("Third break triggered, exitting.")
        sys.exit(0)
    
    
signal.signal(signal.SIGINT, signal_handler)