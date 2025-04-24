import datetime
import pytz
isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios

class KeyPoller():
    def __init__(self, noCancel=False):
        self.noCancel = noCancel

    def __enter__(self):
        if self.noCancel: return self
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT)
            
            self.curEventLength = 0
            self.curKeysLength = 0
            
            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            
        return self
    
    def __exit__(self, type, value, traceback):
        if self.noCancel: return
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
    
    def poll(self):
        if self.noCancel: return None
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None
        
def timestampMillis():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000) 

'''
Datetime after we add seconds_to_add seconds, in local time
'''
def getFutureDatetime(seconds_to_add):
    # Get current datetime (adjust this to yours if you want)
    current_datetime = datetime.datetime.now(pytz.timezone('US/Pacific'))
    
    # Calculate future datetime by adding seconds
    future_datetime = current_datetime + datetime.timedelta(seconds=seconds_to_add)
    
    return future_datetime

'''
Calculate (days, hours, minutes, seconds)
'''
def convertSeconds(seconds):
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    
    # Return as a tuple (days, hours, minutes, seconds)
    return int(days), int(hours), int(minutes), int(seconds)

'''
Display seconds as days, hours, minutes, seconds
'''
def secondsToDisplayStr(seconds):
    day, hour, mins, sec = convertSeconds()
    dispStr = ""
    if day > 0:
        dispStr += f"{round(day)} day{'s' if round(day) > 1 else ''}  "
    if hour > 0:
        dispStr += f"{round(hour)} hour{'s' if round(hour) > 1 else ''} "
    if mins > 0:
        dispStr += f"{round(mins)} minute{'s' if round(mins) > 1 else ''} "
    if sec > 0:
        dispStr += f"{round(sec)} second{'s' if round(sec) > 1 else ''} "
    return dispStr


'''
Flattens an array into a 1D array
For example
# [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
# is flattened into
# [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
'''
def flatten(nestedLists):
    result = []
    if type(nestedLists) is list:
        for n in nestedLists:
            result += flatten(n)
    else:
        result.append(nestedLists)
    return result

'''
Once you do
originalUnflattened = [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
flattened = flatten(originalUnflattened)
# [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
say you have another list of len(flattened)
transformed = [3, 4, 5, 4, 5, 6, 7, 3, 4, 3, 5, 4, 4]
this can "unflatten" that list back into the same shape as originalUnflattened
unflattenedTransformed = unflatten(transformed, originalUnflattened)
# [[[3, 4], [5, [4, 5], 6, 7], 3, 4], [3, 5], [4], 4]
'''
def unflatten(unflattened, nestedLists):
    result, endIndex = unflattenHelper(unflattened, nestedLists, 0)
    return result

def unflattenHelper(unflattened, nestedLists, startIndex):
    if type(nestedLists) is list:
        result = []
        for n in nestedLists:
            resultSubArray, startIndex = unflattenHelper(unflattened, n, startIndex=startIndex)
            result.append(resultSubArray)
    else:
        result = unflattened[startIndex]
        startIndex += 1
    return result, startIndex


def runBatched(inputs, getInputs, processBatch, processOutput, batchSize, noCancel=False):
    unflattenedInputs = [getInputs(input) for input in inputs]
    flattenedInputs = flatten(unflattenedInputs)
    def batchedFunc(startBatch, endBatch):
        batchInputs = flattenedInputs[startBatch:endBatch]
        return processBatch(batchInputs)
    flattenedOutputs = runBatchedSimple(batchedFunc, len(flattenedInputs), batchSize, noCancel=noCancel)
    unflattenedOutputs = unflatten(flattenedOutputs, unflattenedInputs)
    results = []
    for input, modelInputs, output in zip(inputs, unflattenedInputs, unflattenedOutputs):
        results.append(processOutput(input, modelInputs, output))
    return results

def runBatchedSimple(callFunc, n, batchSize, noCancel=False):
    outputs = []
    startTime = timestampMillis()
    # we need keypoller because vllm doen't like to be keyboard interrupted
    with KeyPoller(noCancel) as keypoller:
        for batchStart in range(0, n, batchSize):
            batchEnd = min(n, batchStart+batchSize)
            outputs += callFunc(batchStart, batchEnd)
            elapsed = timestampMillis() - startTime
            secondsPerPrompt = elapsed / (float(batchEnd))
            totalTime = elapsed *  n / float(batchEnd)
            timeLeft = totalTime - elapsed
            dispStr = secondsToDisplayStr(timeLeft/1000.0)
            doneDateTimeStr = getFutureDatetime(timeLeft/1000.0).strftime('%I:%M:%S %p')
            print(batchEnd, "/", n, f"{secondsPerPrompt} millis per item {dispStr}done at {doneDateTimeStr}")
            keys = keypoller.poll()
            if not keys is None:
                print(keys)
                if str(keys) == "c":
                    print("got c")
                    raise ValueError("stopped")
    return outputs