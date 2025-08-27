import datetime
import pytz
from typing import Tuple, List, Dict, Callable, Any
import os
import http.server
import socketserver
import pandas as pd
import itertools
from collections import deque, defaultdict

from .prompts import conversationToString

## Stuff for keypoller support on windows
isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios

# this is needed because vllm doesn't like being interrupted with ctrl-c
# so I listen for the c key and if it's sent then we can interrupt
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

# def getModels() -> Tuple[vllm.LLM, SentenceTransformer]:
#     """Get the default models we use (llm, embeddingModel) for running clio"""
#     #model_str = "Qwen/Qwen2.5-7B-Instruct"
#     model_str = "Qwen/Qwen3-8B"
#     llm = vllm.LLM(model=model_str)
#     embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#     return llm, embeddingModel

def filterDataToEnglish(data : List[List[Dict[str,str]]]) -> List[List[Dict[str,str]]]:
    """Simple filter function that restricts us to only data that has english on all turns"""
    return [conversation for conversation in data if all([turn['language'] == 'English' for turn in conversation])]

def dedup(data: List[List[Dict[str, str]]],
        dedupKeyFunc: Callable[[Any], Any],
        batchSize: int,
        verbose: bool,
        returnMapping: bool=False):
    """Deduplicates the given data, using dedupKeyFunc as item keys, processing batchSize elements at a time"""
        
    existingConvs = {}
    dedupedConvs = []
    mapping = {}
    def processOutputFunc(dataI, s, dataKey):
        if not dataKey in existingConvs:
            existingConvs[dataKey] = len(dedupedConvs)
            mapping[dataI] = len(dedupedConvs)
            dedupedConvs.append(data[dataI])
        else:
            mapping[dataI] = existingConvs[dataKey]

    runBatched(list(range(len(data))),
        getInputs=lambda dataI: dedupKeyFunc(data[dataI]),
        processBatch=lambda dataKeys: dataKeys,
        processOutput=processOutputFunc,
        batchSize=batchSize,
        verbose=verbose)
    
    if returnMapping:
        return dedupedConvs, mapping
    else:
        return dedupedConvs

def getExampleData():
    """Extracts some example data for parsing"""
    dataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildchatsubset.parquet")
    subset = pd.read_parquet(dataPath, engine="pyarrow")
    return [[x for x in subset.iloc[i] if not x is None] for i in range(len(subset))]

def getFullWildchatData(rootPath):
    """Extracts all wildchat data stored in the given directory (they should look like train-000____.parquet)"""
    d = []
    for l in sorted(os.listdir(rootPath))[::-1]: # sort and reverse so reproducable ordering
        if l.startswith("train-000"):
            print(l)
            subset = pd.read_parquet(os.path.join(rootPath, l), engine="pyarrow")
            d += [subset.iloc[i].conversation for i in range(len(subset))]
    return filterDataToEnglish(d)

def timestampMillis() -> int:
    """Get current timestamp in millis"""
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000) 

def getFutureDatetime(seconds_to_add : float) -> datetime.datetime:
    """Datetime after we add seconds_to_add seconds, in local time"""
    # Get current datetime (adjust this to yours if you want)
    current_datetime = datetime.datetime.now(pytz.timezone('US/Pacific'))
    
    # Calculate future datetime by adding seconds
    future_datetime = current_datetime + datetime.timedelta(seconds=seconds_to_add)
    
    return future_datetime

def convertSeconds(seconds) -> Tuple[int, int, int, int]:
    """Calculate (days, hours, minutes, seconds)"""
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    
    # Return as a tuple (days, hours, minutes, seconds)
    return int(days), int(hours), int(minutes), int(seconds)

def secondsToDisplayStr(seconds : float) -> str:
    """Display seconds as days, hours, minutes, seconds"""
    day, hour, mins, sec = convertSeconds(seconds)
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


def flatten(nestedLists):
    """"
    Flattens an array into a 1D array
    For example
    # [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
    # is flattened into
    # [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
    """
    result = []
    if type(nestedLists) is list:
        for n in nestedLists:
            result += flatten(n)
    else:
        result.append(nestedLists)
    return result


def unflatten(unflattened, nestedLists):
    """
    Once you do
    originalUnflattened = [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
    flattened = flatten(originalUnflattened)
    # [2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]
    say you have another list of len(flattened)
    transformed = [3, 4, 5, 4, 5, 6, 7, 3, 4, 3, 5, 4, 4]
    this can "unflatten" that list back into the same shape as originalUnflattened
    unflattenedTransformed = unflatten(transformed, originalUnflattened)
    # [[[3, 4], [5, [4, 5], 6, 7], 3, 4], [3, 5], [4], 4]
    """
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

def runBatched(inputs, getInputs, processBatch, processOutput, batchSize, verbose=True, noCancel=False):
    """
    Utility function that's useful to do batched processing on structured data.

    inputs should be a list of the data you want to process

    It does the following:
    1. Convert each input into (arbitrairly nested, as much as you'd like) arrays using getInputs(input)
    2. Flattens the results of all of those
    3. Passes chunks of size batchSize into processBatch(flattenedBatch)
        Each processBatch call should return as many values as it was given as input.
        The very final call may be smaller than batchSize if things don't evenly divide
    4. Unflattens them back to original structure provided via getInputs, then
    5. Calls processOutput(input, outputFromGetInputs, resultsFromProcessBatch) for each input
        resultsFromProcessBatch will have same nesting structure as outputFromGetInputs
        (so if getInputs returned [["hi"], "there"] then 
        outputFromGetInputs will be [["hi"], "there"] and
        resultsFromProcessBatch will look like [[result1], result2])
    6. Returns an array that has the outputs of processOutput (one entry per input)

    That's the process, but it actually does this in a "streaming" fashion so it only grabs stuff as needed.

    However it'll still return a list of the outputs, if you prefer to iterate through the outputs and not keep them all in memory,
    you can use runBatchedIterator instead
    """
    return list(runBatchedIterator(
        inputs=inputs,
        n=len(inputs),
        getInputs=getInputs,
        processBatch=processBatch,
        processOutput=processOutput,
        batchSize=batchSize,
        noCancel=noCancel,
        verbose=verbose,
    ))

def runBatchedIterator(inputs, n, getInputs, processBatch, processOutput, batchSize, verbose=True, noCancel=False):
    """
    See documentation for runBatched, the main difference is that this will "stream" the outputs as needed instead of putting them all in memory in a big array before returning.
    Also, inputs can be an enumerator if desired.
    Because we no longer know the length of inputs, we require the n parameter which is the length of inputs.
    """
    def getInputsIterator(inputs):
        for input in inputs:
            yield getInputs(input)
            
    def getFlattenedIterator(inputsIter):
        for unflattenedInputs in inputsIter:
            yield flatten(unflattenedInputs)
            
    def getFlattenedOutputsIterator(flattenedIter, runOnBatchFunc):
        curBatch = deque() # this gives us o(1) insertions and removals
        batchEnd = 0
        for flattened in flattenedIter:
            curBatch.extend(flattened)
            while len(curBatch) >= batchSize:
                outputs = processBatch([curBatch.popleft() for _ in range(batchSize)])
                batchEnd += batchSize
                runOnBatchFunc(batchEnd)
                yield outputs
        if len(curBatch) > 0:
            outputs = processBatch(list(curBatch))
            batchEnd += len(curBatch)
            runOnBatchFunc(batchEnd)
            yield outputs

    def onDemandBatchedIter(inputs, runOnBatchFunc):
        nonlocal n
        # tee makes two iterators that share the same source, so we only call getInputs once for each item
        # it's nice that it only stores past stuff until consumed by both (plus a small buffer, depending on implementation)
        inputsIter1, inputsIter2 = itertools.tee(getInputsIterator(inputs))
        flattenedIter1, flattenedIter2 = itertools.tee(getFlattenedIterator(inputsIter1))
        flattenedOutputsIter = getFlattenedOutputsIterator(flattenedIter1, runOnBatchFunc)

        curOutputs = deque() # this gives us o(1) insertions and removals
        for i, (input, inputUnflattened, inputFlattened) in enumerate(zip(inputs, inputsIter2, flattenedIter2)):
            if i == 0: n *= len(inputFlattened) # improve estimate of n
            # fetch outputs until we have as many as we sent in inputs
            while len(curOutputs) < len(inputFlattened):
                curOutputs.extend(next(flattenedOutputsIter))
            # grab that many and unflatten them (make them the shape of inputUnflattened)
            outputsUnflattened = unflatten([curOutputs.popleft() for _ in range(len(inputFlattened))], inputUnflattened)
            # process the outputs and return them
            results = processOutput(input, inputUnflattened, outputsUnflattened)
            yield results

    startTime = timestampMillis()
    # we need keypoller because vllm doen't like to be keyboard interrupted
    with KeyPoller(noCancel) as keypoller:
        def runOnBatchedFunc(batchEnd):
            elapsed = timestampMillis() - startTime
            secondsPerPrompt = elapsed / (float(batchEnd))
            totalTime = elapsed *  n / float(batchEnd)
            timeLeft = totalTime - elapsed
            dispStr = secondsToDisplayStr(timeLeft/1000.0)
            doneDateTimeStr = getFutureDatetime(timeLeft/1000.0).strftime('%I:%M:%S %p')
            if verbose:
                print(batchEnd, "/", n, f"{secondsPerPrompt} millis per item {dispStr}done at {doneDateTimeStr}")
            keys = keypoller.poll()
            if not keys is None:
                print(keys)
                if str(keys) == "c":
                    print("got c")
                    raise ValueError("stopped")   
        
        for output in onDemandBatchedIter(inputs, runOnBatchedFunc):
            yield output
                

def getClosestNames(
    names: List[str],
    embeddingModel
    ) -> Tuple[int, int, float]:
    """
    Get the pair of names that have closest embeddings when using embeddingModel
    Returns (pairI, pairJ, pairCosineSimilarity)
    """
    embedded = embeddingModel.encode(names, show_progress_bar=False)
    sims = cosine_similarity(embedded)
    # middle is 0, make it not
    np.fill_diagonal(sims, -1)
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    print(sims[i,j])
    print(names[i])
    print(names[j])
    return i,j, sims[i,j]


def getDuplicateFacetValues(
        facetValues: List['ConversationFacetData'],
        facetName: str,
        conversations: List[Dict[str, str]],
        llm,
        maxConversationTokens: int
    ):
    """
    Utility method if u want to find [(facetValue, conversationIndicesOfFacetValue, allDuplicateValues)]
    Helpful for debugging when there's too many duplicates
    """
    counts = defaultdict(lambda: [])
    uniques = defaultdict(lambda: set())
    tokenizer = llm.get_tokenizer()
    for conversationI, conversationFacetValues in enumerate(facetValues):
        if conversationI % 1000 == 0: print(conversationI)
        for facetValue in conversationFacetValues.facetValues:
            if facetValue.facet.name == facetName:
                counts[facetValue.value].append(conversationI)
    
    dups = sorted([(k,vs) for (k,vs) in counts.items() if len(vs) > 1], key=lambda x: -len(x[1]))
    for k,vs in dups:
        for conversationI in vs:
            uniques[k].add(conversationToString(conversations[conversationI], tokenizer=tokenizer, maxTokens=maxConversationTokens))
    return [(k,vs,uniques[k]) for (k,vs) in dups]

def runWebui(path, port):
    """
    Runs a simple http server at the given path, using the given port
    """
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=path, **kwargs)
    with KeyPoller() as keypoller:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving at http://localhost:{port}")
            while True:
                httpd.timeout = 0.5          # seconds – how long handle_request() can block
                keys = keypoller.poll()
                if not keys is None:
                    print(keys)
                    if str(keys) == "c":
                        print("got c")
                        raise ValueError("stopped")   
                httpd.handle_request()   # serves at most one request