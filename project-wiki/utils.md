# Table of Contents

* [utils](#utils)
  * [getModels](#utils.getModels)
  * [filterDataToEnglish](#utils.filterDataToEnglish)
  * [dedup](#utils.dedup)
  * [getExampleData](#utils.getExampleData)
  * [getFullWildchatData](#utils.getFullWildchatData)
  * [timestampMillis](#utils.timestampMillis)
  * [getFutureDatetime](#utils.getFutureDatetime)
  * [convertSeconds](#utils.convertSeconds)
  * [secondsToDisplayStr](#utils.secondsToDisplayStr)
  * [flatten](#utils.flatten)
  * [unflatten](#utils.unflatten)
  * [runBatched](#utils.runBatched)
  * [runBatchedIterator](#utils.runBatchedIterator)
  * [getClosestNames](#utils.getClosestNames)
  * [getDuplicateFacetValues](#utils.getDuplicateFacetValues)
  * [runWebui](#utils.runWebui)

<a id="utils"></a>

# utils

<a id="utils.getModels"></a>

#### getModels

```python
def getModels() -> Tuple[vllm.LLM, SentenceTransformer]
```

Get the default models we use (llm, embeddingModel) for running clio

<a id="utils.filterDataToEnglish"></a>

#### filterDataToEnglish

```python
def filterDataToEnglish(
        data: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]
```

Simple filter function that restricts us to only data that has english on all turns

<a id="utils.dedup"></a>

#### dedup

```python
def dedup(data: List[List[Dict[str, str]]], dedupKeyFunc: Callable[[Any], Any],
          batchSize: int, verbose: bool)
```

Deduplicates the given data, using dedupKeyFunc as item keys, processing batchSize elements at a time

<a id="utils.getExampleData"></a>

#### getExampleData

```python
def getExampleData()
```

Extracts some example data for parsing

<a id="utils.getFullWildchatData"></a>

#### getFullWildchatData

```python
def getFullWildchatData(rootPath)
```

Extracts all wildchat data stored in the given directory (they should look like train-000____.parquet)

<a id="utils.timestampMillis"></a>

#### timestampMillis

```python
def timestampMillis() -> int
```

Get current timestamp in millis

<a id="utils.getFutureDatetime"></a>

#### getFutureDatetime

```python
def getFutureDatetime(seconds_to_add: float) -> datetime.datetime
```

Datetime after we add seconds_to_add seconds, in local time

<a id="utils.convertSeconds"></a>

#### convertSeconds

```python
def convertSeconds(seconds) -> Tuple[int, int, int, int]
```

Calculate (days, hours, minutes, seconds)

<a id="utils.secondsToDisplayStr"></a>

#### secondsToDisplayStr

```python
def secondsToDisplayStr(seconds: float) -> str
```

Display seconds as days, hours, minutes, seconds

<a id="utils.flatten"></a>

#### flatten

```python
def flatten(nestedLists)
```

"
Flattens an array into a 1D array
For example
__[[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]__

__is flattened into__

__[2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]__


<a id="utils.unflatten"></a>

#### unflatten

```python
def unflatten(unflattened, nestedLists)
```

Once you do
originalUnflattened = [[[2, 3], [4, [3, 4], 5, 6], 2, 3], [2, 4], [3], 3]
flattened = flatten(originalUnflattened)
__[2, 3, 4, 3, 4, 5, 6, 2, 3, 2, 4, 3, 3]__

say you have another list of len(flattened)
transformed = [3, 4, 5, 4, 5, 6, 7, 3, 4, 3, 5, 4, 4]
this can "unflatten" that list back into the same shape as originalUnflattened
unflattenedTransformed = unflatten(transformed, originalUnflattened)
__[[[3, 4], [5, [4, 5], 6, 7], 3, 4], [3, 5], [4], 4]__


<a id="utils.runBatched"></a>

#### runBatched

```python
def runBatched(inputs,
               getInputs,
               processBatch,
               processOutput,
               batchSize,
               verbose=True,
               noCancel=False)
```

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

<a id="utils.runBatchedIterator"></a>

#### runBatchedIterator

```python
def runBatchedIterator(inputs,
                       n,
                       getInputs,
                       processBatch,
                       processOutput,
                       batchSize,
                       verbose=True,
                       noCancel=False)
```

See documentation for runBatched, the main difference is that this will "stream" the outputs as needed instead of putting them all in memory in a big array before returning.
Also, inputs can be an enumerator if desired.
Because we no longer know the length of inputs, we require the n parameter which is the length of inputs.

<a id="utils.getClosestNames"></a>

#### getClosestNames

```python
def getClosestNames(
        names: List[str],
        embeddingModel: SentenceTransformer) -> Tuple[int, int, float]
```

Get the pair of names that have closest embeddings when using embeddingModel
Returns (pairI, pairJ, pairCosineSimilarity)

<a id="utils.getDuplicateFacetValues"></a>

#### getDuplicateFacetValues

```python
def getDuplicateFacetValues(facetValues: List['ConversationFacetData'],
                            facetName: str, conversations: List[Dict[str,
                                                                     str]],
                            llm: vllm.LLM, maxConversationTokens: int)
```

Utility method if u want to find [(facetValue, conversationIndicesOfFacetValue, allDuplicateValues)]
Helpful for debugging when there's too many duplicates

<a id="utils.runWebui"></a>

#### runWebui

```python
def runWebui(path, port)
```

Runs a simple http server at the given path, using the given port

