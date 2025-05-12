# Table of Contents

* [openclio](#openclio)
  * [runClio](#openclio.runClio)
  * [getNeighborhoods](#openclio.getNeighborhoods)
  * [getHierarchy](#openclio.getHierarchy)
  * [getBaseClusters](#openclio.getBaseClusters)
  * [getFacetValuesEmbeddings](#openclio.getFacetValuesEmbeddings)
  * [getFacetValues](#openclio.getFacetValues)
  * [connectedComponentsFromMask](#openclio.connectedComponentsFromMask)
  * [medoidFromEmbeddings](#openclio.medoidFromEmbeddings)
  * [deduplicateByEmbeddingsAndMergeSources](#openclio.deduplicateByEmbeddingsAndMergeSources)
  * [deduplicateByEmbeddings](#openclio.deduplicateByEmbeddings)
  * [setSeed](#openclio.setSeed)
  * [bestRepresentativePair](#openclio.bestRepresentativePair)
  * [getMedoidViaEmbeddings](#openclio.getMedoidViaEmbeddings)
  * [getCentroidViaEmbeddings](#openclio.getCentroidViaEmbeddings)
  * [getMedoidSummaryAndName](#openclio.getMedoidSummaryAndName)
  * [getMostCommonSummaryAndName](#openclio.getMostCommonSummaryAndName)
  * [removePunctuation](#openclio.removePunctuation)
  * [extractTagValue](#openclio.extractTagValue)
  * [extractAnswerNumberedList](#openclio.extractAnswerNumberedList)
  * [removeNumberFromOutput](#openclio.removeNumberFromOutput)
  * [cleanTrailingTagsInOutput](#openclio.cleanTrailingTagsInOutput)
  * [printHierarchy](#openclio.printHierarchy)

<a id="openclio"></a>

# openclio

<a id="openclio.runClio"></a>

#### runClio

```python
def runClio(facets: List[Facet],
            llm: vllm.LLM,
            embeddingModel: SentenceTransformer,
            data: List[List[Dict[str, str]]],
            outputDirectory: str,
            htmlRoot: str,
            hostWebui: bool = True,
            cfg: OpenClioConfig = None,
            **kwargs) -> OpenClioResults
```

Runs the Clio algorithm on the given conversations, using the given llm and embeddingModel.

Clio extracts facets from each conversation, then for some of those facets it generates a hierarchy you can view.

Once you are done with this, see convertOutputToJsonChunks

Keyword arguments:
- facets -- The facets we will extract from each conversation. You can use facets=openclio.mainFacets to use the facets from the paper.
- llm -- The llm that is used to extract facets and cluster data. This should be a vllm.LLM instance
- embeddingModel -- The embedding model used for clustering data (and a few other things). This should be a SentenceTransformer instance
- data -- The conversations we are running clio on. These should be formatted like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi :3"},...] (see OpenClioConfig in opencliotypes.py if your data isn't formatted like this)
- outputDirectory -- The directory path where we store checkpoints/outputs
- htmlRoot -- The path where the visuals will be stored on your website. For example, "/opencliooutputs"
- hostWebui -- True if we want to host webui (otherwise website files are stored, but not hosted)
- cfg -- Optional, an instance of openclio.OpenClioConfig, this lets you modify some of openclio's settings. Look at the comments for what individual fields mean.

Any extra args you provide will be assigned to your OpenClioConfig

**Returns**:

  An OpenClioResults object that contains the results of Clio. You can use convertOutputToJsonChunks (in writeOutput.py) to write this result to a directory for html browsing.

<a id="openclio.getNeighborhoods"></a>

#### getNeighborhoods

```python
def getNeighborhoods(
        facetStrValues: List[str], embeddingModel: SentenceTransformer,
        cfg: OpenClioConfig, nSamplesOutsideNeighborhood: int
) -> Tuple[FaissKMeans, List[List[int]]]
```

Embed map(valueMap, facetValues) into 
cfg.nAverageClustersPerNeighborhood(len(facetStrValues)) clusters
using kmeans,
then add cfg.nSamplesOutsideNeighborhood extra closest samples to each cluster
return (kmeans, [[neighborhood0...], [neighborhood1...]])

<a id="openclio.getHierarchy"></a>

#### getHierarchy

```python
def getHierarchy(
        facets: List[Facet], llm: vllm.LLM,
        embeddingModel: SentenceTransformer,
        baseClusters: List[Optional[List[ConversationCluster]]],
        cfg: OpenClioConfig) -> List[Optional[List[ConversationCluster]]]
```

Extracts a hierarchy of labels, starting with baseClusters at the lowest level
1. Get "neighborhoods" composed of 
k = cfg.nAverageClustersPerNeighborhood(numValues)
clusters via embedding cluster names and summaries, and adding cfg.nSamplesOutsideNeighborhood extra closest to but outside of the kmeans cluster.
2. Gets roughly cfg.nDesiredHigherLevelNamesPerClusterFunc(len(neighborhood)) names from each of those neighborhoods using llm
3. Dedups those names using llm
4. Assigns each lower level cluster to the possible higher level clusters, using llm
5. Renames the higher level clusters based on what got added to them, using llm
Repeats 1-5 until we have len(currentLevel) <= cfg.minTopLevelSize

Returns a list one element per facet.
That element will be None if shouldMakeFacetClusters(facet) is False, otherwise
That element will be a list of the top level ConversationClusters
You can use children to traverse.

<a id="openclio.getBaseClusters"></a>

#### getBaseClusters

```python
def getBaseClusters(
    facets: List[Facet], llm: vllm.LLM, embeddingModel: SentenceTransformer,
    facetValues: List[ConversationFacetData],
    facetValuesEmbeddings: List[Optional[EmbeddingArray]], cfg: OpenClioConfig,
    runIfNotExist: Callable[[str, Callable[[], Any], bool],
                            Tuple[Any, bool]], dependencyModified: bool
) -> Tuple[List[Optional[FaissKMeans]],
           List[Optional[List[ConversationCluster]]]]
```

Gets the base-level clusters for all facets that have shouldMakeFacetClusters(facet) True
Returns a list of lists, one list of ConversationCluster for each facet that should make facet clusters True
there will be cfg.nBaseClustersFunc(n) number of ConversationClusters in that list

<a id="openclio.getFacetValuesEmbeddings"></a>

#### getFacetValuesEmbeddings

```python
def getFacetValuesEmbeddings(
        facets: List[Facet], facetValues: List[ConversationFacetData],
        embeddingModel: SentenceTransformer,
        cfg: OpenClioConfig) -> List[Optional[EmbeddingArray]]
```

Gets the embeddings of all facet values that have shouldMakeFacetClusters(facet) True
(this is when the facet has a summaryCriteria that is not None)
Returns one element for each facet value
That element will either be None if shouldMakeFacetClusters(facet) is False,
or a numpy array of size [numConversations, embeddingDim]

<a id="openclio.getFacetValues"></a>

#### getFacetValues

```python
def getFacetValues(facets: List[Facet], llm: vllm.LLM,
                   data: List[List[Dict[str, str]]],
                   cfg: OpenClioConfig) -> List[ConversationFacetData]
```

Gets facet values for every conversation, for each of the facets provided, using the provided llm.
Returns a list of ConversationFacetData objects,
one for each conversation

<a id="openclio.connectedComponentsFromMask"></a>

#### connectedComponentsFromMask

```python
def connectedComponentsFromMask(mask: np.ndarray) -> List[np.ndarray]
```

mask: dense *boolean* adjacency matrix (n × n, symmetric, no self-loops)
returns: list of 1-D index arrays – one per connected component

<a id="openclio.medoidFromEmbeddings"></a>

#### medoidFromEmbeddings

```python
def medoidFromEmbeddings(indices: np.ndarray, embs: np.ndarray) -> int
```

embs: unit-norm embeddings (n × d)
indices: indices of the points that form one component
returns: index (WITHIN indices) of the true medoid under cosine distance

<a id="openclio.deduplicateByEmbeddingsAndMergeSources"></a>

#### deduplicateByEmbeddingsAndMergeSources

```python
def deduplicateByEmbeddingsAndMergeSources(
        valuesAndSources: List[Tuple[str, List[int]]],
        embeddingModel: SentenceTransformer,
        tau: float = 0.15)
```

Single-link deduplication.  Returns one representative per duplicate set,
chosen as the exact medoid of each connected component.
Sources for each representitive will be the union of the all sources in connected component

<a id="openclio.deduplicateByEmbeddings"></a>

#### deduplicateByEmbeddings

```python
def deduplicateByEmbeddings(
        values: List[str],
        embeddingModel: SentenceTransformer,
        tau: float = 0.15,
        valueMap: Optional[Callable[[Any], str]] = None) -> List[str]
```

Single-link deduplication.  Returns one representative per duplicate set,
chosen as the exact medoid of each connected component.

<a id="openclio.setSeed"></a>

#### setSeed

```python
def setSeed(seed)
```

Set seeds (to lots of things)

<a id="openclio.bestRepresentativePair"></a>

#### bestRepresentativePair

```python
def bestRepresentativePair(A: List[str], B: List[str],
                           model: SentenceTransformer) -> Tuple[str, str]
```

Return (a_star, b_star) where:
  • b_star  minimises Σ_{a∈A} (1 - cos(a, b))
  • a_star  is the element of A closest to that b_star

<a id="openclio.getMedoidViaEmbeddings"></a>

#### getMedoidViaEmbeddings

```python
def getMedoidViaEmbeddings(values: List[str],
                           embeddingModel: SentenceTransformer) -> str
```

Return the element of `values` that minimises the sum of
cosine distances ( = 1 - cosine similarity ) to all others.

This is the exact medoid under cosine distance.
Complexity:  O(n²) in time,  O(n²) in memory.

<a id="openclio.getCentroidViaEmbeddings"></a>

#### getCentroidViaEmbeddings

```python
def getCentroidViaEmbeddings(values: List[str],
                             embeddingModel: SentenceTransformer) -> str
```

Computes the average of the values embeddings,
then finds the value that is closest to that average
This is sort of a continuous version of "pick the most common element"
But actually what we want is the medoid (term that is closest to all the others)

<a id="openclio.getMedoidSummaryAndName"></a>

#### getMedoidSummaryAndName

```python
def getMedoidSummaryAndName(
        outputs: List[str],
        embeddingModel: SentenceTransformer) -> Tuple[str, str]
```

Continuous version of "get most common"
That gets the embedded value that is closest to all other items (the medoid)
returns (summary, name)

<a id="openclio.getMostCommonSummaryAndName"></a>

#### getMostCommonSummaryAndName

```python
def getMostCommonSummaryAndName(outputs: List[str]) -> Tuple[str, str]
```

Gets most common thing in <summary> tag and <name> tag,
returns (summary, name)
I recommend to use getMedoidSummaryAndName so we act in embedding space instead

<a id="openclio.removePunctuation"></a>

#### removePunctuation

```python
def removePunctuation(output: str) -> str
```

Removes ., ?, and ! from end of a string.
(and strips it before and afterwards)

<a id="openclio.extractTagValue"></a>

#### extractTagValue

```python
def extractTagValue(output: str, tag: str) -> Tuple[bool, str]
```

Gets value contained in <tag>VALUE_HERE</tag>
returns (foundTag, valueInTag)
where foundTag is True if the tag was found

<a id="openclio.extractAnswerNumberedList"></a>

#### extractAnswerNumberedList

```python
def extractAnswerNumberedList(output: str,
                              ignoreNoTrailing: bool = False) -> List[str]
```

If we have
<answer>
1. blah
2. blahhh
3. wow
etc.
</answer>
This will return 
["blah", "blahhh", "wow", ...]

<a id="openclio.removeNumberFromOutput"></a>

#### removeNumberFromOutput

```python
def removeNumberFromOutput(output: str) -> str
```

Removes number. from the front of the output
Like 
"1. hi"
becomes
"hi"
or
"144. wow"
becomes
"wow"

<a id="openclio.cleanTrailingTagsInOutput"></a>

#### cleanTrailingTagsInOutput

```python
def cleanTrailingTagsInOutput(output: str) -> str
```

Removes any trailing </tag> that may existing in the output
Also strips it before and afterwards

<a id="openclio.printHierarchy"></a>

#### printHierarchy

```python
def printHierarchy(parents: List[ConversationCluster])
```

helper function to manually print hierarchy of a specific facet

