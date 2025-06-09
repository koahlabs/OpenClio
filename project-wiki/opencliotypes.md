# Table of Contents

* [opencliotypes](#opencliotypes)
  * [Facet](#opencliotypes.Facet)
    * [name](#opencliotypes.Facet.name)
    * [question](#opencliotypes.Facet.question)
    * [prefill](#opencliotypes.Facet.prefill)
    * [summaryCriteria](#opencliotypes.Facet.summaryCriteria)
    * [numeric](#opencliotypes.Facet.numeric)
    * [getFacetPrompt](#opencliotypes.Facet.getFacetPrompt)
  * [shouldMakeFacetClusters](#opencliotypes.shouldMakeFacetClusters)
  * [OpenClioConfig](#opencliotypes.OpenClioConfig)
    * [seed](#opencliotypes.OpenClioConfig.seed)
    * [verbose](#opencliotypes.OpenClioConfig.verbose)
    * [llmBatchSize](#opencliotypes.OpenClioConfig.llmBatchSize)
    * [embedBatchSize](#opencliotypes.OpenClioConfig.embedBatchSize)
    * [dedupData](#opencliotypes.OpenClioConfig.dedupData)
    * [dedupKeyFunc](#opencliotypes.OpenClioConfig.dedupKeyFunc)
    * [getConversationFunc](#opencliotypes.OpenClioConfig.getConversationFunc)
    * [maxConversationTokens](#opencliotypes.OpenClioConfig.maxConversationTokens)
    * [nBaseClustersFunc](#opencliotypes.OpenClioConfig.nBaseClustersFunc)
    * [maxPointsToSampleInsideCluster](#opencliotypes.OpenClioConfig.maxPointsToSampleInsideCluster)
    * [maxPointsToSampleOutsideCluster](#opencliotypes.OpenClioConfig.maxPointsToSampleOutsideCluster)
    * [nNameDescriptionSamplesPerCluster](#opencliotypes.OpenClioConfig.nNameDescriptionSamplesPerCluster)
    * [minTopLevelSize](#opencliotypes.OpenClioConfig.minTopLevelSize)
    * [nAverageClustersPerNeighborhood](#opencliotypes.OpenClioConfig.nAverageClustersPerNeighborhood)
    * [nSamplesOutsideNeighborhood](#opencliotypes.OpenClioConfig.nSamplesOutsideNeighborhood)
    * [nDesiredHigherLevelNamesPerClusterFunc](#opencliotypes.OpenClioConfig.nDesiredHigherLevelNamesPerClusterFunc)
    * [nCategorizeSamples](#opencliotypes.OpenClioConfig.nCategorizeSamples)
    * [maxChildrenForRenaming](#opencliotypes.OpenClioConfig.maxChildrenForRenaming)
    * [nRenameSamples](#opencliotypes.OpenClioConfig.nRenameSamples)
    * [tokenizerArgs](#opencliotypes.OpenClioConfig.tokenizerArgs)
    * [llmExtraInferenceArgs](#opencliotypes.OpenClioConfig.llmExtraInferenceArgs)
    * [htmlMaxSizePerFile](#opencliotypes.OpenClioConfig.htmlMaxSizePerFile)
    * [htmlConversationFilterFunc](#opencliotypes.OpenClioConfig.htmlConversationFilterFunc)
    * [htmlDataToJsonFunc](#opencliotypes.OpenClioConfig.htmlDataToJsonFunc)

<a id="opencliotypes"></a>

# opencliotypes

<a id="opencliotypes.Facet"></a>

## Facet Objects

```python
@dataclass(frozen=True)
class Facet()
```

<a id="opencliotypes.Facet.name"></a>

#### name

Plan text name of the facet

<a id="opencliotypes.Facet.question"></a>

#### question

The question we are asking about the data

<a id="opencliotypes.Facet.prefill"></a>

#### prefill

Prefill for the LLM output when extracting facet information

<a id="opencliotypes.Facet.summaryCriteria"></a>

#### summaryCriteria

Summary criteria when making hierarchies, this must be not None in order to build hierarchy

<a id="opencliotypes.Facet.numeric"></a>

#### numeric

Either None (if not numeric), or (minValue, maxValue) if this facet extracts a numeric field

<a id="opencliotypes.Facet.getFacetPrompt"></a>

#### getFacetPrompt

takes in tokenizer, facet, conversation (can be anything), cfg and outputs a prompt to "extract" this facet. If None, will use prompts.getFacetPrompt from the paper

<a id="opencliotypes.shouldMakeFacetClusters"></a>

#### shouldMakeFacetClusters

```python
def shouldMakeFacetClusters(facet: Facet) -> bool
```

Returns true if we should make the cluster hierarchy for the given facet

<a id="opencliotypes.OpenClioConfig"></a>

## OpenClioConfig Objects

```python
@dataclass
class OpenClioConfig()
```

Configuration for a run of openclio.

There's a lot of params here. General guide:
- Decrease llmBatchSize if you get gpu out of memory errors
- Decrease maxConversationTokens to around model context length - 1000 (1000 because we need room for prompt and thinking as well)
- set tokenizerArgs to {} if you get an error about "enable_thinking" not supported
- Set llmExtraInferenceArgs to be whatever is the recommended sampler settings for your llm (these will be passed to vllm.SamplingParams)

Beyond that, the default values here are fine and will adapt to your data size.

If you want to modify stuff more, the first place to start is things that change how the hierarchy is shaped (I include here their default values):
- minTopLevelSize=5 
  - Once the current highest level of the hierarchy gets <= this many clusters we'll stop making higher levels.
  - You can increase this if you want a wider hierarchy at the top level.
- nBaseClustersFunc=lambda n: n//10
  - You can change 10 to some larger value if you want more items in each base cluster (10 is number of data points in each base-level cluster, on average)
- nDesiredHigherLevelNamesPerClusterFunc=lambda n: n//2
  - 2 is your "branching factor" of the hierarchy. Higher values will result in more children at each level (and thus, a more shallow hierarchy)

If each data point doesn't look like a conversation ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]), you can modify

dedupKeyFunc = lambda dataPoint: return some value here you can use as a key for deduplicating your data

and

getConversationFunc = lambda dataPoint: return some value here that looks like a conversation

If they don't look like a conversation at all, you can't use facets=openclio.mainFacets.

However you can use facets=openclio.genericSummaryFacets or make your own facets.

If you are making your own facets or using genericSummaryFacets, you can just leave getConversationFunc as it's default value of lambda data: data

And it'll pass the data to your facet's getFacetPrompt function (openclio.genericSummaryFacets just calls str(data))

### General params

<a id="opencliotypes.OpenClioConfig.seed"></a>

#### seed

Useful so runs are deterministic

<a id="opencliotypes.OpenClioConfig.verbose"></a>

#### verbose

Whether to print intermediate outputs and progress bars

<a id="opencliotypes.OpenClioConfig.llmBatchSize"></a>

#### llmBatchSize

Batch size to use when doing llm calls. Larger batch will run faster but takes more gpu memory

<a id="opencliotypes.OpenClioConfig.embedBatchSize"></a>

#### embedBatchSize

Batch size to use when embedding. Larger batch will run faster but takes more gpu memory

<a id="opencliotypes.OpenClioConfig.dedupData"></a>

#### dedupData

Whether to deduplicate the data. This is very important as non-deduped data can result in very large cluster sizes (because all the values are the same)

<a id="opencliotypes.OpenClioConfig.dedupKeyFunc"></a>

#### dedupKeyFunc

The function to use for comparing if two pieces of data are equivalent. If None, will use prompts.conversationToString if it's a list, or just the value otherwise

### Generate Base Clusters params

<a id="opencliotypes.OpenClioConfig.getConversationFunc"></a>

#### getConversationFunc

Function to extract the data (used for looking up facets) from a specific point of data, by default assumes this is the identity function. Useful if your data is like a tuple where one of the entries is the conversation (just return that entry)

<a id="opencliotypes.OpenClioConfig.maxConversationTokens"></a>

#### maxConversationTokens

Max tokens for a conversation, conversations will be truncated after this (rounding to turn boundaries ending with assistant). Important to prevent overwhelming the model context size

<a id="opencliotypes.OpenClioConfig.nBaseClustersFunc"></a>

#### nBaseClustersFunc

Number of base clusters to start with, depends on data size. If unspecified, will set to lambda n: n//10

<a id="opencliotypes.OpenClioConfig.maxPointsToSampleInsideCluster"></a>

#### maxPointsToSampleInsideCluster

Number of points we sample inside the cluster, when determining base cluster names and summaries. More will make longer contexts but give the llm more information

<a id="opencliotypes.OpenClioConfig.maxPointsToSampleOutsideCluster"></a>

#### maxPointsToSampleOutsideCluster

Number of points we sample outside the cluster (as examples of stuff closest to, but *not* in the cluster), when determining base cluster names and summaries. More will make longer contexts but give the llm more information

<a id="opencliotypes.OpenClioConfig.nNameDescriptionSamplesPerCluster"></a>

#### nNameDescriptionSamplesPerCluster

How many times to sample a cluster's name and description. We sample multiple times and take the most frequent answer, so higher values here help avoid any noise from data ordering (but takes longer)

### Hierarchy params

<a id="opencliotypes.OpenClioConfig.minTopLevelSize"></a>

#### minTopLevelSize

Once we've reached this many or less clusters, we have reached the top, stop going higher

<a id="opencliotypes.OpenClioConfig.nAverageClustersPerNeighborhood"></a>

#### nAverageClustersPerNeighborhood

Function that tells us how many number of clusters to have per neighborhood, on average. From G.7, "average number of clusters per neighborhood is 40", so default is lambda n: max(1, n//40) But that's too many for a small model, lets do smaller like 10

<a id="opencliotypes.OpenClioConfig.nSamplesOutsideNeighborhood"></a>

#### nSamplesOutsideNeighborhood

How many samples from outside the k-means cluster to add to each neighborhood. From G.7, "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters on the boundary between neighborhoods are neither overcounted nor undercounted)."

<a id="opencliotypes.OpenClioConfig.nDesiredHigherLevelNamesPerClusterFunc"></a>

#### nDesiredHigherLevelNamesPerClusterFunc

Given number of elements in our neighborhood, return how many higher level cluster names we should have. The default of lambda n: max(1, n//3) will result in there being rougly half the amount of cluster names at each level in the hierarchy.

<a id="opencliotypes.OpenClioConfig.nCategorizeSamples"></a>

#### nCategorizeSamples

How many times to resample assignments of cluster to higher level categories. The most common sample is chosen. More samples will take longer but help decrease noise from ordering of members of this category

<a id="opencliotypes.OpenClioConfig.maxChildrenForRenaming"></a>

#### maxChildrenForRenaming

Maximum number of children in category to display when deciding what to name it, more will make longer prompt but give more accurate classification

<a id="opencliotypes.OpenClioConfig.nRenameSamples"></a>

#### nRenameSamples

How many times to resample the new name and description that we sample, once the children are assigned to a cluster. More samples will take longer but help decrease noise from ordering of children

<a id="opencliotypes.OpenClioConfig.tokenizerArgs"></a>

### Extra Params

#### tokenizerArgs

Extra parameters to pass into our tokenizer when caling apply_chat_template

<a id="opencliotypes.OpenClioConfig.llmExtraInferenceArgs"></a>

#### llmExtraInferenceArgs

Extra parameters to pass into vllm.SamplingParams

<a id="opencliotypes.OpenClioConfig.htmlMaxSizePerFile"></a>

### Website Params

#### password

A string to use, to password protect your webui files. By default, password is None and they will be unprotected.

#### htmlMaxSizePerFile

Maximum size per json file: the data on the website is split up into chunks of this size or less and setup so you can stream the data as needed

<a id="opencliotypes.OpenClioConfig.htmlConversationFilterFunc"></a>

#### htmlConversationFilterFunc

Optional function that takes two inputs (dataPoint: Any, dataPointFacetData: ConversationFacetData) and returns bool if we should include that data on the website.

<a id="opencliotypes.OpenClioConfig.htmlDataToJsonFunc"></a>

#### htmlDataToJsonFunc

Optional function that takes a data point and returns a json of the corresponding conversation. It should look like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]. If you just want to dispaly the data as a string, just return a single entry like this: [{"role": "<whatever you want>", "content": "<your str content>"}]

<a id="opencliotypes.OpenClioConfig.webuiPort"></a>

#### webuiPort

The port used when hosting webui