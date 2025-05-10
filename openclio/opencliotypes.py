from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Callable, Any, List, TypeAlias
import numpy as np
from numpy import typing as npt

from .faissKMeans import FaissKMeans

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]

@dataclass(frozen=True) # frozen=true gives it hash and eq
class Facet:
    name: str
    question: str = ""
    prefill: str = ""
    summaryCriteria: Optional[str] = None
    numeric: Optional[Tuple[int, int]] = None
    getFacetPrompt: Optional[Callable[[Any, "Facet", Any, "OpenClioConfig"], str]] = None # takes in tokenizer, facet, conversation (can be anything), cfg and outputs a prompt to "extract" this facet. If None, will use prompts.getFacetPrompt from the paper

def shouldMakeFacetClusters(facet: Facet) -> bool:
    """Returns true if we should make the cluster hierarchy for the given facet"""
    return facet.summaryCriteria is not None

@dataclass
class FacetValue:
    facet: Facet
    value: str

@dataclass
class ConversationFacetData:
    conversation: List[Any]
    facetValues: List[FacetValue]

@dataclass
class ConversationEmbedding:
    conversation: List[Any]
    embedding: Any
    
@dataclass
class ConversationCluster:
    facet: Facet
    summary: str
    name: str
    children: Optional[List['ConversationCluster']] = None
    parent: Optional['ConversationCluster'] = None
    indices: Optional[np.ndarray] = None

    def __hash__(self):
        return hash((self.summary, self.name, self.facet))
    
    def __eq__(self, other):
        if not isinstance(other, ConversationCluster):
            return False
        return self.summary == other.summary and self.name == other.name and self.facet == other.facet

@dataclass
class OpenClioConfig:
    """
    Configuration for a run of openclio.

    There's a lot of params here. General guide:
    - Decrease llmBatchSize if you get gpu out of memory errors
    - Decrease maxConversationTokens to around model context length - 1000 (1000 because we need room for prompt and thinking as well)
    - set tokenizerArgs to {} if you get an error about "enable_thinking" not supported
    - Set llmExtraInferenceArgs to be whatever is the recommended sampler settings for your llm (these will be passed to vllm.SamplingParams)

    Beyond that, the default values here are fine and will adapt to your data size.

    If you want to modify stuff more, the first place to start is things that change how the hierarchy is shaped (I include here their default values):
    minTopLevelSize=5 
        Once the current highest level of the hierarchy gets <= this many clusters we'll stop making higher levels.
        You can increase this if you want a wider hierarchy at the top level.
    nBaseClustersFunc=lambda n: n//10
        You can change 10 to some larger value if you want more items in each base cluster (10 is number of data points in each base-level cluster, on average)
    nDesiredHigherLevelNamesPerClusterFunc=lambda n: n//2
        2 is your "branching factor" of the hierarchy. Higher values will result in more children at each level (and thus, a more shallow hierarchy)

    If each data point doesn't look like a conversation ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]), you can modify
    dedupKeyFunc = lambda dataPoint: <return some value here you can use as a key for deduplicating your data>
    and
    getConversationFunc = lambda dataPoint: <return some value here that looks like a conversation>

    If they don't look like a conversation at all, you can't use facets=openclio.mainFacets.
    However you can use facets=openclio.genericSummaryFacets or make your own facets.
    If you are making your own facets or using genericSummaryFacets, you can just leave getConversationFunc as it's default value of lambda data: data
    And it'll pass the data to your facet's getFacetPrompt function (openclio.genericSummaryFacets just calls str(data))
    """
    # 
    

    # The rest of the parameters here are fairly reasonable and you can leave at default settings, feel free to consult the comments by them if you'd like to know what they do.

    ### General params
    seed: int = 27 # Useful so runs are deterministic
    verbose: bool = True
    llmBatchSize: int = 1000 # Batch size to use when doing llm calls. Larger batch will run faster but takes more gpu memory
    embedBatchSize: int = 1000 # Batch size to use when embedding. Larger batch will run faster but takes more gpu memory
    dedupData: bool = True # Whether to deduplicate the data. This is very important as non-deduped data can result in very large cluster sizes (because all the values are the same)
    dedupKeyFunc: Optional[Callable[[Any], Any]] = None # The function to use for comparing if two pieces of data are equivalent. If None, will use prompts.conversationToString if it's a list, or just the value otherwise

    ### Generate Base Clusters params
    getConversationFunc: Callable[[Any], List[Dict[str, str]]] = lambda conversation: conversation # function to extract the data (used for looking up facets) from a specific point of data, by default assumes this is the identity function. Useful if your data is like a tuple where one of the entries is the conversation (just return that entry)
    maxConversationTokens: int = 8000 # max tokens for a conversation, conversations will be truncated after this (rounding to turn boundaries ending with assistant). Important to prevent overwhelming the model context size
    nBaseClustersFunc: Callable[[int], int] = lambda n: n//10 # Number of base clusters to start with, depends on data size. If unspecified, will set to lambda n: n//10
    maxPointsToSampleInsideCluster: int = 10 # Number of points we sample inside the cluster, when determining base cluster names and summaries. More will make longer contexts but give the llm more information
    maxPointsToSampleOutsideCluster: int = 10 # Number of points we sample outside the cluster (as examples of stuff closest to, but *not* in the cluster), when determining base cluster names and summaries. More will make longer contexts but give the llm more information
    nNameDescriptionSamplesPerCluster: int = 5 # How many times to sample a cluster's name and description. We sample multiple times and take the most frequent answer, so higher values here help avoid any noise from data ordering (but takes longer)

    ### Hierarchy params
    minTopLevelSize: int = 5 # Once we've reached this many or less clusters, we have reached the top, stop going higher
    # neighborhoods
    nAverageClustersPerNeighborhood: Callable[[int], int] = lambda n: max(1, n//10) # Function that tells us how many number of clusters to have per neighborhood, on average. From G.7, "average number of clusters per neighborhood is 40", so default is lambda n: max(1, n//40) But that's too many for a small model, lets do smaller like 10
    nSamplesOutsideNeighborhood: int = 5 # How many samples from outside the k-means cluster to add to each neighborhood. From G.7, "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters on the boundary between neighborhoods are neither overcounted nor undercounted)." 
    # get names from neighborhoods
    nDesiredHigherLevelNamesPerClusterFunc: Callable[[int], int] = lambda n: max(1, n//3) # Given number of elements in our neighborhood, return how many higher level cluster names we should have. The default of lambda n: max(1, n//3) will result in there being rougly half the amount of cluster names at each level in the hierarchy.
    # dedup (none)
    # assign lower level to higher level categories 
    nCategorizeSamples: int = 5 # How many times to resample assignments of cluster to higher level categories. The most common sample is chosen. More samples will take longer but help decrease noise from ordering of members of this category
    # rename once we see what's in the categories
    maxChildrenForRenaming: int = 10 # Maximum number of children in category to display when deciding what to name it, more will make longer prompt but give more accurate classification
    nRenameSamples: int = 5 # How many times to resample the new name and description that we sample, once the children are assigned to a cluster. More samples will take longer but help decrease noise from ordering of children

    tokenizerArgs: Dict[str, Any] = field(default_factory=lambda: {
        "enable_thinking": False # don't need thinking for the simple things we are doing, also without this we lose prompt prefix (I think?)
    }) # Extra parameters to pass into our tokenizer when caling apply_chat_template

    llmExtraInferenceArgs: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 1000,
         # default qwen non-thinking sampling params
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
    }) # Extra parameters to pass into vllm.SamplingParams

    kmeansArgs: Dict[str, Any] = field(default_factory = lambda: {
        "approximate": True, # since we only have 10 elements per term, by default this would take many hours, this speeds it up a lot
        "verbose": True,
    })

    ### Website settings
    htmlMaxSizePerFile: int = 10000000 # Maximum size per json file: the data on the website is split up into chunks of this size or less and setup so you can stream the data as needed
    htmlConversationFilterFunc: Optional[Callable[[List[Dict[str, str]], ConversationFacetData], bool]] = None # Optional function that takes two inputs (dataPoint: Any, dataPointFacetData: ConversationFacetData) and returns bool if we should include that data on the website.
    htmlDataToJsonFunc: Optional[Callable[[Any], Dict[str, Any]]] = None # Optional function that takes a data point and returns a json of the corresponding conversation. It should look like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]. If you just want to dispaly the data as a string, just return a single entry like this: [{"role": "<whatever you want>", "content": "<your str content>"}]

    ### Webui settings
    webuiPort: int = 8421

@dataclass
class OpenClioResults:
    facets: List[Facet]
    facetValues: List[ConversationFacetData]
    facetValuesEmbeddings: List[Optional[EmbeddingArray]]
    baseClusters: List[Optional[List[ConversationCluster]]]
    rootClusters: List[Optional[List[ConversationCluster]]]
    data: List[List[Dict[str, str]]]
    cfg: OpenClioConfig