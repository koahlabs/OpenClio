import prompts
from prompts import getFacetPrompt, getFacetClusterNamePrompt, getNeighborhoodClusterNamesPrompt, getDeduplicateClusterNamesPrompt, getAssignToHighLevelClusterPrompt, getRenamingHigherLevelClusterPrompt
from dataclasses import dataclass, field
from utils import flatten, unflatten, runBatched
import vllm
import pandas as pd
from typing import Any, Union, Tuple, Optional, Callable, Dict, List, TypeAlias
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import typing as npt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from collections import defaultdict
import torch
import os
import re
import random
import cloudpickle

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]

@dataclass
class FacetExtraInfo:
    minValue: int
    maxValue: int

'''
Returns true if we should make the cluster hierarchy for the given facet
'''
def shouldMakeFacetClusters(facet):
    return facet.summaryCriteria is not None

@dataclass
class Facet:
    name: str
    question: str
    prefill: str = ""
    summaryCriteria: Optional[str] = None
    numeric: Optional[Tuple[int, int]] = None

mainFacets = [
    Facet(
        name="Request",
        question="What is the user’s overall request for the assistant?",
        prefill="The user’s overall request for the assistant is to",
        summaryCriteria="The cluster name should be a sentence in the imperative that captures the user’s request. For example, ‘Brainstorm ideas for a birthday party’ or ‘Help me find a new job.",
    ),
    Facet(
        name="Language",
        question="What are the main languages of the conversation? Do not mention programming languages and do not provide only language codes; only the full names for human languages, like ‘English’ or ‘Arabic; Indonesian’. Do not include language families (just provide the general language, e.g., ‘Arabic’ and not ‘Tripolitanian Arabic’; something that a standard langcodes package would be able to identify). Only consider languages in or after the human’s first request. Output each language as a single full word with no other commentary.",
    ),
    Facet(
        name="Task",
        question="What task is the model being asked to perform in this conversation?",
        prefill="The task is to",
        # I made this, wasn't described in paper
        summaryCriteria="The cluster name should be a clear single sentence that captures the specific action or task the model is being asked to perform in the conversation. It should use an active verb and be written from the model's perspective. For example, ‘Generate a creative writing story about space exploration’ or ‘Calculate the probability of winning a card game’",
    ),
    Facet(
        name="Concerning",
        question="""On a scale of 1 to 5, how concerning or potentially harmful is this conversation from a safety perspective? Use the following scale:
1: Not concerning at all. Completely benign conversation.
Example: A friendly chat about favorite books or movies.

2: Slightly concerning, but likely harmless.
Example: A heated but civil debate about politics.

3: Moderately concerning. May warrant a closer look.
Example: Discussion of controversial topics that could escalate.

4: Very concerning. Likely needs review.
Example: Conversations involving potential self-harm or harmful ideation.

5: Extremely concerning. Immediate review needed.
Example: Explicit threats of violence or illegal activities.
Answer with only a single number from 1 to 5.""",
        numeric=(1,5),
    )
]


# Todo: When getting name and description, instead of counting frequency, choose element that has smallest embedded distance from the average choice


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

@dataclass
class OpenClioConfig:

    ### General params
    seed: int = 27 # Useful so runs are deterministic
    llmBatchSize: int = 1000 # Batch size to use when doing llm calls. Larger batch will run faster but takes more gpu memory
    embedBatchSize: int = 1000 # Batch size to use when embedding. Larger batch will run faster but takes more gpu memory
    

    ### Generate Base Clusters params
    nBaseClustersFunc: Callable[[int], int] = lambda n: n//10 # Number of base clusters to start with, depends on data size. If unspecified, will set to lambda n: n//10
    maxPointsToSampleInsideCluster : int = 10 # Number of points we sample inside the cluster, when determining base cluster names and summaries. More will make longer contexts but give the llm more information
    maxPointsToSampleOutsideCluster : int = 10 # Number of points we sample outside the cluster (as examples of stuff closest to, but *not* in the cluster), when determining base cluster names and summaries. More will make longer contexts but give the llm more information
    nNameDescriptionSamplesPerCluster: int = 5 # How many times to sample a cluster's name and description. We sample multiple times and take the most frequent answer, so higher values here help avoid any noise from data ordering (but takes longer)

    ### Hierarchy params
    minTopLevelSize: int = 5 # Once we've reached this many or less clusters, we have reached the top, stop going higher
    # neighborhoods
    nAverageClustersPerNeighborhood: Callable[[int], int] = lambda n: max(1, n//40) # Function that tells us how many number of clusters to have per neighborhood, on average. From G.7, "average number of clusters per neighborhood is 40", so default is lambda n: max(1, n//40)
    nSamplesOutsideNeighborhood: int = 5 # How many samples from outside the k-means cluster to add to each neighborhood. From G.7, "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters on the boundary between neighborhoods are neither overcounted nor undercounted)." 
    # get names from neighborhoods
    nDesiredHigherLevelNamesPerClusterFunc: Callable[[int], int] = lambda n: max(1, n//2) # Given number of elements in our neighborhood, return how many higher level cluster names we should have. The default of lambda n: max(1, n//2) will result in there being rougly half the amount of cluster names at each level in the hierarchy.
    # dedup (none)
    # assign lower level to higher level categories 
    nCategorizeSamples: int = 5 # How many times to resample assignments of cluster to higher level categories. The most common sample is chosen. More samples will take longer but help decrease noise from ordering of members of this category
    # rename once we see what's in the categories
    maxChildrenForRenaming: int = 10 # Maximum number of children in category to display when deciding what to name it, more will make longer prompt but give more accurate classification
    nRenameSamples: int = 5 # How many times to resample the new name and description that we sample, once the children are assigned to a cluster. More samples will take longer but help decrease noise from ordering of children

    llmExtraInferenceArgs: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 1000
    }) # Extra parameters to pass into vllm.SamplingParams


def getModels():
    model_str = "Qwen/Qwen2.5-7B-Instruct"
    llm = vllm.LLM(model=model_str)
    embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return llm, embeddingModel


def getData():
    d = pd.read_parquet("train-00000-of-00006.parquet", engine="pyarrow")
    return [d.iloc[i].conversation for i in range(len(d))]




@dataclass
class OpenClioResults:
    conversationsFacets: List[ConversationFacetData]
    conversationsEmbedings: List[Optional[EmbeddingArray]]
    baseKMeans: List[KMeans]
    baseClusters: List[Optional[List[ConversationCluster]]]
    rootClusters: List[Optional[List[ConversationCluster]]]

# conversationsFacetsSmol, conversationsEmbedingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacetsSmol, conversationsEmbedingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacetsSmol, conversationsEmbedings=conversationsEmbedingsSmol, kMeans=kMeansSmol, baseClusters=baseClustersSmol)
# conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacets, conversationsEmbedings=conversationsEmbedings, kMeans=kMeans, baseClusters=baseClusters)
def runClio(facets : List[Facet], 
            llm : vllm.LLM,
            embeddingModel : SentenceTransformer,
            conversations : List[str],
            cfg : OpenClioConfig = None,
            conversationsFacets: List[ConversationFacetData] = None,
            conversationsEmbedings: List[Optional[EmbeddingArray]] = None,
            baseKMeans: List[KMeans] = None,
            baseClusters: List[Optional[List[ConversationCluster]]] = None
        ) -> OpenClioResults:

    if cfg is None:
        cfg = OpenClioConfig()

    setSeed(cfg.seed)

    print("Getting facets")
    conversationsFacets: List[ConversationFacetData] = getFacets(
        facets=facets, 
        llm=llm,
        conversations=conversations,
        cfg=cfg) if conversationsFacets is None else conversationsFacets
    
    print("Getting embeddings")
    conversationsEmbedings: List[Optional[EmbeddingArray]] = getEmbeddings(
        facets=facets,
        conversationsFacets=conversationsFacets,
        embeddingModel=embeddingModel,
        cfg=cfg) if conversationsEmbedings is None else conversationsEmbedings
    
    with open("chonkers/tmpClioResultsTotes1.pkl", "wb") as f:
        cloudpickle.dump((conversationsFacets, conversationsEmbedings), f)
    print("Getting base clusters")
    baseKMeans, baseClusters = getBaseClusters(
        facets=facets,
        llm=llm,
        conversationsFacets=conversationsFacets,
        conversationsEmbedings=conversationsEmbedings,
        cfg=cfg) if baseKMeans is None or baseClusters is None else (baseKMeans, baseClusters)
    with open("chonkers/tmpClioResultsTotes2.pkl", "wb") as f:
        cloudpickle.dump((conversationsFacets, conversationsEmbedings, baseKMeans, baseClusters), f)
    
    print("Getting higher level clusters")
    rootClusters : List[Optional[List[ConversationCluster]]] = getHierarchy(
        facets=facets,
        llm=llm,
        embeddingModel=embeddingModel,
        baseClusters=baseClusters,
        cfg=cfg
    )

    return OpenClioResults(
        conversationsFacets=conversationsFacets,
        conversationsEmbedings=conversationsEmbedings,
        baseKMeans=baseKMeans,
        baseClusters=baseClusters,
        rootClusters=rootClusters
    )

'''
Get the pair of names that have closest embeddings when using embeddingModel
Returns (pairI, pairJ, pairCosineSimilarity)
'''
def getClosestNames(
    names : List[str],
    embeddingModel: SentenceTransformer
    ) -> Tuple[int, int, float]:
    embedded = embeddingModel.encode(names)
    sims = cosine_similarity(embedded)
    # middle is 0, make it not
    np.fill_diagonal(sims, -1)
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    print(sims[i,j])
    print(names[i])
    print(names[j])
    return i,j, sims[i,j]

def printHierarchyHelper(
    parents : List[ConversationCluster],
    indent: str) -> List[str]:
    lines = []
    for parent in parents:
        lines.append(indent + parent.name)
        if not parent.children is None:
            lines += printHierarchyHelper(parent.children, indent + "  ")
    return lines

def printHierarchy(parents : List[ConversationCluster]):
    resLines = printHierarchyHelper(parents, indent="")
    print("\n".join(resLines))

'''
Embed map(valueMap, facetValues) into 
cfg.nAverageClustersPerNeighborhood(len(facetStrValues)) clusters
using kmeans,
then add cfg.nSamplesOutsideNeighborhood extra closest samples to each cluster
return (kmeans, [[neighborhood0...], [neighborhood1...]])
'''
def getNeighborhoods(
    facetStrValues: List[str],
    embeddingModel: SentenceTransformer,
    cfg: OpenClioConfig) -> Tuple[KMeans, List[List[int]]]:
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]
    
    facetClusterEmbeddings = np.stack(runBatched(facetStrValues,
                                   getInputs=lambda facetStrValue: facetStrValue,
                                   processBatch=processBatchFunc,
                                   processOutput=lambda facetValue, facetValuePrompt, outputEmbeddings: outputEmbeddings,
                                   batchSize=cfg.embedBatchSize))
    facetNeighborhoods = []
    numValues = len(facetStrValues)
    # in the paper this is numClusters // 40
    k = cfg.nAverageClustersPerNeighborhood(numValues)
    kmeans = KMeans(n_clusters=min(numValues, k), random_state=cfg.seed)
    kmeans.fit(preprocessing.normalize(facetClusterEmbeddings))
    distances = cdist(facetClusterEmbeddings, kmeans.cluster_centers_)
    for clusterIndex in range(len(kmeans.cluster_centers_)):
        # Get points belonging to this cluster
        clusterPointsIndices = np.where(kmeans.labels_ == clusterIndex)[0]
        # Get closest points not in this cluster
        outsideClusterIndices = np.where(kmeans.labels_ != clusterIndex)[0]
        closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != clusterIndex, clusterIndex])]
        # From G.7:
        # "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters)
        #  on the boundary between neighborhoods are neither overcounted nor undercounted"
        clusterIndicesInNeighborhood = list(clusterPointsIndices) + list(closestPointsOutsideClusterIndices[:cfg.nSamplesOutsideNeighborhood])
        facetNeighborhoods.append(clusterIndicesInNeighborhood)

    return kmeans, facetNeighborhoods

'''
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
'''
def getHierarchy(
    facets : List[Facet],
    llm : vllm.LLM,
    embeddingModel : SentenceTransformer,
    baseClusters : List[Optional[List[ConversationCluster]]],
    cfg : OpenClioConfig) -> List[Optional[List[ConversationCluster]]]:
    seed = cfg.seed
    def processBatchFuncLLM(prompts : List[str]) -> List[str]:
        nonlocal seed # we increment it so duplicate entries will get distinct things
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
        modelOutputs = llm.generate(prompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs] 
    
    topLevelParents = []
    for facetI, facet in enumerate(facets):
        if not shouldMakeFacetClusters(facet):
            topLevelParents.append(None)
            continue

        curLevelFacetClusters : List[ConversationCluster] = baseClusters[facetI]
        level = 0
        while len(curLevelFacetClusters) > cfg.minTopLevelSize:

            #### Get embeddings for clusters ####

            Sources: TypeAlias = List[int]

            print(f"facet {facet} level {level}")

            print("getting category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=list(map(lambda cluster: f"{cluster.name}\n{cluster.summary}", curLevelFacetClusters)),
                        embeddingModel=embeddingModel,
                        cfg=cfg)

            #### Get higher level category names ####

            print("getting higher level category names")
            def getInputsFunc(clusterIndicesInNeighborhood : Sources) -> str:
                clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
                # shuffle ordering
                random.shuffle(clustersInNeighborhood)
                return getNeighborhoodClusterNamesPrompt(facet, llm.get_tokenizer(), clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood)))
            
            def processOutputFunc(clusterIndicesInNeighborhood : Sources, clusterPrompt : str, clusterNamesOutput : str) -> List[Tuple[str, Sources]]:
                # also store where it came from
                # also remove punctuation
                return [(removePunctuation(output).strip(), clusterIndicesInNeighborhood) for output in extractAnswerNumberedList(clusterNamesOutput)]
            
            def dedupAndMergeSources(values : List[Tuple[str, Sources]]) -> List[Tuple[str, Sources]]:
                resultValues = defaultdict(lambda: set())
                for (value, sources) in values:
                    resultValues[value] |= set(sources)
                return sorted(list(resultValues.items()))
            
            # dedup exact named matches
            higherCategories = dedupAndMergeSources(
                    flatten(
                        runBatched(facetNeighborhoods,
                            getInputs=getInputsFunc,
                            processBatch=processBatchFuncLLM,
                            processOutput=processOutputFunc,
                            batchSize=cfg.llmBatchSize)
                    )
            )

            #### Dedup higher categories ####
            
            print("getting higher level category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=[name for (name, sources) in higherCategories],
                        embeddingModel=embeddingModel,
                        cfg=cfg)
            
            print("deduping higher level categories")
            def getInputsFunc(higherCategoryIndicesInNeighborhoods : List[Tuple[str, Sources]]) -> str:
                # 0 is value, 1 is sources
                higherCategoriesInNeighborhood = [higherCategories[i][0] for i in higherCategoryIndicesInNeighborhoods]
                targetAmount =  max(1, len(higherCategoriesInNeighborhood)-1) # aim for -1 (arbitrary), but prompt lets it do more or less as needed
                if len(higherCategoriesInNeighborhood) == 2:
                    targetAmount = 2 # for only two, it'll mangle the categories if we ask it to dedup them into one, so don't do that
                return getDeduplicateClusterNamesPrompt(facet, llm.get_tokenizer(), higherCategoriesInNeighborhood, targetAmount)
            
            def processOutputFunc(
                    higherCategoryIndicesInNeighborhoods : List[Tuple[str, Sources]],
                    higherCategoryDedupPrompt : str,
                    higherCategoryDedupOutput : str) -> List[Tuple[str, Sources]]:
                # get sources in terms of original categories (union over all the different higher category inputs to this dedup)
                allSources = set()
                for (higherCategory, higherCategorySources) in [higherCategories[i] for i in higherCategoryIndicesInNeighborhoods]:
                    allSources |= set(higherCategorySources)
                allSources = sorted(list(allSources))
                return [(removePunctuation(output).strip(), allSources) for output in extractAnswerNumberedList(higherCategoryDedupOutput)]
            
            # todo: size 1 or 2 clusters, just ignore them (size 2 maybe set desired to 2? unless very high overlap in embed? idk)
            dedupedCategories = dedupAndMergeSources(
                    flatten(
                        runBatched(facetNeighborhoods,
                            getInputs=getInputsFunc,
                            processBatch=processBatchFuncLLM,
                            processOutput=processOutputFunc,
                            batchSize=cfg.llmBatchSize)
                    )
            )
            
            #### Assign to new best fit higher-level cluster ####

            # (they didn't specify how to choose what to put here, but I figure just tracking where parents came from and using all those that might have come from x should work fine)
            print("Assigning to best fit higher-level clusters")
            baseClusterPotentialHigherLevelClusters : List[List[str]] = [[] for _ in curLevelFacetClusters]
            for category, sources in dedupedCategories:
                for sourceI in sources:
                    baseClusterPotentialHigherLevelClusters[sourceI].append(category)
            
            def getInputsFunc(facetClusterData : Tuple[ConversationCluster, List[str]]) -> List[str]:
                facetCluster, potentialHigherLevelClusters = facetClusterData
                assignToHigherCategoryPrompts = []
                for i in range(cfg.nCategorizeSamples):
                    random.shuffle(potentialHigherLevelClusters)
                    assignToHigherCategoryPrompts.append(getAssignToHighLevelClusterPrompt(llm.get_tokenizer(), clusterToAssign=facetCluster, higherLevelClusters=potentialHigherLevelClusters))
                return assignToHigherCategoryPrompts

            # name and summary will be generated later
            parents : Dict[str, ConversationCluster] = dict(
                [
                    (categoryName.lower().strip(), ConversationCluster(facet=facet, name=categoryName, summary="")) 
                    for (categoryName, categorySources) in dedupedCategories
                ]
            )
            
            def processOutputFunc(
                    facetClusterData : Tuple[ConversationCluster, List[str]],
                    assignToHigherCategoryPrompts : List[str],
                    assignToHigherCategoryOutput : List[str]
                ):
                facetCluster, potentialHigherLevelClusters = facetClusterData
                counts = defaultdict(lambda: [])
                for output in assignToHigherCategoryOutput:
                    foundOutput, outputValue = extractTagValue(output, "answer")
                    # remove cluster and punctuation if it added it
                    outputValue = removePunctuation(outputValue.replace("<cluster>", "").replace("</cluster>", "").strip()).strip()
                    if foundOutput:
                        counts[outputValue.lower()].append(outputValue)
                mostCommonKey, mostCommonValue = max(list(counts.items()), key=lambda x: len(x[1]))
                if mostCommonKey in parents.keys():
                    if parents[mostCommonKey].children is None:
                        parents[mostCommonKey].children = []
                    parents[mostCommonKey].children.append(facetCluster)
                    facetCluster.parent = parents[mostCommonKey]
                else:
                    raise ValueError("(todo: use embedding to lookup) Could not find key: " + mostCommonKey)
                return None

            runBatched(list(zip(curLevelFacetClusters, baseClusterPotentialHigherLevelClusters)),
                getInputs=getInputsFunc,
                processBatch=processBatchFuncLLM,
                processOutput=processOutputFunc,
                batchSize=cfg.llmBatchSize)

            # remove any parents that didn't have any children assigned
            for parentKey, parentValue in list(parents.items()):
                if parentValue.children is None or len(parentValue.children) == 0:
                    del parents[parentKey]

            #### Rename categories based on which children they were given ####

            print("Renaming categories based on children")
            def getInputsFunc(parent : ConversationCluster) -> List[str]:
                renamingPrompts = []
                for _ in range(cfg.nRenameSamples):
                    random.shuffle(parent.children)
                    renamingPrompts.append(getRenamingHigherLevelClusterPrompt(facet, llm.get_tokenizer(), parent.children[:cfg.maxChildrenForRenaming]))
                return renamingPrompts
            
            def processOutputFunc(parent : ConversationCluster, renamePrompts : List[str], renamingOutputs : List[str]):
                summary, name = getMostCommonSummaryAndName(renamingOutputs)
                parent.summary = summary
                parent.name = name
        
            runBatched(list(parents.values()),
                getInputs=getInputsFunc,
                processBatch=processBatchFuncLLM,
                processOutput=processOutputFunc,
                batchSize=cfg.llmBatchSize)

            # Now those parents are our current level, go up higher
            curLevelFacetClusters = list(parents.values())
            level += 1
        topLevelParents.append(curLevelFacetClusters)
    return topLevelParents


'''
Gets the base-level clusters for all facets that have shouldMakeFacetClusters(facet) True
Returns (listOfKMeansForFacets, listOfListOfConversationClusters)
where 
listOfKMeansForFacets has a kmeans (or None) for each facet,
and
listOfListOfConversationClusters has a list of ConversationCluster for each facet
there will be cfg.nBaseClusters number of ConversationClusters in that list
'''
def getBaseClusters(
        facets : List[Facet],
        llm : vllm.LLM,
        conversationsFacets : List[ConversationFacetData],
        conversationsEmbedings : List[Optional[EmbeddingArray]],
        cfg : OpenClioConfig
    ) -> Tuple[List[Optional[KMeans]], List[Optional[List[ConversationCluster]]]]:
    tokenizer = llm.get_tokenizer()
    kMeansFacets = [None] * len(facets)

    seed = cfg.seed
    def getInputsFunc(facetAndEmbeddings : Tuple[int, Tuple[Facet, EmbeddingArray]]) -> List[List[str]]:
        nonlocal seed
        seed += 1 # do this so repeated entries get different outputs
        lookupClusterPrompts = []
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetClusters(facet):
            print(f"Running kmeans for facet {facet.name}")
            n = facetEmbeddings.shape[0]
            kmeans = KMeans(n_clusters=min(n, cfg.nBaseClustersFunc(n)), random_state=seed)
            kmeans.fit(preprocessing.normalize(facetEmbeddings))
            distances = cdist(facetEmbeddings, kmeans.cluster_centers_)
            kMeansFacets[facetI] = kmeans

            # For each cluster, find the index of the closest point
            for clusterIndex in range(len(kmeans.cluster_centers_)):
                # Get points belonging to this cluster
                clusterPointsIndices = np.where(kmeans.labels_ == clusterIndex)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(cfg.maxPointsToSampleInsideCluster, clusterPointsIndices.shape[0]), replace=False)
                # Get closest points not in this cluster
                outsideClusterIndices = np.where(kmeans.labels_ != clusterIndex)[0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != clusterIndex, clusterIndex])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(cfg.maxPointsToSampleOutsideCluster, closestPointsOutsideClusterIndices.shape[0])]

                # grab the facet values
                clusterFacetValues = [conversationsFacets[i].facetValues[facetI].value for i in clusterPointsIndices]
                clusterOutsideValues = [conversationsFacets[i].facetValues[facetI].value for i in sampledOutsideClusterIndices]
                # generate the cluster name prompt
                clusterPrompts = []
                for _ in range(cfg.nNameDescriptionSamplesPerCluster):
                    # randomize the ordering to avoid positional biases
                    random.shuffle(clusterFacetValues)
                    random.shuffle(clusterOutsideValues)
                    prompt = getFacetClusterNamePrompt(tokenizer, facet, clusterFacetValues, clusterOutsideValues)
                    clusterPrompts.append(prompt)
                lookupClusterPrompts.append(clusterPrompts)
        else:
            kMeansFacets.append(None)
        return lookupClusterPrompts
    
    def processBatchFunc(batchOfPrompts : List[str]) -> List[str]:
        nonlocal seed
        seed += 1 # do this so repeated entries get different outputs
        samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(
            facetAndEmbeddings : Tuple[int, Tuple[Facet, EmbeddingArray]],
            clusterPrompts: List[List[str]],
            outputs: List[List[str]]
        ) -> List[ConversationCluster]:
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetClusters(facet):
            outputClusters = []
            kmeans = kMeansFacets[facetI]
            for clusterIndex, clusterOutputs in zip(range(len(kmeans.cluster_centers_)), outputs):
                clusterPointsIndices = np.arange(len(conversationsFacets))[kmeans.labels_ == clusterIndex]
                summary, name = getMostCommonSummaryAndName(clusterOutputs)
                outputClusters.append(
                    ConversationCluster(
                        facet=facet,
                        summary=summary,
                        name=name,
                        indices=clusterPointsIndices,
                    )
                )
            return outputClusters
        else:
            return None

    return kMeansFacets, runBatched(list(enumerate(zip(facets, conversationsEmbedings))),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize)

'''
Gets the embeddings of all facet values that have shouldMakeFacetClusters(facet) True
(this is when the facet has a summaryCriteria that is not None)
Returns one element for each facet value
That element will either be None if shouldMakeFacetClusters(facet) is False,
or a numpy array of size [numConversations, embeddingDim]
'''
def getEmbeddings(
        facets: List[Facet],
        conversationsFacets : List[ConversationFacetData],
        embeddingModel : SentenceTransformer,
        cfg: OpenClioConfig) -> List[Optional[EmbeddingArray]]:
    def getInputsFunc(facetI : int) -> List[str]:
        facetInputs = []
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            for conversationFacetData in conversationsFacets:
                facetValue = conversationFacetData.facetValues[facetI].value
                facetInputs.append(facetValue)
        return facetInputs
    
    def processBatchFunc(batchOfTextInputs : List[str]) -> List[npt.NDArray[np.float32]]:
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(facetI : int, facetInputs : List[str], embeddings : List[npt.NDArray[np.float32]]) -> Optional[EmbeddingArray]:
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            return np.stack(embeddings)
        else:
            return None
    
    return runBatched(list(range(len(facets))),
                    getInputs=getInputsFunc,
                    processBatch=processBatchFunc,
                    processOutput=processOutputFunc,
                    batchSize=cfg.embedBatchSize)


'''
Converts a conversation like
[
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hi :3"}
]
into a corresponding string
User:
Hi there
Assistant:
Hi :3
'''
def conversationToString(conversation : List[Dict[str, str]]) -> str:
    return "\n".join([f"{turn['role']}:\n{turn['content']}" for turn in conversation])

'''
Gets facet values for every conversation, for each of the facets provided, using the provided llm.
Returns a list of ConversationFacetData objects,
one for each conversation
'''
def getFacets(
        facets : List[Facet],
        llm : vllm.LLM,
        conversations: List[List[Dict[str, str]]],
        cfg : OpenClioConfig
    ) -> List[ConversationFacetData]:
    def getInputsFunc(conversation : List[Dict[str, str]]) -> List[str]:
        conversationStr = conversationToString(conversation)
        # runBatched will automatically flatten these into us for nice batched usage,
        # then unflatten them back before calling processOutputFunc
        # so we can send in whatever sort of nested lists we want (though in this case it's only one deep)
        return [getFacetPrompt(llm.get_tokenizer(), conversationStr, facet.question, facet.prefill) for facet in facets]
    seed = cfg.seed
    def processBatchFunc(batchOfPrompts : List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(conversation : List[Dict[str, str]], conversationPrompts : List[str], facetOutputs : List[str]) -> ConversationFacetData:
        return ConversationFacetData(
            conversation=conversation,
            facetValues=[
                FacetValue(
                    facet=facet,
                    value=cleanTrailingTagsInOutput(
                        extractTagValue(value, "answer")[1])
                ) for (facet, value) in zip(facets, facetOutputs)]
        )

    return runBatched(conversations,
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize)






##### Various utility parsing stuff #####


# from https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def setSeed(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

'''
Gets most common thing in <summary> tag and <name> tag,
returns (summary, name)
Todo: instead of counting (most will be unique),
'''
def getMostCommonSummaryAndName(outputs: List[str]) -> Tuple[str, str]:
    summaryCounts = defaultdict(lambda: 0)
    nameCounts = defaultdict(lambda: 0)
    for output in outputs:
        # re.DOTALL makes . match newlines too (by default it does not)
        matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
        if len(matches) > 0:
            for summary, name in matches:
                summaryCounts[cleanOutput(summary)] += 1
                nameCounts[cleanOutput(name)] += 1
    def largestCountItem(counts, fieldName):
        if len(counts) == 0: return f"<Could not extract {fieldName}>"
        counts = sorted(list(counts.items()), key=lambda x: (-x[1], x[0]))
        largestKey, largestValue = counts[0]
        return largestKey
    summary = largestCountItem(summaryCounts, "summary")
    name = largestCountItem(nameCounts, "name")
    return summary, name

'''
Removes ., ?, and ! from end of a string.
(and strips it before and afterwards)
'''
def removePunctuation(output : str) -> str:
    output = output.strip()
    if output.endswith("."):
        output = output[:-1].strip()
    if output.endswith("?"):
        output = output[:-1].strip()
    if output.endswith("!"):
        output = output[:-1].strip()
    return output

'''
Gets value contained in <tag>VALUE_HERE</tag>
returns (foundTag, valueInTag)
where foundTag is True if the tag was found
'''
def extractTagValue(output: str, tag: str) -> Tuple[bool, str]:
    posOfTag = output.lower().find(f"<{tag}>")
    if posOfTag != -1:
        output = output[posOfTag + len(f"<{tag}>"):].strip()
    endOfTagPos = output.lower().find(f"</{tag}>")
    if endOfTagPos != -1:
        output = output[:endOfTagPos].strip()
    return posOfTag != -1 and endOfTagPos != -1, output

'''
If we have
<answer>
1. blah
2. blahhh
3. wow
etc.
</answer>
This will return 
["blah", "blahhh", "wow", ...]
'''
def extractAnswerNumberedList(output : str) -> List[str]:
    results = []
    foundAnswerTag, answer = extractTagValue(output, "answer")
    if foundAnswerTag:
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) > 0]
    return results

'''
Removes number. from the front of the output
Like 
"1. hi"
becomes
"hi"
or
"144. wow"
becomes
"wow"
'''
def removeNumberFromOutput(output : str) -> str:
    return re.sub(r"^\d*?\.", "", output.strip(), count=1).strip()

'''
Removes any trailing </tag> that may existing in the output
Also strips it before and afterwards
'''
def cleanTrailingTagsInOutput(output : str) -> str:
    return re.findall(r"(.*?)(?:(?:</)|$)", output.strip(), re.DOTALL)[0].strip()
