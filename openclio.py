import prompts
from prompts import getFacetPrompt, getFacetClusterNamePrompt, getNeighborhoodClusterNamesPrompt, getDeduplicateClusterNamesPrompt, getAssignToHighLevelClusterPrompt, getRenamingHigherLevelClusterPrompt
from dataclasses import dataclass
from utils import flatten, unflatten, runBatched
import vllm
import pandas as pd
from typing import Any, Union, Tuple, Optional, Callable, Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from collections import defaultdict
import re
import random

@dataclass
class FacetExtraInfo:
    minValue: int
    maxValue: int

@dataclass
class Facet:
    name: str
    question: str
    prefill: str = ""
    summaryCriteria: Optional[str] = None
    numeric: Optional[Tuple[int, int]] = None

facets = [
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

facetExtraInfos = {
    
}



# Todo: When getting name and description, instead of counting frequency, choose element that has smallest embedded distance from the average choice


@dataclass
class FacetValue:
    facet: Facet
    value: str

@dataclass
class ConversationFacetData:
    conversation: list[Any]
    facetValues: list[FacetValue]

@dataclass
class ConversationEmbedding:
    conversation: list[Any]
    embedding: Any
    
@dataclass
class ConversationCluster:
    facet: Facet
    summary: str
    name: str
    children: Optional[list[ConversationCluster]] = None
    parent: Optional[ConversationCluster] = None
    indices: Optional[np.ndarray[np.int32]] = None

@dataclass
class OpenClioConfig:

    ### General params
    seed: int = 27 # Useful so runs are deterministic
    llmBatchSize: int = 1000 # Batch size to use when doing llm calls. Larger batch will run faster but takes more gpu memory
    embedBatchSize: int = 1000 # Batch size to use when embedding. Larger batch will run faster but takes more gpu memory
    

    ### Generate Base Clusters params
    nOfBaseClusters: Optional[int] = None # Number of base clusters to start with. If unspecified, will set to data size / 10
    maxPointsToSampleInsideCluster : int = 10 # Number of points we sample inside the cluster, when determining base cluster names and summaries. More will make longer contexts but give the llm more information.
    maxPointsToSampleOutsideCluster : int = 10 # Number of points we sample outside the cluster (as examples of stuff closest to, but *not* in the cluster), when determining base cluster names and summaries. More will make longer contexts but give the llm more information.
    nNameDescriptionSamplesPerCluster: int = 5 # How many times to sample a cluster's name and description. We sample multiple times and take the most frequent answer, so higher values here help avoid any noise from data ordering (but takes longer).

    ### Hierarchy params
    nAverageClustersPerNeighborhood: Callable[[int], int] = lambda n: max(1, n//40) # Function that tells us how many number of clusters to have per neighborhood, on average. From G.7, "average number of clusters per neighborhood is 40", so default is lambda n: max(1, n//40)
    nSamplesOutsideNeighborhood: int = 5 # How many samples from outside the k-means cluster to add to each neighborhood. From G.7, "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters on the boundary between neighborhoods are neither overcounted nor undercounted)." 
    nCategorizeSamples: int = 5 # How many times to resample assignments of cluster to higher level categories. The most common sample is chosen. More samples will take longer but help decrease noise from orderings.


    ### Resume params
    conversationsFacets: Optional[]

    llmExtraInferenceArgs: Dict[str, Any] = {
        "max_tokens": 1000
    } # Extra parameters to pass into vllm.SamplingParams



def getModels():
    model_str = "Qwen/Qwen2.5-7B-Instruct"
    llm = vllm.LLM(model=model_str)
    embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return llm, embeddingModel


def getData():
    d = pd.read_parquet("train-00000-of-00006.parquet", engine="pyarrow")
    return [d.iloc[i].conversation for i in range(len(d))]

# conversationsFacetsSmol, conversationsEmbedingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacetsSmol, conversationsEmbedingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacetsSmol, conversationsEmbedings=conversationsEmbedingsSmol, kMeans=kMeansSmol, baseClusters=baseClustersSmol)
# conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacets, conversationsEmbedings=conversationsEmbedings, kMeans=kMeans, baseClusters=baseClusters)
def runClio(facets : List[Facet], 
            llm : vllm.LLM,
            embeddingModel : SentenceTransformer,
            conversations : List[str],
            cfg : OpenClioConfig,
            conversationsFacets=None,
            conversationsEmbedings=None,
            kMeans=None,
            baseClusters=None):
    np.random.seed(seed)
    random.seed(seed)
    print("Getting facets")
    conversationsFacets = getFacets(
        facets=facets, 
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        conversations=conversations,
        cfg=cfg, **kwargs) if conversationsFacets is None else conversationsFacets
    print("Getting embeddings")
    conversationsEmbedings = getEmbeddings(
        facets=facets,
        conversationsFacets=conversationsFacets,
        embeddingModel=embeddingModel,
        batchSize=embedBatchSize) if conversationsEmbedings is None else conversationsEmbedings
    print("Getting base clusters")
    
    kMeans, baseClusters = getBaseClusters(
        facets=facets,
        llm=llm,
        conversationsFacets=conversationsFacets,
        conversationsEmbedings=conversationsEmbedings,
        numberOfBaseClusters=numberOfBaseClusters,
        nLLMSamplesPerCluster=nLLMSamplesPerCluster,
        batchSize=llmBatchSize,
        seed=seed,
        **kwargs) if kMeans is None or baseClusters is None else (kMeans, baseClusters)
    print("Getting higher level clusters")
    higherCategories, dedupedCategories, parents = getHierarchy(
        facets=facets,
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        embeddingModel=embeddingModel,
        baseClusters=baseClusters,
        nClustersOutside=nClustersOutside,
        nCategorizeSamples=nCategorizeSamples,
        desiredNames=desiredNames,
        embedBatchSize=embedBatchSize,
        llmBatchSize=llmBatchSize,
        seed=seed,
        **kwargs
    )

    return conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories, dedupedCategories, parents

'''
Get the pair of names that have closest embeddings when using embeddingModel
Returns (pairI, pairJ, pairCosineSimilarity)
'''
def getClosestNames(
    names : List[str],
    embeddingModel: SentenceTransformer
    ) -> Tuple[int, int, int]:
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
    resLines = printHierarchyHelper(list(parents.values()), indent="")
    print("\n".join(resLines))

    
'''
Embed map(valueMap, facetValues) into 
cfg.nAverageClustersPerNeighborhood(len(facetStrValues)) clusters
using kmeans,
then add cfg.nSamplesOutsideNeighborhood extra closest samples to each cluster
return (kmeans, [[neighborhood0...], [neighborhood1...]])
'''
def getNeighborhoods(
    facetStrValues: List[Any],
    embeddingModel: SentenceTransformer,
    cfg: OpenClioConfig) -> Tuple[KMeans, List[List[int]]]:
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]
    
    facetClusterEmbeddings = np.stack(runBatched(facetStrValues,
                                   getInputs=lambda facetStrValue: facetStrValue,
                                   processBatch=processBatchFunc,
                                   processOutput=lambda facetValue, facetValuePrompt, outputEmbeddings: outputEmbeddings,
                                   batchSize=embedBatchSize))
    facetNeighborhoods = []
    numValues = len(facetStrValues)
    # in the paper this is numClusters // 40
    k = cfg.nAverageClustersPerNeighborhood(numValues)
    kmeans = KMeans(n_clusters=k, random_state=cfg.seed)
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

def getHierarchy(
    facets : List[Facet],
    llm : vllm.LLM,
    embeddingModel : SentenceTransformer,
    baseClusters : List[Optional[List[]]]):
    def processBatchFuncLLM(prompts):
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **kwargs)
        modelOutputs = llm.generate(prompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs] 
    
    for facetI, facet in enumerate(facets):
        if not shouldMakeFacetClusters(facet): continue

        curLevelFacetClusters = baseClusters[facetI]

        #### Get embeddings for clusters ####

        print("getting category neighborhoods")
        kmeans, facetNeighborhoods = getNeighborhoods(facet,
                    facetValues=curLevelFacetClusters,
                    valueMap=lambda cluster: f"{cluster.name}\n{cluster.summary}",
                    embeddingModel=embeddingModel,
                    embedBatchSize=embedBatchSize,
                    kFunc=lambda n: max(1, n//40),
                    nClustersOutside=nClustersOutside,
                    seed=seed)

        #### Get higher level category names ####

        print("getting higher level category names")
        def getInputsFunc(clusterIndicesInNeighborhood):
            clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
            # shuffle ordering
            random.shuffle(clustersInNeighborhood)
            # idk desiredNames
            return getNeighborhoodClusterNamesPrompt(facet, tokenizer, clustersInNeighborhood, desiredNames)
        
        def processOutputFunc(clusterIndicesInNeighborhood, clusterPrompt, clusterNamesOutput):
            # also store where it came from
            # also remove punctuation
            return [(removePunctuation(output).strip(), clusterIndicesInNeighborhood) for output in extractAnswerNumberedList(clusterNamesOutput)]
                
        def dedupAndMergeSources(values):
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
                        batchSize=llmBatchSize)
                )
        )

        #### Dedup higher categories ####
        
        print("getting higher level category neighborhoods")
        kmeans, facetNeighborhoods = getNeighborhoods(facet,
                    facetValues=higherCategories,
                    # [1] is the sources
                    valueMap=lambda categoryName: categoryName[0],
                    embeddingModel=embeddingModel,
                    embedBatchSize=embedBatchSize,
                    kFunc=lambda n: n//5, # picked this arbitrairly
                    nClustersOutside=0, # picked this arbitrairly
                    seed=seed)
        
        print("deduping higher level categories")
        def getInputsFunc(higherCategoryIndicesInNeighborhoods):
            # [1] is the sources
            higherCategoriesInNeighborhood = [higherCategories[i][0] for i in higherCategoryIndicesInNeighborhoods]
            targetAmount =  max(1, len(higherCategoriesInNeighborhood)-1)
            if len(higherCategoriesInNeighborhood) == 2:
                targetAmount = 2 # for only two, it'll mangle the categories if we ask it to dedup them into one, so don't do that
            return getDeduplicateClusterNamesPrompt(facet, tokenizer, higherCategoriesInNeighborhood, targetAmount)
        
        def processOutputFunc(higherCategoryIndicesInNeighborhoods, higherCategoryDedupPrompts, higherCategoryDedupOutput):
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
                        batchSize=llmBatchSize)
                )
        )
        
        #### Assign to new best fit higher-level cluster ####

        # (they didn't specify how to choose what to put here, but I figure just tracking where parents came from and using all those that might have come from x should work fine)
        print("Assigning to best fit higher-level clusters")
        baseClusterPotentialHigherLevelClusters = [[] for _ in curLevelFacetClusters]
        for category, sources in dedupedCategories:
            for sourceI in sources:
                baseClusterPotentialHigherLevelClusters[sourceI].append(category)
        
        def getInputsFunc(facetClusterData):
            facetCluster, potentialHigherLevelClusters = facetClusterData
            assignToHigherCategoryPrompts = []
            for i in range(nCategorizeSamples):
                random.shuffle(potentialHigherLevelClusters)
                assignToHigherCategoryPrompts.append(getAssignToHighLevelClusterPrompt(llm.get_tokenizer(), clusterToAssign=facetCluster, higherLevelClusters=potentialHigherLevelClusters))
            print(assignToHigherCategoryPrompts[0])
            return assignToHigherCategoryPrompts

        # name and summary will be generated later
        parents = dict([(categoryName.lower().strip(), ConversationCluster(facet=facet, name=categoryName, summary="")) for (categoryName, categorySources) in dedupedCategories])
        
        def processOutputFunc(facetClusterData, assignToHigherCategoryPrompts, assignToHigherCategoryOutput):
            facetCluster, potentialHigherLevelClusters = facetClusterData
            counts = defaultdict(lambda: [])
            for output in assignToHigherCategoryOutput:
                foundOutput, outputValue = extractTagValue(output, "answer")
                # remove cluster if it added it
                outputValue.replace("<cluster>", "").replace("</cluster>", "").strip()
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
            batchSize=llmBatchSize)

        # remove any parents that didn't have any children assigned
        for parentKey, parentValue in list(parents.items()):
            if parentValue.children is None or len(parentValue.children) == 0:
                del parents[parent]
        #### Rename categories based on which children they were given ####
        print("Renaming categories based on children")
        def getInputsFunc(parent):
            renamingPrompts = []
            for _ in range(nCategorizeSamples):
                random.shuffle(parent.children[:maxNChildrenForRenaming])
                renamingPrompts.append(getRenamingHigherLevelClusterPrompt(facet, llm.get_tokenizer(), parent.children))
            return renamingPrompts
        
        def processOutputFunc(parent, renamePrompt, renamingOutputs):
            summary, name = getMostCommonSummaryAndName(renamingOutputs)
            parent.summary = summary
            parent.name = name
     
        runBatched(list(parents.values()),
            getInputs=getInputsFunc,
            processBatch=processBatchFuncLLM,
            processOutput=processOutputFunc,
            batchSize=llmBatchSize)

        return higherCategories, dedupedCategories, parents



def getBaseClusters(
    facets : List[Facet],
    llm : vllm.LLM,
    conversationsFacets,
    conversationsEmbedings):
    tokenizer = llm.get_tokenizer()
    kMeansFacets = []
    def getInputsFunc(facetAndEmbeddings):
        lookupClusterPrompts = []
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetClusters(facet):
            print(f"Running kmeans for facet {facet.name}")
            kmeans = KMeans(n_clusters=numberOfBaseClusters, random_state=seed)
            kmeans.fit(preprocessing.normalize(facetEmbeddings))
            distances = cdist(facetEmbeddings, kmeans.cluster_centers_)
            kMeansFacets.append(kmeans)

            # For each cluster, find the index of the closest point
            for clusterIndex in range(len(kmeans.cluster_centers_)):
                # Get points belonging to this cluster
                clusterPointsIndices = np.where(kmeans.labels_ == clusterIndex)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(maxPointsToSampleInCluster, clusterPointsIndices.shape[0]), replace=False)
                # Get closest points not in this cluster
                outsideClusterIndices = np.where(kmeans.labels_ != clusterIndex)[0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != clusterIndex, clusterIndex])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(maxPointsToSampleOutsideCluster, clusterPointsIndices.shape[0])]

                # grab the facet values
                clusterFacetValues = [conversationsFacets[i].facetValues[facetI].value for i in clusterPointsIndices]
                clusterOutsideValues = [conversationsFacets[i].facetValues[facetI].value for i in sampledOutsideClusterIndices]
                # generate the cluster name prompt
                clusterPrompts = []
                for _ in range(numNameDescriptionSamplesPerCluster):
                    # randomize the ordering to avoid positional biases
                    random.shuffle(clusterFacetValues)
                    random.shuffle(clusterOutsideValues)
                    prompt = getFacetClusterNamePrompt(tokenizer, facet, clusterFacetValues, clusterOutsideValues)
                    clusterPrompts.append(prompt)
                lookupClusterPrompts.append(clusterPrompts)
        else:
            kMeansFacets.append(None)
        return lookupClusterPrompts
    
    def processBatchFunc(batchOfPrompts):
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **kwargs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(facetAndEmbeddings, clusterPrompts, outputs):
        outputClusters = []
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetClusters(facet):
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

    return kMeansFacets, runBatched(list(enumerate(zip(facets, conversationsEmbedings))),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=batchSize)

def shouldMakeFacetClusters(facet):
    return facet.summaryCriteria is not None

def getEmbeddings(
        facets: List[Facet],
        conversationsFacets,
        embeddingModel,
        cfg: ):
    def getInputsFunc(facetI):
        facetInputs = []
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            for conversationFacetData in conversationsFacets:
                facetValue = conversationFacetData.facetValues[facetI].value
                facetInputs.append(facetValue)
        return facetInputs
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(facetI, facetInputs, embeddings):
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
            return True, output
        return False, output
    return False, output

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
def extractAnswerNumberedList(output : str) : List[str]:
    results = []
    foundAnswerTag, answer = extractTagValue(output, "answer")
    if foundAnswerTag:
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) >= 0]
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
def removeNumberFromOutput(output : str) : str:
    return re.sub(r"^\d*?\.", "", output.strip(), count=1).strip()

'''
Removes any trailing </tag> that may existing in the output
Also strips it before and afterwards
'''
def cleanTrailingTagsInOutput(output : str) : str:
    return re.findall(r"(.*?)(?:(?:</)|$)", output.strip(), re.DOTALL)[0].strip()


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
Gets values for every conversation for each of the facets provided,
Using the provided llm.
'''
def getFacets(
    facets : List[Facet],
    llm : vllm.LLM,
    conversations: List[Dict[str, str]]) -> List[ConversationFacetData]:
    def getInputsFunc(conversation):
        conversationStr = conversationToString(conversation)
        # runBatched will automatically flatten these into us for nice batched usage,
        # then unflatten them back before calling processOutputFunc
        # so we can send in whatever sort of nested lists we want (though in this case it's only one deep)
        return [getFacetPrompt(llm.get_tokenizer(), conversationStr, facet.question, facet.prefill) for facet in facets]

    def processBatchFunc(batchOfPrompts):
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **kwargs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(conversation, conversationPrompts, facetOutputs):
        def getAnswer(output):
            matches = re.findall(r"(.*?)(?:</answer>|$)", output, re.DOTALL)
            if len(matches) > 0:
                for answer in matches:
                    return cleanOutput(answer)
            return cleanOutput(output) # failed to match, just return cleaned output
        facetValues = [FacetValue(facet=facet, value=getAnswer(value)) for (facet, value) in zip(facets, facetOutputs)]
        return ConversationFacetData(
            conversation=conversation,
            facetValues=facetValues
        )

    return runBatched(conversations,
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=batchSize)


