from dataclasses import dataclass, field
import vllm
import pandas as pd
from typing import Any, Union, Tuple, Optional, Callable, Dict, List, TypeAlias
import faiss 
import functools
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import typing as npt
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from collections import defaultdict
import json
import torch
import os
import re
import random
import cloudpickle
from pathlib import Path

import .prompts as prompts
from .prompts import getFacetPrompt, getFacetClusterNamePrompt, getNeighborhoodClusterNamesPrompt, getDeduplicateClusterNamesPrompt, getAssignToHighLevelClusterPrompt, getRenamingHigherLevelClusterPrompt
from .utils import flatten, unflatten, runBatched, dedup
from .opencliotypes import Facet, FacetValue, ConversationFacetData, ConversationEmbedding, ConversationCluster, OpenClioConfig, OpenClioResults
from .faissKMeans import FaissKMeans
from .writeOutput import convertOutputToWebpage

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]

def shouldMakeFacetClusters(facet: Facet) -> bool:
    """Returns true if we should make the cluster hierarchy for the given facet"""
    return facet.summaryCriteria is not None

# these are facets from the paper
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

genericSummaryFacets = [
    Facet(
        name="Summary",
        getFacetPrompt=functools.partial(
            prompts.summarizeFacetPrompt,
            dataToStr=lambda data: str(data)
        ),
        summaryCriteria="The cluster name should be a clear single sentence that accurately captures the examples."
    )
]

    


# conversationsFacetsSmol, conversationsEmbeddingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacetsSmol, conversationsEmbeddingsSmol, kMeansSmol, baseClustersSmol, higherCategoriesSmol, dedupedCategoriesSmol, parentsSmol = clio.runClio(clio.facets, llm, embed, data[0:1000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacetsSmol, conversationsEmbeddings=conversationsEmbeddingsSmol, kMeans=kMeansSmol, baseClusters=baseClustersSmol)
# conversationsFacets, conversationsEmbeddings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000)
# conversationsFacets, conversationsEmbeddings, kMeans, baseClusters, higherCategories, dedupedCategories, parents = clio.runClio(clio.facets, llm, embed, data[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, nClustersOutside=5, nCategorizeSamples=5, desiredNames=5, seed=27, max_tokens=1000, conversationsFacets=conversationsFacets, conversationsEmbeddings=conversationsEmbeddings, kMeans=kMeans, baseClusters=baseClusters)
def runClio(facets: List[Facet], 
            llm: vllm.LLM,
            embeddingModel: SentenceTransformer,
            conversations: List[List[Dict[str, str]]],
            outputDirectory: str,
            htmlRoot: str,
            cfg: OpenClioConfig = None,
            **kwargs
        ) -> OpenClioResults:
    """
    Runs the Clio algorithm on the given conversations, using the given llm and embeddingModel.

    Clio extracts facets from each conversation, then for some of those facets it generates a hierarchy you can view.

    Once you are done with this, see convertOutputToJsonChunks
    
    Keyword arguments:
    facets -- The facets we will extract from each conversation. You can use facets=openclio.mainFacets to use the facets from the paper.
    llm -- The llm that is used to extract facets and cluster data. This should be a vllm.LLM instance
    embeddingModel -- The embedding model used for clustering data (and a few other things). This should be a SentenceTransformer instance
    conversations -- The conversations we are running clio on. These should be formatted like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi :3"},...]
    outputDirectory -- The directory path where we store checkpoints/outputs
    htmlRoot -- The path where the visuals will be stored on your website. For example, "/opencliooutputs"
    cfg -- Optional, an instance of openclio.OpenClioConfig, this lets you modify some of openclio's settings. Look at the comments for what individual fields mean.
    Any extra args you provide will be assigned to your OpenClioConfig
    Returns:
    An OpenClioResults object that contains the results of Clio. You can use convertOutputToJsonChunks (in writeOutput.py) to write this result to a directory for html browsing.
    """

    # make the output directory to store checkpoints if it does not exist
    Path(outputDirectory).mkdir(parents=True, exist_ok=True)

    dependencyModified = False
    def runIfNotExist(path, runFunc, dependencyModified):
        fullPath = os.path.join(outputDirectory, path)
        if os.path.exists(fullPath) and not dependencyModified: # recompute if dependency modified
            try:
                result = cloudpickle.load(fullPath)
                print(f"Resuming from {fullPath}")
                return result, False
            except:
                print(f"Failed to load from path {fullPath}, ignoring")
                print(traceback.format_exc())
        res = runFunc()
        with open(fullPath, "wb") as f:
            cloudpickle.dump(res, f)
        return res, True
    
    if cfg is None:
        cfg = OpenClioConfig(**kwargs)
    else:
        for k,v in kwargs.items():
            if hasattr(cfg, k):
                cfg.k = v
            else:
                raise ValueError(f"Unknown OpenClioConfig key {k} with value {v}")
    
    
    if cfg.dedupData:
        dedupKeyFunc = cfg.dedupKeyFunc
        if dedupKeyFunc is None:
            if len(conversations) > 0 and type(conversations[0]) is list or type(conversations[0]) is np.ndarray:
                # tokenize and truncate key func
                tokenizer = llm.get_tokenizer()
                dedupKeyFunc = lambda conversation: prompts.conversationToString(conversation, tokenizer=tokenizer, maxTokens=cfg.maxConversationTokens)
            else:
                # identity key func
                dedupKeyFunc = lambda x: x
        print("Deduping data")
        conversations, dependencyModified = runIfNotExist("dedupedData.pkl", lambda:
            dedup(conversations, dedupKeyFunc=dedupKeyFunc, batchSize=cfg.llmBatchSize)
        )
    

    print("Getting facet values")
    setSeed(cfg.seed) # doing this before each function call helps ensure reproducability if they resume
    facetValues, dependencyModified = 
        runIfNotExist("facetValues.pkl", lambda:
            getFacetValues(
                facets=facets, 
                llm=llm,
                conversations=conversations,
                cfg=cfg
            ),
            dependencyModified=dependencyModified
        )
    
    print("Getting facet value embeddings")
    setSeed(cfg.seed)
    facetValuesEmbeddings, dependencyModified = 
        runIfNotExist("facetValuesEmbeddings.pkl", lambda:
            getFacetValuesEmbeddings(
                facets=facets,
                facetValues=facetValues,
                embeddingModel=embeddingModel,
                cfg=cfg
            ),
            dependencyModified=dependencyModified
        )
    
    print("Getting base clusters")
    setSeed(cfg.seed)
    (baseKMeans, baseClusters), dependencyModified = 
        runIfNotExist("baseKMeansAndClusters.pkl", 
            getBaseClusters(
                facets=facets,
                llm=llm,
                embeddingModel=embeddingModel,
                facetValues=facetValues,
                facetValuesEmbeddings=facetValuesEmbeddings,
                cfg=cfg
            ),
            dependencyModified=dependencyModified
        )

    print("Getting higher level clusters")
    setSeed(cfg.seed)
    rootClusters, dependencyModified =
        runIfNotExist("rootClusters.pkl", lambda:
            getHierarchy(
                facets=facets,
                llm=llm,
                embeddingModel=embeddingModel,
                baseClusters=baseClusters,
                cfg=cfg
            ),
            dependencyModified=dependencyModified
        )
    
    res, dependencyModified = runIfNotExist("results.pkl", lambda:
        OpenClioResults(
            facets=facets,
            facetValues=facetValues,
            facetValuesEmbeddings=facetValuesEmbeddings,
            baseKMeans=baseKMeans,
            baseClusters=baseClusters,
            rootClusters=rootClusters,
            conversations=conversations
        ),
        dependencyModified=dependencyModified
    )

    htmlOutputPath = os.path.join(outputDirectory, htmlRoot)
    # clear old outputs
    if os.path.exists(htmlOutputPath):
        shutil.rmtree(htmlOutputPath)
    Path(htmlOutputPath).mkdir(parents=True, exist_ok=True)
    convertOutputToWebpage(
        output=output,
        rootHtmlPath=htmlRoot,
        targetDir=htmlOutputPath,
        maxSizePerFile=cfg.maxSizePerFile,
        conversationFilter=cfg.htmlConversationFilterFunc,
        dataToJson=cfg.htmlDataToJsonFunc,
    )


    return res

def getNeighborhoods(
    facetStrValues: List[str],
    embeddingModel: SentenceTransformer,
    cfg: OpenClioConfig,
    nSamplesOutsideNeighborhood: int) -> Tuple[FaissKMeans, List[List[int]]]:
    """
    Embed map(valueMap, facetValues) into 
    cfg.nAverageClustersPerNeighborhood(len(facetStrValues)) clusters
    using kmeans,
    then add cfg.nSamplesOutsideNeighborhood extra closest samples to each cluster
    return (kmeans, [[neighborhood0...], [neighborhood1...]])
    """
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs, show_progress_bar=False)
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
    kmeans = FaissKMeans(n_clusters=min(numValues, k), random_state=cfg.seed, **cfg.kmeansArgs)
    # we have to normalize for this to be cosine similarity
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
        clusterIndicesInNeighborhood = list(clusterPointsIndices) + list(closestPointsOutsideClusterIndices[:nSamplesOutsideNeighborhood])
        facetNeighborhoods.append(clusterIndicesInNeighborhood)

    return kmeans, facetNeighborhoods

def getHierarchy(
    facets: List[Facet],
    llm: vllm.LLM,
    embeddingModel: SentenceTransformer,
    baseClusters: List[Optional[List[ConversationCluster]]],
    cfg: OpenClioConfig) -> List[Optional[List[ConversationCluster]]]:
    """
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
    """
    seed = cfg.seed
    def processBatchFuncLLM(prompts: List[str]) -> List[str]:
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

        curLevelFacetClusters: List[ConversationCluster] = baseClusters[facetI]
        level = 0
        while len(curLevelFacetClusters) > cfg.minTopLevelSize:

            #### Get embeddings for clusters ####

            Sources: TypeAlias = List[int]

            print(f"facet {facet} level {level}")

            print("getting category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=list(map(lambda cluster: f"{cluster.name}\n{cluster.summary}", curLevelFacetClusters)),
                        embeddingModel=embeddingModel,
                        cfg=cfg,
                        nSamplesOutsideNeighborhood=cfg.nSamplesOutsideNeighborhood)

            #### Get higher level category names ####

            print("getting higher level category names")
            tokenizer = llm.get_tokenizer()
            def getInputsFunc(clusterIndicesInNeighborhood: Sources) -> str:
                clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
                clusterNamePrompts = []
                for _ in range(2):
                    random.shuffle(clustersInNeighborhood)# shuffle ordering
                    clusterNamePrompts.append(getNeighborhoodClusterNamesPrompt(facet, tokenizer, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood))))
                return clusterNamePrompts                    
            
            def processOutputFunc(clusterIndicesInNeighborhood: Sources, clusterNamePrompts: str, clusterNamesOutputs: str) -> List[Tuple[str, Sources]]:
                # also store where it came from
                # also remove punctuation
                for clusterNamesOutput in clusterNamesOutputs:
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput)
                    if len(clusterNames) > 0:
                        return [(removePunctuation(clusterName).strip(), clusterIndicesInNeighborhood) for clusterName in clusterNames]
                '''
                # don't extract partial because stuck in loops tends to be low quality
                print("Failed, extracting partial")
                print(clusterNamePrompts[0])
                print(clusterNamesOutputs[0])
                for clusterNamesOutput in clusterNamesOutputs:
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput, ignoreNoTrailing=True)
                    # cut it off at desired because it probably got stuck in a loop and made lots of unhelpful ones
                    desired = max(1, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clusterIndicesInNeighborhood)))
                    clusterNames = clusterNames[:desired]
                    if len(clusterNames) != 0: # success at partial parsing
                        break
                '''
                print("Failed to extract any names for cluster: manually retrying")
                clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
                # shuffle ordering
                while True:
                    random.shuffle(clustersInNeighborhood)
                    prompt = getNeighborhoodClusterNamesPrompt(facet, tokenizer, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood)))
                    print(prompt)
                    nonlocal seed # we increment it so duplicate entries will get distinct things
                    seed += 1
                    samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
                    modelOutputs = llm.generate([prompt], sampling_params=samplingParams, use_tqdm=False)
                    clusterNamesOutput = [modelOutput.outputs[0].text for modelOutput in modelOutputs][0]
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput, ignoreNoTrailing=True)
                    if len(clusterNames) != 0:
                        print("Success at manual retry")
                        break
                    else:
                        print("Failed manual retry, trying again")
                        print(output)
                    return [(removePunctuation(clusterName).strip(), clusterIndicesInNeighborhood) for clusterName in clusterNames[:desired]]
            
            def dedupAndMergeSources(values: List[Tuple[str, Sources]]) -> List[Tuple[str, Sources]]:
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
            print(f"Got {len(higherCategories)} potential higher categories")

            #### Dedup higher categories ####
            
            print("getting higher level category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=[name for (name, sources) in higherCategories],
                        embeddingModel=embeddingModel,
                        cfg=cfg,
                        nSamplesOutsideNeighborhood=0) # don't grab extra outside neighborhood or our "dedup" will result in more categories, not less (as the overlap leads to double counting worded differently so they don't get deduped)
            
            print("deduping higher level categories")
            def getInputsFunc(higherCategoryIndicesInNeighborhoods: List[Tuple[str, Sources]]) -> str:
                # 0 is value, 1 is sources
                higherCategoriesInNeighborhood = [higherCategories[i][0] for i in higherCategoryIndicesInNeighborhoods]
                targetAmount =  max(1, len(higherCategoriesInNeighborhood)//2) # aim for -1 (arbitrary), but prompt lets it do more or less as needed
                if len(higherCategoriesInNeighborhood) == 2:
                    targetAmount = 2 # for only two, it'll mangle the categories if we ask it to dedup them into one, so don't do that
                return getDeduplicateClusterNamesPrompt(facet, tokenizer, higherCategoriesInNeighborhood, targetAmount)
            
            def processOutputFunc(
                    higherCategoryIndicesInNeighborhoods: List[Tuple[str, Sources]],
                    higherCategoryDedupPrompt: str,
                    higherCategoryDedupOutput: str) -> List[Tuple[str, Sources]]:
                # get sources in terms of original categories (union over all the different higher category inputs to this dedup)
                allSources = set()
                higherCategoriesInNeighborhood = [higherCategories[i] for i in higherCategoryIndicesInNeighborhoods]
                for (higherCategory, higherCategorySources) in higherCategoriesInNeighborhood:
                    allSources |= set(higherCategorySources)
                allSources = sorted(list(allSources))
                extractedOptions = extractAnswerNumberedList(higherCategoryDedupOutput)
                # fall back to dedup based on embedding (usually this means model got stuck in a loop, and it's better we ignore its outputs)
                if len(extractedOptions) == 0:
                    # no dedup extracted, falling back to dedup based on embedding 
                    extractedOptions = deduplicateByEmbeddings([cat for (cat, sources) in higherCategoriesInNeighborhood], embeddingModel=embeddingModel, tau=0.1)
                return [(removePunctuation(output).strip(), allSources) for output in extractedOptions]


            
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

            # also just dedup by embeddings as an extra check, as the deduplicateByEmbeddings above can add too many extras because categories overlap
            dedupedCategories = deduplicateByEmbeddings(
                values=dedupedCategories,
                embeddingModel=embeddingModel,
                tau=0.1,
                valueMap=lambda x: x[0]
            )
            
            print(f"Got {len(dedupedCategories)} deduped higher categories")

            #### Assign to new best fit higher-level cluster ####

            # (they didn't specify how to choose what to put here, but I figure just tracking where parents came from and using all those that might have come from x should work fine)
            print("Assigning to best fit higher-level clusters")
            baseClusterPotentialHigherLevelClusters: List[List[str]] = [[] for _ in curLevelFacetClusters]
            for category, sources in dedupedCategories:
                for sourceI in sources:
                    baseClusterPotentialHigherLevelClusters[sourceI].append(category)
            
            def getInputsFunc(facetClusterData: Tuple[ConversationCluster, List[str]]) -> List[str]:
                facetCluster, potentialHigherLevelClusters = facetClusterData
                assignToHigherCategoryPrompts = []
                for i in range(cfg.nCategorizeSamples):
                    random.shuffle(potentialHigherLevelClusters)
                    assignToHigherCategoryPrompts.append(getAssignToHighLevelClusterPrompt(tokenizer, clusterToAssign=facetCluster, higherLevelClusters=potentialHigherLevelClusters))
                return assignToHigherCategoryPrompts

            # name and summary will be generated later
            parents: Dict[str, ConversationCluster] = dict(
                [
                    (categoryName.lower().strip(), ConversationCluster(facet=facet, name=categoryName, summary="")) 
                    for (categoryName, categorySources) in dedupedCategories
                ]
            )
            
            def processOutputFunc(
                    facetClusterData: Tuple[ConversationCluster, List[str]],
                    assignToHigherCategoryPrompts: List[str],
                    assignToHigherCategoryOutput: List[str]
                ):
                facetCluster, potentialHigherLevelClusters = facetClusterData
                assignedClusters = []
                for output in assignToHigherCategoryOutput:
                    foundOutput, outputValue = extractTagValue(output, "answer")
                    # remove cluster and punctuation if it added it
                    outputValue = removePunctuation(outputValue.replace("<cluster>", "").replace("</cluster>", "").strip()).strip()
                    if foundOutput:
                        assignedClusters.append(outputValue)
                # in the embedding space, find the entry
                # bestHigherLevelClusterAssignedTo
                # in potentialHigherLevelClusters that has smallest total distance to all entries of assignedClusters
                # once we have that, bestAssignedCluster is the entry that has smallest distance to bestHigherLevelClusterAssignedTo
                # This approach helps us avoid the model slightly renaming things and helps us pick the most representative pair
                # I invented this idk what they do but this seems the obvious thing to do imo so they probably do this and just didn't say
                if len(assignedClusters) == 0:
                    # failed to extract cluster from llm, fall back to embedding of the cluster
                    assignedClusters.append(facetCluster.summary + "\n" + facetCluster.name)
                if len(potentialHigherLevelClusters) == 0:
                    print("got empty potentialHigherLevelClusters??")
                    print(assignedClusters)
                    print(potentialHigherLevelClusters)
                # lookup in embedding space the best representative pair
                # this finds term in potentialHigherLevelClusters that has smallest total distance summed over all assignedClusters
                bestAssignedCluster, bestHigherLevelClusterAssignedTo = bestRepresentativePair(assignedClusters, potentialHigherLevelClusters, embeddingModel)
                parent = parents[bestHigherLevelClusterAssignedTo.lower().strip()]
                if parent.children is None:
                    parent.children = []
                parent.children.append(facetCluster)
                facetCluster.parent = parent
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
            def getInputsFunc(parent: ConversationCluster) -> List[str]:
                renamingPrompts = []
                for _ in range(cfg.nRenameSamples):
                    random.shuffle(parent.children)
                    renamingPrompts.append(getRenamingHigherLevelClusterPrompt(facet, tokenizer, parent.children[:cfg.maxChildrenForRenaming]))
                return renamingPrompts
            
            def processOutputFunc(parent: ConversationCluster, renamePrompts: List[str], renamingOutputs: List[str]):
                # if only have one child, just copy name and summary, no need to drift
                uniqueChildren = set()
                for child in parent.children:
                    uniqueChildren.add((child.name.lower(), child.summary.lower()))
                if len(uniqueChildren) == 1:
                    child = parent.children[0]
                    parent.name = child.name
                    parent.summary = child.summary
                else:
                    summary, name = getMedoidSummaryAndName(renamingOutputs, embeddingModel)
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
            print(f"Now have {len(curLevelFacetClusters)} on level {level}")
        topLevelParents.append(curLevelFacetClusters)
    return topLevelParents

def getBaseClusters(
        facets: List[Facet],
        llm: vllm.LLM,
        embeddingModel: SentenceTransformer,
        facetValues: List[ConversationFacetData],
        facetValuesEmbeddings: List[Optional[EmbeddingArray]],
        cfg: OpenClioConfig
    ) -> Tuple[List[Optional[FaissKMeans]], List[Optional[List[ConversationCluster]]]]:
    """
    Gets the base-level clusters for all facets that have shouldMakeFacetClusters(facet) True
    Returns (listOfKMeansForFacets, listOfListOfConversationClusters)
    where 
    listOfKMeansForFacets has a kmeans (or None) for each facet,
    and
    listOfListOfConversationClusters has a list of ConversationCluster for each facet
    there will be cfg.nBaseClusters number of ConversationClusters in that list
    """
    tokenizer = llm.get_tokenizer()
    seed = cfg.seed
    kMeansFacets = [None] * len(facets)
    baseClusters = [None] * len(facets)
    for facetI, facet in enumerate(facets):
        if shouldMakeFacetClusters(facet):
            facetEmbeddings = facetValuesEmbeddings[facetI]
            n = facetEmbeddings.shape[0]
            print(f"Running kmeans for facet {facet.name}")
            kmeans = FaissKMeans(n_clusters=min(n, cfg.nBaseClustersFunc(n)), random_state=cfg.seed, **cfg.kmeansArgs)
            # we have to normalize for this to be cosine similarity
            kmeans.fit(preprocessing.normalize(facetEmbeddings))
            print("Done with kmeans, computing all distances to cluster centers")
            distancesToCenters = cdist(facetEmbeddings, kmeans.cluster_centers_)
            print("Done with distances, getting base cluster names and summaries")
            kMeansFacets[facetI] = kmeans
            with open("chonkers/basekmeansanddistances.pkl", "wb") as f:
                cloudpickle.dump((kmeans, distancesToCenters), f)

            def getInputsFunc(clusterIndex : int) -> List[str]:
                # Get points belonging to this cluster
                clusterPointsIndices = np.where(kmeans.labels_ == clusterIndex)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(cfg.maxPointsToSampleInsideCluster, clusterPointsIndices.shape[0]), replace=False)
                # Get closest points not in this cluster
                outsideClusterIndices = np.where(kmeans.labels_ != clusterIndex)[0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distancesToCenters[kmeans.labels_ != clusterIndex, clusterIndex])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(cfg.maxPointsToSampleOutsideCluster, closestPointsOutsideClusterIndices.shape[0])]

                # grab the (deduplicated) facet values
                clusterFacetValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in clusterPointsIndices])))
                clusterOutsideValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in sampledOutsideClusterIndices])))
                # generate the cluster name prompt
                clusterPrompts = []
                for _ in range(cfg.nNameDescriptionSamplesPerCluster):
                    # randomize the ordering to avoid positional biases
                    random.shuffle(clusterFacetValues)
                    random.shuffle(clusterOutsideValues)
                    prompt = getFacetClusterNamePrompt(tokenizer, facet, clusterFacetValues, clusterOutsideValues)
                    clusterPrompts.append(prompt)
                return clusterPrompts
            
            def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
                nonlocal seed
                seed += 1 # do this so repeated entries get different outputs
                samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
                try:
                    modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
                except:
                    with open("chonkers/badInputs.pkl", "wb") as f:
                        cloudpickle.dump(batchOfPrompts, f)
                    raise
                return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

            def processOutputFunc(
                    clusterIndex: int,
                    clusterPrompts: List[str],
                    clusterOutputs: List[str]
                ) -> ConversationCluster:
                    clusterPointsIndices = np.arange(len(facetEmbeddings))[kmeans.labels_ == clusterIndex]
                    summary, name = getMedoidSummaryAndName(clusterOutputs, embeddingModel)
                    return ConversationCluster(
                            facet=facet,
                            summary=summary,
                            name=name,
                            indices=clusterPointsIndices,
                        )
            facetBaseClusters = runBatched(range(len(kmeans.cluster_centers_)),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize)
            baseClusters[facetI] = facetBaseClusters
    return kMeansFacets, baseClusters
        
    

    return kMeansFacets, runBatched(list(enumerate(zip(facets, facetValuesEmbeddings))),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize)

def getFacetValuesEmbeddings(
        facets: List[Facet],
        facetValues: List[ConversationFacetData],
        embeddingModel: SentenceTransformer,
        cfg: OpenClioConfig) -> List[Optional[EmbeddingArray]]:
    """
    Gets the embeddings of all facet values that have shouldMakeFacetClusters(facet) True
    (this is when the facet has a summaryCriteria that is not None)
    Returns one element for each facet value
    That element will either be None if shouldMakeFacetClusters(facet) is False,
    or a numpy array of size [numConversations, embeddingDim]
    """
    def getInputsFunc(facetI: int) -> List[str]:
        facetInputs = []
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            for facetData in facetValues:
                facetValue = facetData.facetValues[facetI].value
                facetInputs.append(facetValue)
        return facetInputs
    
    def processBatchFunc(batchOfTextInputs: List[str]) -> List[npt.NDArray[np.float32]]:
        embedded = embeddingModel.encode(batchOfTextInputs, show_progress_bar=False)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(facetI: int, facetInputs: List[str], embeddings: List[npt.NDArray[np.float32]]) -> Optional[EmbeddingArray]:
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




def getFacetValues(
        facets: List[Facet],
        llm: vllm.LLM,
        conversations: List[List[Dict[str, str]]],
        cfg: OpenClioConfig
    ) -> List[ConversationFacetData]:
    """
    Gets facet values for every conversation, for each of the facets provided, using the provided llm.
    Returns a list of ConversationFacetData objects,
    one for each conversation
    """
    tokenizer = llm.get_tokenizer()
    def getInputsFunc(conversation: List[Dict[str, str]]) -> List[str]:
        # runBatched will automatically flatten these into us for nice batched usage,
        # then unflatten them back before calling processOutputFunc
        # so we can send in whatever sort of nested lists we want (though in this case it's only one deep)
        conversation = cfg.getConversationFunc(conversation) # map it, if needed
        inputs = []
        for facet in facets:
            if facet.getFacetPrompt is None:
                facetInput = prompts.getFacetPrompt(tokenizer, facet, conversation, cfg)
            else:
                facetInput = facet.getFacetPrompt(tokenizer, facet, conversation, cfg)
            inputs.append(facetInput)
        return inputs
    seed = cfg.seed
    def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **cfg.llmExtraInferenceArgs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(conversation: List[Dict[str, str]], conversationPrompts: List[str], facetOutputs: List[str]) -> ConversationFacetData:
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
def connectedComponentsFromMask(mask: np.ndarray) -> List[np.ndarray]:
    """
    mask: dense *boolean* adjacency matrix (n × n, symmetric, no self-loops)
    returns: list of 1-D index arrays – one per connected component
    """
    graph = csr_matrix(mask, dtype=bool)
    n_components, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == k) for k in range(n_components)]

def medoidFromEmbeddings(indices: np.ndarray,
                            embs: np.ndarray) -> int:
    """
    embs: unit-norm embeddings (n × d)
    indices: indices of the points that form one component
    returns: index (WITHIN indices) of the true medoid under cosine distance
    """
    sub = embs[indices]                       # |C| × d
    sim = cosine_similarity(sub)              # |C| × |C|
    distSums = (1.0 - sim).sum(axis=1)
    return indices[int(np.argmin(distSums))] # global index

def deduplicateByEmbeddings(
        values: List[str],
        embeddingModel: SentenceTransformer,
        tau: float = 0.15,          # distance threshold (0.15 ≈ cosine ≥ 0.85)
        valueMap: Optional[Callable[[Any], str]] = None
) -> List[str]:
    """
    Single-link deduplication.  Returns one representative per duplicate set,
    chosen as the exact medoid of each connected component.
    """
    if len(values) == 0:
        return []

    # 1. Embed once, L2-normalise so cosine == dot product
    valuesAsStr = list(map(valueMap, values)) if valueMap is not None else values
    emb = preprocessing.normalize(embeddingModel.encode(valuesAsStr, show_progress_bar=False))

    # 2. Dense distance matrix  (O(n²) memory!)
    sim = cosine_similarity(emb)
    dist = 1.0 - sim

    # 3. Boolean adjacency under threshold (no self-loops)
    mask = (dist <= tau) & ~np.eye(len(values), dtype=bool)

    # 4. Connected components (single-link duplicate sets)
    components = connectedComponentsFromMask(mask)

    # 5. Medoid for every component
    representatives = []
    for comp in components:
        if comp.size == 1:
            representatives.append(values[comp[0]])
        else:
            medoid_idx = medoidFromEmbeddings(comp, emb)
            representatives.append(values[medoid_idx])

    return representatives

# from https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def setSeed(seed):
    """
    Set seeds (to lots of things)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def bestRepresentativePair(
    A: List[str],
    B: List[str],
    model: SentenceTransformer
) -> Tuple[str, str]:
    """
    Return (a_star, b_star) where:
      • b_star  minimises Σ_{a∈A} (1 - cos(a, b))
      • a_star  is the element of A closest to that b_star
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("A and B must be non-empty")

    # 1 . Encode & L2-normalise so cosine == dot product
    AEmb = preprocessing.normalize(model.encode(A, convert_to_numpy=True, show_progress_bar=False))
    BEmb = preprocessing.normalize(model.encode(B, convert_to_numpy=True, show_progress_bar=False))

    # 2 . Cosine similarity matrix  (n × m)
    sim = cosine_similarity(AEmb, BEmb)         # fast, vectorised

    # 3 . For every b_j compute total distance to all a_i
    distSumsForEachB = (1.0 - sim).sum(axis=0)           # shape (m,)

    closestB = int(np.argmin(distSumsForEachB))
    bStar = B[closestB]

    # 4 . Find a_i closest to that bStar            
    distOfEachAToBStar = 1.0 - sim[:, closestB]                 # shape (n,)
    bestAIndex = int(np.argmin(distOfEachAToBStar))
    aStar = A[bestAIndex]

    return aStar, bStar

def getMedoidViaEmbeddings(
    values: List[str],
    embeddingModel: SentenceTransformer
) -> str:
    """
    Return the element of `values` that minimises the sum of
    cosine distances ( = 1 - cosine similarity ) to all others.

    This is the exact medoid under cosine distance.
    Complexity:  O(n²) in time,  O(n²) in memory.
    """
    if len(values) == 0:
        raise ValueError("`values` must contain at least one string.")

    # 1. Embed and L2-normalise so that
    #    cosine(u, v) == u @ v   (dot product after normalisation)
    embeddings = preprocessing.normalize(
        embeddingModel.encode(values, convert_to_numpy=True, show_progress_bar=False)
    )                                # shape = (n, d)

    # 2. Pair-wise cosine similarity matrix  (n × n)
    #    sim[i, j] = cosine( values[i], values[j] )
    sim = cosine_similarity(embeddings)           # fast & vectorised

    # 3. Convert similarity → distance  and add up per row
    #    dist = 1 - sim   (because vectors are unit-norm)
    dist_sums = (1.0 - sim).sum(axis=1)           # shape = (n,)

    # 4. Index of smallest total distance = medoid
    medoid_idx = int(np.argmin(dist_sums))
    return values[medoid_idx]

def getCentroidViaEmbeddings(
    values: List[str],
    embeddingModel: SentenceTransformer) -> str:
    """
    Computes the average of the values embeddings,
    then finds the value that is closest to that average
    This is sort of a continuous version of "pick the most common element"
    But actually what we want is the medoid (term that is closest to all the others)
    """
    normalizedValues = preprocessing.normalize(embeddingModel.encode(values, show_progress_bar=False))
    avg = normalizedValues.mean(axis=0)
    sims = cosine_similarity(wow, wow.mean(axis=0).reshape(1, -1)).flatten()
    return values[np.argmax(sims)]

def getMedoidSummaryAndName(outputs: List[str], embeddingModel: SentenceTransformer) -> Tuple[str, str]:
    """
    Continuous version of "get most common"
    That gets the embedded value that is closest to all other items (the medoid)
    returns (summary, name)
    """
    summaries = []
    names = []
    for output in outputs:
        # re.DOTALL makes . match newlines too (by default it does not)
        matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
        if len(matches) > 0:
            for summary, name in matches:
                summaries.append(removePunctuation(summary.strip()))
                names.append(removePunctuation(name.strip()))
    # remove empty strings
    summaries = [summary for summary in summaries if len(summary) > 0]
    names = [name for name in names if len(name) > 0]
    if len(summaries) == 0: summaries.append("<Could not extract summary>")
    if len(names) == 0: names.append("<Could not extract name>")
    return getMedoidViaEmbeddings(summaries, embeddingModel), getMedoidViaEmbeddings(names, embeddingModel)

def getMostCommonSummaryAndName(outputs: List[str]) -> Tuple[str, str]:
    """
    Gets most common thing in <summary> tag and <name> tag,
    returns (summary, name)
    I recommend to use getMedoidSummaryAndName so we act in embedding space instead
    """
    summaryCounts = defaultdict(lambda: 0)
    nameCounts = defaultdict(lambda: 0)
    for output in outputs:
        # re.DOTALL makes . match newlines too (by default it does not)
        matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
        if len(matches) > 0:
            for summary, name in matches:
                summaryCounts[cleanTrailingTagsInOutput(summary)] += 1
                nameCounts[cleanTrailingTagsInOutput(name)] += 1
    def largestCountItem(counts, fieldName):
        if len(counts) == 0: return f"<Could not extract {fieldName}>"
        counts = sorted(list(counts.items()), key=lambda x: (-x[1], x[0]))
        largestKey, largestValue = counts[0]
        return largestKey
    summary = largestCountItem(summaryCounts, "summary")
    name = largestCountItem(nameCounts, "name")
    return summary, name

def removePunctuation(output: str) -> str:
    """
    Removes ., ?, and ! from end of a string.
    (and strips it before and afterwards)
    """
    output = output.strip()
    if output.endswith("."):
        output = output[:-1].strip()
    if output.endswith("?"):
        output = output[:-1].strip()
    if output.endswith("!"):
        output = output[:-1].strip()
    return output

def extractTagValue(output: str, tag: str) -> Tuple[bool, str]:
    """
    Gets value contained in <tag>VALUE_HERE</tag>
    returns (foundTag, valueInTag)
    where foundTag is True if the tag was found
    """
    posOfTag = output.lower().find(f"<{tag}>")
    if posOfTag != -1:
        output = output[posOfTag + len(f"<{tag}>"):].strip()
    endOfTagPos = output.lower().find(f"</{tag}>")
    if endOfTagPos != -1:
        output = output[:endOfTagPos].strip()
    return posOfTag != -1 and endOfTagPos != -1, output

def extractAnswerNumberedList(output: str, ignoreNoTrailing: bool = False) -> List[str]:
    """
    If we have
    <answer>
    1. blah
    2. blahhh
    3. wow
    etc.
    </answer>
    This will return 
    ["blah", "blahhh", "wow", ...]
    """
    results = []
    foundAnswerTag, answer = extractTagValue(output, "answer")
    if foundAnswerTag:
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) > 0]
    elif ignoreNoTrailing:
        posOfTag = output.lower().find("<answer>")
        if posOfTag != -1:
            output = output[posOfTag + len(f"<answer>"):].strip()
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) > 0]
        results = results[:-1] # ignore last one since it's probably partially formed, we got cut off early
    return results

def removeNumberFromOutput(output: str) -> str:
    """
    Removes number. from the front of the output
    Like 
    "1. hi"
    becomes
    "hi"
    or
    "144. wow"
    becomes
    "wow"
    """
    return re.sub(r"^\d*?\.", "", output.strip(), count=1).strip()

def cleanTrailingTagsInOutput(output: str) -> str:
    """
    Removes any trailing </tag> that may existing in the output
    Also strips it before and afterwards
    """
    return re.findall(r"(.*?)(?:(?:</)|$)", output.strip(), re.DOTALL)[0].strip()


def printHierarchyHelper(
    parents: List[ConversationCluster],
    indent: str) -> List[str]:
    lines = []
    for parent in parents:
        lines.append(indent + parent.name)
        if not parent.children is None:
            lines += printHierarchyHelper(parent.children, indent + "  ")
    return lines

def printHierarchy(parents: List[ConversationCluster]):
    """
    helper function to manually print hierarchy of a specific facet
    """
    resLines = printHierarchyHelper(parents, indent="")
    print("\n".join(resLines))
    with open("hierarchy.txt", "w") as f:
        f.write("\n".join(resLines))