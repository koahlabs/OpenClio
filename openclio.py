import prompts
from prompts import getFacetPrompt, getFacetClusterNamePrompt, getNeighborhoodClusterNamesPrompt
from dataclasses import dataclass
from utils import flatten, unflatten, runBatched
import vllm
import pandas as pd
from typing import Any, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
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

def conversationToString(conversation):
    return "\n".join([f"{turn['role']}:\n{turn['content']}" for turn in conversation])

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
    indices: np.ndarray[np.int32]

def getModels():
    model_str = "Qwen/Qwen2.5-7B-Instruct"
    llm = vllm.LLM(model=model_str)
    embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return llm, embeddingModel


def getData():
    d = pd.read_parquet("train-00000-of-00006.parquet", engine="pyarrow")
    return [d.iloc[i].conversation for i in range(len(d))]

# facets, embeds, kmeans, clusters = clio.runClio(clio.facets, llm, embed, d[0:10000], llmBatchSize=1000, embedBatchSize=1000, numberOfBaseClusters=1000, nPointsToSample=10, nLLMSamplesPerCluster=5, seed=27, max_tokens=1000)
def runClio(facets, 
            llm,
            embeddingModel,
            conversations,
            llmBatchSize,
            embedBatchSize,
            numberOfBaseClusters,
            nPointsToSample,
            nLLMSamplesPerCluster,
            nClustersOutside, # idk what they picked they didn't say
            desiredNames, # idk what they picked they didn't say
            seed,
            conversationsFacets=None,
            conversationsEmbedings=None,
            **kwargs):
    np.random.seed(seed)
    random.seed(seed)
    print("Getting facets")
    conversationsFacets = getFacets(
        facets=facets, 
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        conversations=conversations,
        batchSize=llmBatchSize,
        seed=seed, **kwargs) if conversationsFacets is None else conversationsFacets
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
        nPointsToSample=nPointsToSample,
        nLLMSamplesPerCluster=nLLMSamplesPerCluster,
        batchSize=llmBatchSize,
        seed=seed,
        **kwargs)
    print("Getting higher level clusters")
    higherCategories = getHierarchy(
        facets=facets,
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        embeddingModel=embeddingModel,
        baseClusters=baseClusters,
        nClustersOutside=nClustersOutside,
        nLLMSamplesPerCluster=nLLMSamplesPerCluster,
        desiredNames=desiredNames,
        seed=seed
    )

    return conversationsFacets, conversationsEmbedings, kMeans, baseClusters, higherCategories

def getHierarchy(facets, llm, tokenizer, embeddingModel, baseClusters, nClustersOutside, nLLMSamplesPerCluster, desiredNames, seed):
    
    curLevelClusters = baseClusters
    #### Get embeddings for clusters ####
    def getInputsFunc(facetI):
        facet = facets[facetI]
        clusterEmbedPrompts = []
        if shouldMakeFacetClusters(facet):
            clusters = curLevelClusters[facetI]
            for cluster in clusters:
                clusterEmbedPrompts.append(f"{cluster.name}\n{cluster.summary}")
        return clusterEmbedPrompts
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(facetI, clusterEmbedPrompts, outputEmbeddings):
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            return np.stack(outputEmbeddings)
        else:
            return None
    level = 0
    print(f"Embedding higher level cluster on level {level}")
    clusterEmbeddings = runBatched(list(range(len(facets))),
                                   getInputs=getInputsFunc,
                                   processBatch=processBatchFunc,
                                   processOutput=processOutputFunc)
    

    #### Get higher level category names ####
    kmeansOutputs = [None for _ in range(len(facets))]
    def getInputsFunc(facetI):
        facet = facets[facetI]
        allClusterPrompts = []
        if shouldMakeFacetClusters(facet):
            print(f"Finding neighbors for facet {facet.name} on level {level}")
            facetClusterEmbeddings = clusterEmbeddings[facetI]
            numClusters = facetClusterEmbeddings.shape[0]
            # from G.7 we want about 40 per item
            newK = numClusters // 40
            kmeans = KMeans(n_clusters=newK, random_state=seed)
            kmeans.fit(facetClusterEmbeddings)
            kmeansOutputs[facetI] = kmeans
            distances = cdist(facetClusterEmbeddings, kmeans.cluster_centers_)
            for cluster_idx in range(len(kmeans.cluster_centers_)):
                # Get points belonging to this cluster
                clusterPointsIndices = np.where(kmeans.labels_ == cluster_idx)[0]
                # Get closest points not in this cluster
                outsideClusterIndices = np.where(kmeans.labels_ != cluster_idx)[0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != cluster_idx, cluster_idx])]
                # From G.7:
                # "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters)
                #  on the boundary between neighborhoods are neither overcounted nor undercounted"
                clusterIndicesInNeighborhood = list(clusterPointsIndices) + list(closestPointsOutsideClusterIndices[:nClustersOutside])
                clustersInNeighborhood = [curLevelClusters[i] for i in clusterIndicesInNeighborhood]
                clusterPrompts = []
                for _ in range(nLLMSamplesPerCluster):
                    # shuffle ordering
                    random.shuffle(clustersInNeighborhood)
                    # idk desiredNames
                    clusterPrompts.append(getNeighborhoodClusterNamesPrompt(facet, tokenizer, clustersInNeighborhood, desiredNames))
                
                allClusterPrompts.append(clusterPrompts)
        
    
    def processBatchFunc(clusterPrompts):
        nonlocal seed
        seed += 1
        samplingParams = vllm.SamplingParams(seed=seed, **kwargs)
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]
    
    def processOutputFunc(facetI, clusterPrompts, clusterNamesOutputs):
        facet = facets[facetI]
        higherCategories = []
        if shouldMakeFacetClusters(facet): 
            kmeans = kmeansOutputs[facetI]
            for cluster_idx, clusterOutputs in zip(range(len(kmeans.cluster_centers_)), clusterNamesOutputs):
                for output in clusterOutputs:
                    posOfAnswer = output.lower().find("<answer>")
                    if posOfAnswer != -1:
                        posOfAnswer = cleanOutput(output[posOfAnswer + len("<answer>"):])
                        answers = [re.sub("^\d*?\.", "", line.strip(), count=1) for line in posOfAnswer.split("\n") if len(line.strip()) >= 0]
                        for answer in answers:
                            higherCategories.apppend(answer)
        return higherCategories

    higherCategories = runBatched(list(range(len(facets))),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=batchSize)

    return higherCategories
    


    # list(enumerate(zip(facets, baseClusters)))



def getBaseClusters(facets, llm, conversationsFacets, conversationsEmbedings, numberOfBaseClusters, seed, nPointsToSample, nLLMSamplesPerCluster, batchSize, **kwargs):
    tokenizer = llm.get_tokenizer()
    kMeansFacets = []
    def getInputsFunc(facetAndEmbeddings):
        lookupClusterPrompts = []
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetClusters(facet):
            print(f"Running kmeans for facet {facet.name}")
            kmeans = KMeans(n_clusters=numberOfBaseClusters, random_state=seed)
            kmeans.fit(facetEmbeddings)
            distances = cdist(facetEmbeddings, kmeans.cluster_centers_)
            kMeansFacets.append(kmeans)

            # For each cluster, find the index of the closest point
            for cluster_idx in range(len(kmeans.cluster_centers_)):
                # Get points belonging to this cluster
                clusterPointsIndices = np.where(kmeans.labels_ == cluster_idx)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(nPointsToSample, clusterPointsIndices.shape[0]), replace=False)
                # Get closest points not in this cluster
                outsideClusterIndices = np.where(kmeans.labels_ != cluster_idx)[0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != cluster_idx, cluster_idx])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(nPointsToSample, clusterPointsIndices.shape[0])]

                # grab the facet values
                clusterFacetValues = [conversationsFacets[i].facetValues[facetI].value for i in clusterPointsIndices]
                clusterOutsideValues = [conversationsFacets[i].facetValues[facetI].value for i in sampledOutsideClusterIndices]
                # generate the cluster name prompt
                clusterPrompts = []
                for _ in range(nLLMSamplesPerCluster):
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
            for cluster_idx, clusterOutputs in zip(range(len(kmeans.cluster_centers_)), outputs):
                clusterPointsIndices = np.arange(len(conversationsFacets))[kmeans.labels_ == cluster_idx]
                summaryCounts = defaultdict(lambda: 0)
                nameCounts = defaultdict(lambda: 0)
                for output in clusterOutputs:
                    # re.DOTALL makes . match newlines too (by default it does not)
                    matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
                    if len(matches) > 0:
                        for summary, name in matches:
                            summaryCounts[cleanOutput(summary)] += 1
                            nameCounts[cleanOutput(name)] += 1
                def largestCountItem(counts):
                    if len(counts) == 0: return "<Could not extract summary>"
                    counts = sorted(list(counts.items()), key=lambda x: (-x[1], x[0]))
                    largestKey, largestValue = counts[0]
                    return largestKey
                summary = largestCountItem(summaryCounts)
                name = largestCountItem(nameCounts)
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

def getEmbeddings(facets, conversationsFacets, embeddingModel, batchSize):
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
                    batchSize=batchSize)

def cleanOutput(output):
    return re.findall(r"(.*?)(?:(?:</)|$)", output.strip())[0].strip()

def getFacets(facets, llm, tokenizer, conversations, batchSize, seed, **kwargs):
    def getInputsFunc(conversation):
        conversationStr = conversationToString(conversation)
        # runBatched will automatically flatten these into us for nice batched usage,
        # then unflatten them back before calling processOutputFunc
        # so we can send in whatever sort of nested lists we want (though in this case it's only one deep)
        return [getFacetPrompt(tokenizer, conversationStr, facet.question, facet.prefill) for facet in facets]

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


