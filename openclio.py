import prompts
from prompts import getFacetPrompt, getFacetClusterNamePrompt
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
def runClio(facets, llm, embeddingModel, conversations, llmBatchSize, embedBatchSize, numberOfBaseClusters, nPointsToSample, nLLMSamplesPerCluster, seed, **kwargs):
    np.random.seed(seed)
    print("Getting facets")
    conversationsFacets = getFacets(
        facets=facets, 
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        conversations=conversations,
        batchSize=llmBatchSize,
        seed=seed, **kwargs)
    print("Getting embeddings")
    conversationsEmbedings = getEmbeddings(
        facets=facets,
        conversationsFacets=conversationsFacets,
        embeddingModel=embeddingModel,
        batchSize=embedBatchSize)
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
    return conversationsFacets, conversationsEmbedings, kMeans, baseClusters

def getBaseClusters(facets, llm, conversationsFacets, conversationsEmbedings, numberOfBaseClusters, seed, nPointsToSample, nLLMSamplesPerCluster, batchSize, **kwargs):

    tokenizer = llm.get_tokenizer()
    kMeansFacets = []
    def getInputsFunc(facetAndEmbeddings):
        lookupClusterPrompts = []
        facetI, (facet, facetEmbeddings) = facetAndEmbeddings
        if shouldMakeFacetEmbedding(facet):
            print(f"Running kmeans for facet {facet}")
            kmeans = KMeans(n_clusters=numberOfBaseClusters, random_state=seed)
            kmeans.fit(facetEmbeddings)
            distances = cdist(facetEmbeddings, kmeans.cluster_centers_)
            kMeansFacets.append(kmeans)

            # For each cluster, find the index of the closest point
            for cluster_idx in range(len(kmeans.cluster_centers_)):
                # Get points belonging to this cluster
                clusterPointsIndices = np.arange(len(conversationsFacets))[kmeans.labels_ == cluster_idx]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(nPointsToSample, clusterPointsIndices.shape[0]), replace=False)
                # Get closest points not in this cluster
                closestPointsOutsideClusterIndices = np.argsort(distances[kmeans.labels_ != cluster_idx, cluster_idx])
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(nPointsToSample, clusterPointsIndices.shape[0])]
                clusterFacetValues = [conversationsFacets[i].facetValues[facetI].value for i in clusterPointsIndices]
                clusterOutsideValues = [conversationsFacets[i].facetValues[facetI].value for i in sampledOutsideClusterIndices]
                print(clusterFacetValues)
                clusterPrompts = []
                for _ in range(nLLMSamplesPerCluster):
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
        if shouldMakeFacetEmbedding(facet):
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

def shouldMakeFacetEmbedding(facet):
    return facet.summaryCriteria is not None

def getEmbeddings(facets, conversationsFacets, embeddingModel, batchSize):
    def getInputsFunc(conversationFacetData : ConversationFacetData):
        resultInputs = []
        for facetValue in conversationFacetData.facetValues:
            facetInputArr = []
            if shouldMakeFacetEmbedding(facetValue.facet):
                facetInputArr.append(facetValue.value)
            resultInputs.append(facetInputArr)
        return resultInputs
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(conversationFacetData, facetInputs, embeddings):
        return embeddings
    
    outputEmbeddings = runBatched(conversationsFacets,
                                       getInputs=getInputsFunc,
                                       processBatch=processBatchFunc,
                                       processOutput=processOutputFunc,
                                       batchSize=batchSize)
    # make one large numpy array for each facet that has embeddings
    numFacets = len(facets)
    resultEmbeddings = [[] for _ in range(numFacets)]
    for facetData, embeddings in zip(conversationsFacets, outputEmbeddings):
        for facetI, (facetValue, embedding) in enumerate(zip(facetData.facetValues, embeddings)):
            if shouldMakeFacetEmbedding(facetValue.facet):
                resultEmbeddings[facetI].append(embedding[0])
    for facetI in range(numFacets):
        if len(resultEmbeddings[facetI]) > 0:
            resultEmbeddings[facetI] = np.stack(resultEmbeddings[facetI])

    return resultEmbeddings
        
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


