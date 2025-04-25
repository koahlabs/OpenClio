from prompts import getFacetPrompt
from dataclasses import dataclass
from utils import flatten, unflatten, runBatched
import vllm
import pandas as pd
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class FacetExtraInfo:
    minValue: int
    maxValue: int

@dataclass
class Facet:
    name: str
    question: str
    prefill: str
    numeric: bool

facets = [
    Facet(
        name="Request",
        question="What is the user’s overall request for the assistant?",
        prefill="The user’s overall request for the assistant is to",
        numeric=False,
    ),
    Facet(
        name="Language",
        question="What are the main languages of the conversation? Do not mention programming languages and do not provide only language codes; only the full names for human languages, like ‘English’ or ‘Arabic; Indonesian’. Do not include language families (just provide the general language, e.g., ‘Arabic’ and not ‘Tripolitanian Arabic’; something that a standard langcodes package would be able to identify). Only consider languages in or after the human’s first request. Output each language as a single full word with no other commentary.",
        prefill="",
        numeric=False,
    ),
    Facet(
        name="Task",
        question="What task is the model being asked to perform in this conversation?",
        prefill="The task is to",
        numeric=False,
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
        prefill="",
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
    

def getModels():
    model_str = "Qwen/Qwen2.5-7B-Instruct"
    llm = vllm.LLM(model=model_str)
    embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return llm, embeddingModel


def getData():
    d = pd.read_parquet("train-00000-of-00006.parquet", engine="pyarrow")
    return [d.iloc[i].conversation for i in range(len(d))]


def runClio(llm, embeddingModel, conversations, llmBatchSize, embedBatchSize, **kwargs):
    conversationsFacets = getFacets(llm, llm.get_tokenizer(), conversations, llmBatchSize, **kwargs)
    conversationsEmbedings = getEmbeddings(conversationsFacets, embeddingModel, embedBatchSize)
    return conversationsFacets, conversationsEmbedings


def getEmbeddings(conversationsFacets, embeddingModel, batchSize):
    def getInputsFunc(conversationFacetData : ConversationFacetData):
        resultInputs = []
        for facetValue in conversationFacetData.facetValues:
            facetInputArr = []
            if facetValue.facet.numeric == False:
                facetInputArr.append(facetValue.value)
            resultInputs.append(facetInputArr)
        return resultInputs
    
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(conversationFacetData, facetInputs, embeddings):
        resultEmbeddings = []
        for facetValue, outputEmbeddings in zip(conversationFacetData.facetValues, embeddings):
            if facetValue.facet.numeric == False:
                resultEmbeddings.append(outputEmbeddings[0])
            else:
                minValueInclusive, maxValueInclusive = facetValue.facet.numeric
                try:
                    intValue = int(facetValue.value.split()[0])
                except ValueError:
                    intValue = maxValueInclusive + 1
                # 1 so it's exclusive, 2 because we also have unknown value
                outputEmbeddings2 = np.zeros([maxValueInclusive + 2 - minValueInclusive])
                outputEmbeddings2[:] = -1
                outputEmbeddings2[intValue-minValueInclusive] = 1
                # normalize
                outputEmbeddings2 /= np.sum(outputEmbeddings2)
                resultEmbeddings.append(outputEmbeddings2)
        return np.concatenate(resultEmbeddings)
    
    return runBatched(conversationsFacets,
                                       getInputs=getInputsFunc,
                                       processBatch=processBatchFunc,
                                       processOutput=processOutputFunc,
                                       batchSize=batchSize)


def getFacets(llm, tokenizer, conversations, batchSize, **kwargs):
    def getInputsFunc(conversation):
        conversationStr = conversationToString(conversation)
        # runBatched will automatically flatten these into us for nice batched usage,
        # then unflatten them back before calling processOutputFunc
        # so we can send in whatever sort of nested lists we want (though in this case it's only one deep)
        return [getFacetPrompt(tokenizer, conversationStr, facet.question, facet.prefill) for facet in facets]

    samplingParams = vllm.SamplingParams(**kwargs)
    def processBatchFunc(batchOfPrompts):
        modelOutputs = llm.generate(batchOfPrompts, sampling_params=samplingParams, use_tqdm=False)
        return [modelOutput.outputs[0].text for modelOutput in modelOutputs]

    def processOutputFunc(conversation, conversationPrompts, facetOutputs):
        facetValues = [FacetValue(facet=facet, value=value) for (facet, value) in zip(facets, facetOutputs)]
        return ConversationFacetData(
            conversation=conversation,
            facetValues=facetValues
        )

    return runBatched(conversations,
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=batchSize)


