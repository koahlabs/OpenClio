from prompts import getFacetPrompt
from dataclasses import dataclass
from utils import flatten, unflatten, runBatched
import vllm
from typing import Any

@dataclass
class Facet:
    name: str
    question: str
    prefill: str

facets = [
    Facet(
        name="Request",
        question="What is the user’s overall request for the assistant?",
        prefill="The user’s overall request for the assistant is to",
    ),
    Facet(
        name="Language",
        question="What are the main languages of the conversation? Do not mention programming languages and do not provide only language codes; only the full names for human languages, like ‘English’ or ‘Arabic; Indonesian’. Do not include language families (just provide the general language, e.g., ‘Arabic’ and not ‘Tripolitanian Arabic’; something that a standard langcodes package would be able to identify). Only consider languages in or after the human’s first request. Output each language as a single full word with no other commentary.",
        prefill="",
    ),
    Facet(
        name="Task",
        question="What task is the model being asked to perform in this conversation?",
        prefill="The task is to"
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
    )
]

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


