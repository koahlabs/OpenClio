# OpenClio
Open source version of [Anthropic's Clio: A system for privacy-preserving insights into real-world AI use](https://www.anthropic.com/research/clio)

Designed to run using local LLMs via [VLLM](https://github.com/vllm-project/vllm).

See an example run of this on ~400,000 english conversations from Wildchat [here](https://www.phylliida.dev/modelwelfare/wildchat/).

## How do I use?

```
pip install git+https://github.com/Phylliida/OpenClio.git
```

```python
import openclio as clio
import vllm
from sentence_transformers import SentenceTransformer

# load 10000 wildchat conversations
data = clio.getExampleData()
# Load models
llm = vllm.LLM(model="Qwen/Qwen3-8B")
embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Run clio, output to static website, and run webui
outputDirectory = "output"
outputWebsitePath = "/clioResults"
# keep in mind VLLM doesn't like to be interrupted with ctrl-c and will hang, so you can just press c if you are in the console and it'll listen and break
clio.runClio(facets=clio.mainFacets, llm=llm, embeddingModel=embeddingModel, data=data, outputDirectory=outputDirectory, htmlRoot=outputWebsitePath)
```

That'll provide a link for you, go there, and you should see your clio outputs!

![Tree view](https://github.com/Phylliida/OpenClio/blob/main/project-wiki/assets/exampleHierarchy.png?raw=true)

![Conversation View](https://github.com/Phylliida/OpenClio/blob/main/project-wiki/assets/exampleChat.png?raw=true)

As you browse, the hash of the website will be modified. This lets you share specific conversations or tree states via URL.

You can also put the outputted files into your own website, it's just a single static html file that loads json files.

The data is split up into many compressed chunks (determined by [htmlMaxSizePerFile](/project-wiki/opencliotypes.md#opencliotypes.OpenClioConfig.htmlMaxSizePerFile), by default 10MB chunks) and streamed as the user browses the tree.

To see what other parameters you can pass to [runClio](project-wiki/openclio.md#openclio.runClio), see the docs for [OpenClioConfig](project-wiki/opencliotypes.md#opencliotypes.OpenClioConfig). Any of these fields can be passed into runClio and they will be [used](openclio/openclio.py#L118).

## What if I want to categorize non-conversation data?

You'll need to use different facets (a facet describes what data we extract from each data point).

clio.mainFacets looks like this

```python
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
```

These use standard prompts defined in [openclio/prompts.py](https://github.com/Phylliida/OpenClio/blob/main/openclio/prompts.py) from the paper, which assume you are working with conversations.

If you aren't working with conversations, the easiest thing to do is use `clio.genericSummaryFacets`, which looks like this

```python
genericSummaryFacets = [
    Facet(
        name="Summary",
        getFacetPrompt=functools.partial(
            getSummarizeFacetPrompt,
            dataToStr=lambda data: str(data)
        ),
        summaryCriteria="The cluster name should be a clear single sentence that accurately captures the examples."
    )
]
```

Where `getSummarizeFacetPrompt` looks like this

```python
from openclio import doCachedReplacements, Facet, OpenClioConfig
from typing import Callable, Dict, Any
def getSummarizeFacetPrompt(tokenizer, facet: Facet, data: Any, cfg: OpenClioConfig, dataToStr: Callable[[Any], str], tokenizerArgs: Dict[str, Any]) -> str:
    return doCachedReplacements(
        funcName="getSummarizeFacetPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """Please summarize the provided data in a single sentence:

<data>
{dataREPLACE}
</data>

Put your answer in this format:

<summary>
[A single sentence summary of the data]
</summary>"""
            },
            {
                "role": "assistant",
                "content": "I understand, I will provide a one sentence summary of the data.\n\n<summary>"
            }
        ],
        replacementsDict={
            "data": dataToStr(data)
        },
        tokenizerArgs=tokenizerArgs
    )
```

[`doCachedReplacements`](project-wiki/prompts.md#prompts.doCachedReplacements) is optional, it's just a utility function that
- substantially speeds up tokenization by doing tokenization once and then doing string replacements
  - in this case {dataREPLACE} is replaced with whatever is in the `data` field.
- Uses `tokenizer.apply_chat_template` and then converts the tokens back to a string

You can see the code [here](https://github.com/Phylliida/OpenClio/blob/main/openclio/prompts.py#L9) it's fairly simple.

In general, your function should just return the string that is then later passed into llm.generate.

Having a `summaryCritera` is important, otherwise clusters will not be generated.

## Related Work and Citations

- [Kura](https://github.com/ivanleomk/kura) is a seperate implementation of some parts of Clio.
- [Wildchat](https://wildchat.allen.ai/) is the data I used when testing/on that website.
