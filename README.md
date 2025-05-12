# OpenClio
Open source version of Anthropic's Clio: A system for privacy-preserving insights into real-world AI use

How do I use?

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
clio.runClio(facets=clio.mainFacets, llm=llm, embeddingModel=embeddingModel, data=data, outputDirectory=outputDirectory, htmlRoot=outputWebsitePath)
```

That'll provide a link for you, go there, and you should see your cleo outputs!

![Tree view](https://github.com/Phylliida/OpenClio/blob/main/project-wiki/assets/exampleHierarchy.png?raw=true)

![Conversation View](https://github.com/Phylliida/OpenClio/blob/main/project-wiki/assets/exampleChat.png?raw=true)

As you browse, the hash of the website will be modified. This lets you share specific conversations or tree states via URL.

You can also put the outputted files into your own website, it's just a single static html file that loads json files.

To see what other parameters you can pass to runClio, see the docs for [https://github.com/Phylliida/OpenClio/blob/main/project-wiki/opencliotypes.md#opencliotypes.OpenClioConfig](OpenClioConfig). Any of these fields can be passed into runClio and they will be [https://github.com/Phylliida/OpenClio/blob/main/openclio/openclio.py#L118](used).