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

That'll provide a link for you, go there, and you should see your cleo outputs! You can also put the files into your own website.