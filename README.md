# OpenClio
Open source version of Anthropic's Clio: A system for privacy-preserving insights into real-world AI use

How do I use?

```
pip install git+https://github.com/tangentlabs/django-oscar-paypal.git
```

```python
import openclio as clio
import vllm
from sentence_transformers import SentenceTransformer

# load 10000 wildchat conversations
data = clio.getExampleData()
# Load models
llm = vllm.LLM(model=model_str)
embeddingModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Run clio and output to website
outputDirectory = "output"
outputWebsitePath = "/clioResults"
clio.runClio(facets=clio.mainFacets, llm=llm, embeddingModel=embeddingModel, data=data, outputDirectory=outputDirectory, htmlRoot=outputWebsitePath)

# Run webui (optional, you can also just take the files at output/clioResults and put them on your own website (at /clioResults), it's just a static website)
import 
```