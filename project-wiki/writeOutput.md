# Table of Contents

* [writeOutput](#writeOutput)
  * [convertOutputToWebpage](#writeOutput.convertOutputToWebpage)

<a id="writeOutput"></a>

# writeOutput

<a id="writeOutput.convertOutputToWebpage"></a>

#### convertOutputToWebpage

```python
def convertOutputToWebpage(output: OpenClioResults,
                           rootHtmlPath: str,
                           targetDir: str,
                           maxSizePerFile: int,
                           conversationFilter: Callable[
                               [List[Dict[str, str]], ConversationFacetData],
                               bool] = None,
                           dataToJson: Callable[[Any], Dict[str, Any]] = None,
                           verbose=True)
```

Converts the given output to a static webpage and json files, dumped to targetDir
It's split up into multiple json files, each of max size maxSizePerFile, and streamed as needed
Keyword arguments:
output -- The openclio results outputs
rootHtmlPath -- the relative path the html page will be stored on your server (something like "/clioresults")
targetDir -- where we dump all the files
maxSizePerFile -- the maximum size of each json data file, in bytes. I recommend 10000000
conversationFilter -- a filter that takes (conversation, facetData) and return bool if it should be included on the webpage
dataToJson -- Takes a data point and returns a json of the corresponding conversation. It should look like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]. If you just want to dispaly the data as a string, just return a single entry like this: [{"role": "<whatever you want>", "content": "<your str content>"}]

