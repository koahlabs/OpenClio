from typing import List, Dict, Tuple, Callable, Any
import os
import json
from collections import defaultdict
from pathlib import Path
from .opencliotypes import Facet, FacetValue, ConversationFacetData, ConversationEmbedding, ConversationCluster, OpenClioConfig, OpenClioResults, EmbeddingArray, shouldMakeFacetClusters

# 0 is root/highest level
def getAllClustersAtLevel(facetI: int, output: OpenClioResults, level: int):
    def getAllClustersAtLevelHelper(cluster: ConversationCluster, targetLevel: int):
        if targetLevel == 0:
            yield cluster
        else:
            if cluster.children is not None:
                for child in cluster.children:
                    for childCluster in getAllClustersAtLevelHelper(cluster=child, targetLevel=targetLevel-1):
                        yield childCluster
    for rootCluster in output.rootClusters[facetI]:
        for resultCluster in getAllClustersAtLevelHelper(rootCluster, targetLevel=level):
            yield resultCluster

# we are guaranteed every cluster except base level is non empty so we can just dfs
def getNumLevels(output: OpenClioResults, facetI: int):
    def getDepthHelper(currentItem):
        if currentItem.children is None:
            return 1
        else:
            return getDepthHelper(currentItem.children[0])+1
    return getDepthHelper(output.rootClusters[facetI][0])

# we want to gather stuff together until we reach maxSize
def encodeFacetDataInChunks(facetI: int, output: OpenClioResults, maxSizePerFile: int, dataToJson: Callable[[Any], Dict[str, Any]], outputPath: str, htmlRoot: str, verbose: bool):
    fileMap = {}

    facet = output.facets[facetI]

    facetJson = {
        "name": facet.name,
        "question": facet.question,
        "prefill": facet.prefill,
        "summaryCriteria": "" if facet.summaryCriteria is None else facet.summaryCriteria,
        "numeric": [] if facet.numeric is None else list(facet.numeric)
    }
    facetDir = Path(outputPath) / facet.name
    facetDir.mkdir(parents=True, exist_ok=True)
    facetHtmlDir = Path(htmlRoot) / facet.name

    def encodeConversations(filteredIndices):
        conversations = []
        for conversationI in filteredIndices:
            data = output.facetValues[conversationI]
            facetValue = data.facetValues[facetI]
            conversation = data.conversation
            conversations.append({
                "facetValue": facetValue.value,
                "allFacetValues": [{"facet": value.facet.name, "value": value.value} for value in data.facetValues],
                "conversation": dataToJson(output.data[conversationI])
            })
        return conversations
    
    def resetFlags(cluster: ConversationCluster):
        cluster.stored = False
        [clearStoredFlags(child) for child in cluster.children] if cluster.children is not None else None

    def removeFlags(cluster: ConversationCluster):
        del cluster.stored
        [removeFlags(child) for child in cluster.children] if cluster.children is not None else None
        
    
    # first pass, store all sizes recursively
    def storeSizes(cluster: ConversationCluster):
        if cluster.stored:
            return len(fileMap[cluster]) # dummy size for size of file path
        totalSize = 0
        if cluster.children is None:
            # todo: add facet values
            if cluster.filteredIndices is not None:
                totalSize += len(json.dumps(encodeConversations(cluster.filteredIndices)))
        else:
            totalSize += sum([storeSizes(x) for x in cluster.children]) if cluster.children is not None else 0
        
        totalSize += len(json.dumps({
            "summary": cluster.summary,
            "name": cluster.name,
            "numConvs": cluster.numConversations,
        }))
        cluster.totalSize = totalSize
        return totalSize
    
    def getClusterJson(cluster: ConversationCluster):
        if cluster.stored:
            return {"numConvs": cluster.numConversations, "path": fileMap[cluster]}
        resJson = {
            "summary": cluster.summary,
            "name": cluster.name,
            "numConvs": cluster.numConversations,
        }
        if cluster.children is None and cluster.filteredIndices is not None:
            resJson["conversations"] = encodeConversations(cluster.filteredIndices)
        elif cluster.children is not None:
            resJson["children"] = [getClusterJson(child) for child in cluster.children]
        return resJson
            
    curIndex = 0
    def storeIfNotTooLarge(cluster: ConversationCluster):
        if cluster.stored: raise ValueError("We already stored this cluster")
        allChildrenStored = all([child.stored for child in cluster.children]) if not cluster.children is None else True
        # store if all of our children have already been stored (no further place to expand, this is as small as we get)
        # or if size including children is less than maxSize
        if cluster.totalSize < maxSizePerFile or allChildrenStored:
            nonlocal curIndex
            jsonData = getClusterJson(cluster)
            cluster.stored = True
            outputPathForCluster = facetDir / f"data{curIndex}.json"
            htmlPathForCluster = facetHtmlDir / f"data{curIndex}.json"
            curIndex += 1
            with open(outputPathForCluster, "w") as f:
                f.write(jsonData)
            if verbose:
                print(outputPathForCluster)
            fileMap[cluster] = htmlPathForCluster
        # otherwise, continue to recurse until we can
        else:
            if not cluster.children is None:
                for child in cluster.children:
                    storeIfNotTooLarge(child)

    rootClusterHtmlPaths = []
    for rootCluster in output.rootClusters[facetI]:
        clearStoredFlags(rootCluster)
        # repeat reductions until we finally store root level
        while not rootCluster.stored:
            storeSizes(rootCluster)
            storeIfNotTooLarge(rootCluster)
        # remove the flags so they don't stick around and waste memory
        removeFlags(rootCluster)

        rootClusterHtmlPaths.append(fileMap[rootCluster])
    return {"facet": facetJson, "hierarchy": rootClusterHtmlPaths}

def storeConversationCounts(output: OpenClioResults):
    for rootClusters in output.rootClusters:
        if not rootClusters is None:
            def getAndStoreNumConversations(cluster: ConversationCluster) -> int:
                if cluster.children is None:
                    if cluster.filteredIndices is not None:
                        numConversations = len(cluster.filteredIndices)
                else:
                    numConversations = sum([getAndStoreNumConversations(child) for child in cluster.children])
                cluster.numConversations = numConversations
                return numConversations
            [getAndStoreNumConversations(cluster) for cluster in rootClusters]
                

def portOldInstances(output: OpenClioResults):
    def portCluster(cluster: ConversationCluster):
        children = [portCluster(child) for child in cluster.children] if cluster.children is not None else None
        return ConversationCluster(name=cluster.name, summary=cluster.summary, facet=None, indices=cluster.indices, parent=None, children=children)
    def portRecursively(rootClusters: List[Optional[List[ConversationCluster]]]):
        for roots in rootClusters:
            if not roots is None:
                yield [portCluster(cluster) for cluster in roots]
            else:
                yield None
    output.rootClusters = list(portRecursively(output.rootClusters))
    return output

def filterToEnglish(conversation, conversationFacetData):
    for facetValue in conversationFacetData.facetValues:
        if facetValue.facet.name == "Language":
            if facetValue.value.lower().strip() == "english": # this misses stuff like "90's english" but that's ok
                return True
    return False

# aim for 10MB or smaller files, and filter to only english ones
# clio.convertOutputToJsonChunks(cats, targetDir="chonkers/cliowildchat1", rootHtmlPath="/modelwelfare", maxSizePerFile=10000000, conversationFilter=clio.filterToEnglish)

def convertOutputToWebpage(output: OpenClioResults, rootHtmlPath: str, targetDir: str, maxSizePerFile: int, conversationFilter: Callable[[List[Dict[str, str]], ConversationFacetData], bool]=None, dataToJson: Callable[[Any], Dict[str, Any]] = None, verbose=True):
    """
    Converts the given output to a static webpage and json files, dumped to targetDir
    It's split up into multiple json files, each of max size maxSizePerFile, and streamed as needed
    Keyword arguments:
    output -- The openclio results outputs
    rootHtmlPath -- the relative path the html page will be stored on your server (something like "/clioresults")
    targetDir -- where we dump all the files
    maxSizePerFile -- the maximum size of each json data file, in bytes. I recommend 10000000
    conversationFilter -- a filter that takes (conversation, facetData) and return bool if it should be included on the webpage
    dataToJson -- Takes a data point and returns a json of the corresponding conversation. It should look like [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey :3"}, ...]. If you just want to dispaly the data as a string, just return a single entry like this: [{"role": "<whatever you want>", "content": "<your str content>"}]
    """
    if dataToJson is None:
        dataToJson = lambda conversation: [{'role': turn['role'], "content": turn['content']} for turn in conversation]

    # store these for data analysis uses
    if conversationFilter is None:
        conversationFilter = lambda conv, facetData: True
    for facetI, facet in enumerate(output.facets):
        if shouldMakeFacetClusters(facet):
            numLevels = getNumLevels(output, facetI)
            for conv in getAllClustersAtLevel(facetI, output=output, level=numLevels-1):
                conv.filteredIndices = [convIndex for convIndex in conv.indices if conversationFilter(output.data[convIndex], output.facetValues[convIndex])]

    storeConversationCounts(output)

    rootJson = []
    for facetI, facet in enumerate(output.facets):
        if verbose: print(f"facet {facet.name}")
        if shouldMakeFacetClusters(facet):
            facetJson = encodeFacetDataInChunks(facetI=facetI, output=output, maxSizePerFile=maxSizePerFile, dataToJson=dataToJson, verbose=verbose)
            rootJson.append(facetJson)
    with open(os.path.join(targetDir, "rootObjects.json"), "w") as f:
        json.dump(rootJson, f)

    pathContainingTemplate = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pathContainingTemplate, "websiteTemplate.html"), "r") as templateF:
        templateText = templateF.read().replace("ROOTOBJECTSJSON", os.path.join(rootHtmlPath, "rootObjects.json"))
        with open(os.path.join(targetDir, "index.html"), "w") as outputIndex:
            outputIndex.write(templateText)
