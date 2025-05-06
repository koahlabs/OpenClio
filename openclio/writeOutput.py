from typing import List, Dict, Tuple, Callable, Any
import os
import json
from collections import defaultdict
from pathlib import Path
from opencliotypes import Facet, FacetValue, ConversationFacetData, ConversationEmbedding, ConversationCluster, OpenClioConfig, OpenClioResults

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

def encodeClusterAsJson(facetI: int, output: OpenClioResults, cluster: ConversationCluster, currentLevel: int, highestLevelInclusive: int, fileMap: Dict[Tuple[str, str], str], dataToJson: Callable[[Any], Dict[str, Any]]):
    def encodeClusterJsonHelper(cluster: ConversationCluster, currentLevel: int, highestLevelInclusive: int):
        # at the highest level, just store references to children via their files
        if currentLevel >= highestLevelInclusive:
            children = [{"numConvs": child.numConversations, "path": fileMap[child]} for child in cluster.children if child.numConversations > 0] if cluster.children is not None else [] 
        else:
            children = [encodeClusterJsonHelper(child, currentLevel=currentLevel+1, highestLevelInclusive=highestLevelInclusive) for child in cluster.children if child.numConversations > 0] if cluster.children is not None else [] 
        res = {
            "summary": cluster.summary,
            "name": cluster.name,
            "children": children,
            "numConvs": cluster.numConversations,
        }
        if cluster.children is None and cluster.filteredIndices is not None:
            # todo: add facet values
            conversations = []
            for conversationI in cluster.filteredIndices:
                data = output.facetValues[conversationI]
                facetValue = data.facetValues[facetI]
                conversation = data.conversation
                conversations.append({
                    "facetValue": facetValue.value,
                    "allFacetValues": [{"facet": value.facet.name, "value": value.value} for value in data.facetValues],
                    "conversation": dataToJson(output.conversations[conversationI])
                })
            res["conversations"] = conversations
        return res
    return json.dumps(encodeClusterJsonHelper(cluster=cluster, currentLevel=currentLevel, highestLevelInclusive=highestLevelInclusive))

def encodeLevels(facetI: int, output: OpenClioResults, lowerLevelInclusive: int, highestLevelInclusive: int, fileMap: Dict[Tuple[str, str], str], dataToJson: Callable[[Any], Dict[str, Any]]):
    for cluster in getAllClustersAtLevel(facetI=facetI, output=output, level=lowerLevelInclusive):
        if cluster.numConversations > 0:
            yield cluster, encodeClusterAsJson(facetI=facetI, output=output, cluster=cluster, currentLevel=lowerLevelInclusive, highestLevelInclusive=highestLevelInclusive, fileMap=fileMap, dataToJson=dataToJson)

def findLowestLevelThatKeepsSizeLessThanMaxSize(facetI: int, output: OpenClioResults, curLevel: int, maxSizePerFile: int, dataToJson: Callable[[Any], Dict[str, Any]]):
    dummyFileMap = defaultdict(lambda: "aaaaaaaaaaaaaaaaaa"*5) # dummy file path
    if curLevel == 0: # we are already at top level, just use that
        return 0
    # go curLevel-1, curLevel-2, ..., 0
    for lowerLevel in range(max(0, curLevel-1), -1, -1):
        for cluster, jsonData in encodeLevels(facetI, output, lowerLevel, curLevel, fileMap=dummyFileMap, dataToJson=dataToJson):
            if len(jsonData) > maxSizePerFile:
                # we went too far, use previous one
                return min(curLevel, lowerLevel+1)
    # we can encode everything, use 0
    return 0

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

def convertOutputToWebpage(output: OpenClioResults, rootHtmlPath: str, targetDir: str, maxSizePerFile: int, conversationFilter: Callable[[List[Dict[str, str]], ConversationFacetData], bool]=None, dataToJson: Callable[[Any], Dict[str, Any]] = None):
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
                conv.filteredIndices = [convIndex for convIndex in conv.indices if conversationFilter(output.conversations[convIndex], output.facetValues[convIndex])]

    storeConversationCounts(output)

    rootJson = []
    for facetI, facet in enumerate(output.facets):
        print(f"facet {facet.name}")
        facetJson = {
            "name": facet.name,
            "question": facet.question,
            "prefill": facet.prefill,
            "summaryCriteria": "" if facet.summaryCriteria is None else facet.summaryCriteria,
            "numeric": [] if facet.numeric is None else list(facet.numeric)
        }
        if shouldMakeFacetClusters(facet):
            facetTargetDir = Path(targetDir) / facet.name
            facetTargetDir.mkdir(parents=True, exist_ok=True)

            globalInd = 0
            rootItems = []

            curLevel = numLevels-1
            fileMap = {}
            while curLevel >= 0:
                lowerLevel = findLowestLevelThatKeepsSizeLessThanMaxSize(facetI=facetI, output=output, curLevel=curLevel, maxSizePerFile=maxSizePerFile, dataToJson=dataToJson)
                print(f"Level {lowerLevel} to {curLevel}")
                facetTargetDirLevels = facetTargetDir / f"levels{lowerLevel}{curLevel}"
                facetTargetDirLevels.mkdir(parents=True, exist_ok=True)
                for cluster, jsonData in encodeLevels(facetI=facetI, output=output, lowerLevelInclusive=lowerLevel, highestLevelInclusive=curLevel, fileMap=fileMap, dataToJson=dataToJson):
                    outputPath = facetTargetDirLevels / f"data{globalInd}.json"
                    with open(outputPath, "w") as f:
                        f.write(jsonData)
                    htmlPath = os.path.join(rootHtmlPath, facet.name, f"levels{lowerLevel}{curLevel}", f"data{globalInd}.json")
                    fileMap[cluster] = htmlPath
                    if lowerLevel == 0:
                        rootItems.append({"path": htmlPath, "numConvs": cluster.numConversations})
                    globalInd += 1
                curLevel = lowerLevel - 1
            rootJson.append({"facet": facetJson, "hierarchy": rootItems})
    with open(os.path.join(targetDir, "rootObjects.json"), "w") as f:
        json.dump(rootJson, f)

    pathContainingTemplate = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pathContainingTemplate, "websiteTemplate.html"), "r") as templateF:
        templateText = templateF.read().replace("ROOTOBJECTSJSON", os.path.join(rootHtmlPath, "rootObjects.json"))
        with open(os.path.join(targetDir, "index.html"), "w") as 
            
