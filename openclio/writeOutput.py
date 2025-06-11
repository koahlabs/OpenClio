from typing import List, Dict, Tuple, Callable, Any, Optional
import os
import json
from collections import defaultdict
from pathlib import Path
import gzip
from sentence_transformers import SentenceTransformer
from .opencliotypes import Facet, FacetValue, ConversationFacetData, ConversationEmbedding, ConversationCluster, OpenClioConfig, OpenClioResults, EmbeddingArray, shouldMakeFacetClusters
from .lzutf8 import compress
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from .utils import runBatched
from numpy import typing as npt
import umap
import secrets
import numpy as np
import struct
import concave_hull
from .prompts import conversationToString

ITERATIONS = 800_000
MAGIC = b"EJ01"          # 4-byte file signature

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
def encodeFacetDataInChunks(facetI: int, output: OpenClioResults, hashMapping: List[Any], maxSizePerFile: int, dataToJson: Callable[[Any], Dict[str, Any]], targetDir: str, rootHtmlPath: str, verbose: bool, encryptKey: AESGCM = None, salt: bytes = None):
    fileMap = {}

    facet = output.facets[facetI]

    facetJson = {
        "name": facet.name,
        "question": facet.question,
        "prefill": facet.prefill,
        "summaryCriteria": "" if facet.summaryCriteria is None else facet.summaryCriteria,
        "numeric": [] if facet.numeric is None else list(facet.numeric)
    }
    facetDir = Path(targetDir) / facet.name
    facetDir.mkdir(parents=True, exist_ok=True)
    facetHtmlDir = Path(rootHtmlPath) / facet.name

    mappingFromConvIToFiles = np.zeros([len(output.data),2], np.int32)

    def encodeConversations(filteredIndices, conversationsArray: List[Any], curFileIndex: int = -1):
        conversations = []
        for conversationI in filteredIndices:
            indexInConversationsArray = len(conversationsArray)
            if curFileIndex != -1: # -1 means just probing, not writing
                # write a tuple, which file, and location in file
                mappingFromConvIToFiles[conversationI,0] = curFileIndex
                mappingFromConvIToFiles[conversationI,1] = indexInConversationsArray
            data = output.facetValues[conversationI]
            facetValue = data.facetValues[facetI]
            conversation = data.conversation
            conversationsArray.append({"conversation": dataToJson(output.data[conversationI]), "hash": hashMapping[facetI][conversationI]})
            conversations.append({
                "facetValue": facetValue.value,
                "allFacetValues": [{"facet": value.facet.name, "value": value.value} for value in data.facetValues],
                "conversation": indexInConversationsArray,
            })
        return conversations
    
    def resetStoredFlags(cluster: ConversationCluster):
        cluster.stored = False
        [resetStoredFlags(child) for child in cluster.children] if cluster.children is not None else None

    def removeStoredFlagsAndSizes(cluster: ConversationCluster):
        del cluster.stored
        del cluster.totalSize
        [removeStoredFlagsAndSizes(child) for child in cluster.children] if cluster.children is not None else None
        
    def storeConcaveHulls(cluster: ConversationCluster):
        cluster.concaveHullIndices = getClusterConcaveHullIndices(cluster)
        [storeConcaveHulls(child) for child in cluster.children] if cluster.children is not None else None

    def removeConcaveHulls(cluster: ConversationCluster):
        del cluster.concaveHullIndices
        [removeConcaveHulls(child) for child in cluster.children] if cluster.children is not None else None

    # first pass, store all sizes recursively
    def storeSizes(cluster: ConversationCluster):
        if cluster.stored:
            return len(fileMap[cluster]) # dummy size for size of file path
        totalSize = 0
        if cluster.children is None:
            # todo: add facet values
            if cluster.filteredIndices is not None:
                conversationsArray = []
                jsonValues = encodeConversations(cluster.filteredIndices, conversationsArray=conversationsArray)
                totalSize += len(json.dumps([jsonValues, conversationsArray]))
        else:
            totalSize += sum([storeSizes(x) for x in cluster.children]) if cluster.children is not None else 0
        
        totalSize += len(json.dumps({
            "summary": cluster.summary,
            "name": cluster.name,
            "numConvs": cluster.numConversations,
            "concaveHull": cluster.concaveHullIndices, # todo: store this as byte array for more concise
        }))
        cluster.totalSize = totalSize
        return totalSize
    
    def getClusterJson(cluster: ConversationCluster, conversationsArray: List[Any], curFileIndex: int = -1):
        if cluster.stored:
            return {"numConvs": cluster.numConversations, "path": fileMap[cluster]}
        resJson = {
            "summary": cluster.summary,
            "name": cluster.name,
            "numConvs": cluster.numConversations,
            "concaveHull": cluster.concaveHullIndices,
        }
        if cluster.children is None and cluster.filteredIndices is not None:
            resJson["conversations"] = encodeConversations(cluster.filteredIndices, conversationsArray=conversationsArray, curFileIndex=curFileIndex)
        elif cluster.children is not None:
            resJson["children"] = [getClusterJson(child, conversationsArray=conversationsArray, curFileIndex=curFileIndex) for child in cluster.children]
        return resJson
    
    def getAllClusterChildren(cluster: ConversationCluster, clusterIndices: List[int]):
        if cluster.children is None:
            clusterIndices.extend(cluster.filteredIndices)
        else:
            for child in cluster.children:
                getAllClusterChildren(child, clusterIndices)

    def getClusterConcaveHullIndices(cluster: ConversationCluster):
        clusterIndices = []
        getAllClusterChildren(cluster, clusterIndices)
        clusterIndices = np.array(clusterIndices)
        if len(clusterIndices) == 0:
            return []
        childrenPoints = output.umap[facetI][clusterIndices]
        convexHullIndices = concave_hull.convex_hull_indexes(childrenPoints)
        # size one or zero gives segfault
        if len(convexHullIndices) == 0:
            # convex hull failed, just do all the points
            concaveHullIndices = np.arange(childrenPoints.shape[0])
        else:
            concaveHullIndices = concave_hull.concave_hull_indexes(childrenPoints, convex_hull_indexes=convexHullIndices)
        # lookup indices in larger set
        return clusterIndices[concaveHullIndices].tolist()


    curIndex = 0
    def storeIfNotTooLarge(cluster: ConversationCluster):
        if cluster.stored: raise ValueError("We already stored this cluster")
        allChildrenStored = all([child.stored for child in cluster.children]) if not cluster.children is None else True
        # store if all of our children have already been stored (no further place to expand, this is as small as we get)
        # or if size including children is less than maxSize
        if cluster.totalSize < maxSizePerFile or allChildrenStored:
            nonlocal curIndex
            conversationsArray = []
            jsonData = getClusterJson(cluster, conversationsArray=conversationsArray, curFileIndex=curIndex)
            # add the actual conversation data separately (inside they just have references, this makes it easy to lookup single conversation from a file)
            jsonData['conversationsData'] = conversationsArray
            cluster.stored = True
            outputPathForCluster = facetDir / f"data{curIndex}.json.gz"
            htmlPathForCluster = facetHtmlDir / f"data{curIndex}.json.gz"

            curIndex += 1
            if encryptKey is None:
                with gzip.open(outputPathForCluster, "wt", encoding="utf-8") as gz:
                    json.dump(jsonData, gz, separators=(",", ":"))
            else:
                rawJson = json.dumps(jsonData, separators=(",", ":")).encode()
                gzipped = gzip.compress(rawJson, compresslevel=9)
                nonce  = secrets.token_bytes(12)
                cipher = encryptKey.encrypt(nonce, gzipped, None)
                header  = (
                    MAGIC +
                    struct.pack("<I", ITERATIONS) +  # little-endian uint32
                    salt +
                    nonce
                )
                with open(outputPathForCluster, "wb") as f:
                    f.write(header)
                    f.write(cipher)

            if verbose:
                print(outputPathForCluster)
            fileMap[cluster] = htmlPathForCluster.as_posix() # this avoids windows adding backslashes
        # otherwise, continue to recurse until we can
        else:
            if not cluster.children is None:
                for child in cluster.children:
                    storeIfNotTooLarge(child)

    rootClusterHtmlPaths = []
    for clusterI, rootCluster in enumerate(output.rootClusters[facetI]):
        resetStoredFlags(rootCluster)
        storeConcaveHulls(rootCluster)
        # repeat reductions until we finally store root level
        while not rootCluster.stored:
            storeSizes(rootCluster)
            storeIfNotTooLarge(rootCluster)
        # remove the flags so they don't stick around and waste memory
        removeStoredFlagsAndSizes(rootCluster)
        removeConcaveHulls(rootCluster)

        rootClusterHtmlPaths.append({"numConvs": rootCluster.numConversations, "path": fileMap[rootCluster]})

        print(f"Finished cluster {clusterI+1}/{len(output.rootClusters[facetI])}")

    # for the actual points, store them in a minimal byte array    
    pointsData = output.umap[facetI].astype("<f4").tobytes()
    outputPointsPath = facetDir / f"points.bin"
    htmlPointsPath = facetHtmlDir / f"points.bin"
    with open(outputPointsPath, "wb") as f:
        f.write(pointsData)

    conversationsMappingData = mappingFromConvIToFiles.astype("<i4").tobytes()
    outputConversationsMappingPath = facetDir / f"conversationsMapping.bin"
    htmlConversationsMappingPath = facetHtmlDir / f"conversationsMapping.bin"
    with open(outputConversationsMappingPath, "wb") as f:
        f.write(pointsData)

    return {"facet": facetJson, "hierarchy": rootClusterHtmlPaths, "points": htmlPointsPath.as_posix()}

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


def computeUmapHelper(embeddingArr: EmbeddingArray, verbose: bool = False):
    # unique=True is very important otherwise it gets stuck
    umapModel = umap.UMAP(n_components=2, unique=True, verbose=verbose)
    return umapModel.fit_transform(embeddingArr)

def computeUmap(data: List[Any], facetValuesEmbeddings: List[Optional[EmbeddingArray]], embeddingModel: SentenceTransformer, tokenizer, cfg: OpenClioConfig):
    cfg.print("Running umap on facet values")
    resUmaps = [(computeUmapHelper(embeddingArr, verbose=cfg.verbose) if embeddingArr is not None else None) for embeddingArr in facetValuesEmbeddings]
    cfg.print("Embedding conversations for umap")
    # fallback to default conversation to string
    conversationToStringFunc = cfg.conversationToStrFunc if cfg.conversationToStrFunc is not None else lambda conv: conversationToString(conv, tokenizer=tokenizer, maxTokens=-1)

    def processBatchFunc(batchOfTextInputs: List[str]) -> List[npt.NDArray[np.float32]]:
        embedded = embeddingModel.encode(batchOfTextInputs, show_progress_bar=False)
        return [embedded[i] for i in range(len(batchOfTextInputs))]
    
    embeddedConversations = np.stack(runBatched(data,
                    getInputs=lambda conv: conversationToStringFunc(conv),
                    processBatch=processBatchFunc,
                    processOutput=lambda conv, inputs, emb: emb,
                    batchSize=cfg.embedBatchSize,
                    verbose=cfg.verbose))
    
    cfg.print("Running umap on embedded conversations")
    conversationsUmap = computeUmapHelper(embeddingArr=embeddedConversations, verbose=cfg.verbose)
    # last index is the umap over embeddings of conversations (instead of facet values)
    resUmaps.append(conversationsUmap)
    return resUmaps

# aim for 10MB or smaller files, and filter to only english ones
# clio.convertOutputToJsonChunks(cats, targetDir="chonkers/cliowildchat1", rootHtmlPath="/modelwelfare", maxSizePerFile=10000000, conversationFilter=clio.filterToEnglish)
def convertOutputToWebpage(output: OpenClioResults, rootHtmlPath: str, targetDir: str, maxSizePerFile: int, conversationFilter: Callable[[List[Dict[str, str]], ConversationFacetData], bool]=None, dataToJson: Callable[[Any], Dict[str, Any]] = None, verbose=True, password: str=None):
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

    hashMapping = getHashMapping(output=output, conversationFilter=conversationFilter, verbose=verbose)

    # store these for data analysis uses
    if conversationFilter is None:
        conversationFilter = lambda conv, facetData: True
    for facetI, facet in enumerate(output.facets):
        if shouldMakeFacetClusters(facet):
            numLevels = getNumLevels(output, facetI)
            for conv in getAllClustersAtLevel(facetI, output=output, level=numLevels-1):
                conv.filteredIndices = [convIndex for convIndex in conv.indices if conversationFilter(output.data[convIndex], output.facetValues[convIndex])]

    storeConversationCounts(output)

    if password is None:
        encryptKey, salt = None, None
    else:
        salt       = secrets.token_bytes(16)
        encryptKey = AESGCM(
            PBKDF2HMAC(
                hashes.SHA256(), 32, salt, ITERATIONS
            ).derive(password.encode())
        )
    
    # make rootObjects.json that holds references to all the files
    rootJson = []
    for facetI, facet in enumerate(output.facets):
        if verbose: print(f"facet {facet.name}")
        if shouldMakeFacetClusters(facet):
            facetJson = encodeFacetDataInChunks(facetI=facetI, output=output, hashMapping=hashMapping, maxSizePerFile=maxSizePerFile, dataToJson=dataToJson, targetDir=targetDir, rootHtmlPath=rootHtmlPath, verbose=verbose, encryptKey=encryptKey, salt=salt)

            rootJson.append(facetJson)
    with open(os.path.join(targetDir, "rootObjects.json"), "w") as f:
        json.dump(rootJson, f)

    # write umap points (the last entry in output.umap is for umap of conversation embeddings overall, instead of umap of embeddings of individual facet values)
    pointsData = output.umap[-1].astype("<f4").tobytes()
    outputPointsPath = rootHtmlPath / f"points.bin"
    with open(outputPointsPath, "wb") as f:
        f.write(pointsData)

    # write webpage
    pathContainingTemplate = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pathContainingTemplate, "websiteTemplate.html"), "r") as templateF:
        templateText = templateF.read() \
            .replace("ROOTPOINTS", os.path.join(rootHtmlPath, "points.bin")) \
            .replace("ROOTOBJECTSJSON", os.path.join(rootHtmlPath, "rootObjects.json")) \
            .replace("ISPASSWORDPROTECTED", "true" if password is not None else "false")
        with open(os.path.join(targetDir, "index.html"), "w") as outputIndex:
            outputIndex.write(templateText)
    


def getHashMapping(output: OpenClioResults, conversationFilter: Callable[[List[Dict[str, str]], ConversationFacetData], bool]=None, verbose=True) -> List[List[str]]:
    # store these for data analysis uses
    if conversationFilter is None:
        conversationFilter = lambda conv, facetData: True
    for facetI, facet in enumerate(output.facets):
        if shouldMakeFacetClusters(facet):
            numLevels = getNumLevels(output, facetI)
            for conv in getAllClustersAtLevel(facetI, output=output, level=numLevels-1):
                conv.filteredIndices = [convIndex for convIndex in conv.indices if conversationFilter(output.data[convIndex], output.facetValues[convIndex])]

    storeConversationCounts(output)

    allFacetMappings = []
    for facetI, facet in enumerate(output.facets):
        facetMappings = [None for _ in range(len(output.data))]
        if shouldMakeFacetClusters(facet):
            curPath = f'f{facetI}'
            facetHash = curPath
            if verbose: print(facet)
            for rootClusterI, rootCluster in enumerate(sorted(output.rootClusters[facetI], key=lambda rootCluster: -rootCluster.numConversations)):
                rootClusterPath = f"{curPath}.{rootClusterI}"
                rootClusterHash = f"{facetHash},{rootClusterPath}"
                def getClusterUrlMapping(cluster: ConversationCluster, curPath: str, curHash: str):
                    # get url for this cluster
                    if cluster.children is None:
                        # $c to signal that we are displaying the list of conversations
                        # The second bit tells us which conversation
                        convHash = f"{curHash},{curPath}$c,c!{curPath}!"
                        if cluster.filteredIndices is not None:
                            for i, filteredI in enumerate(cluster.filteredIndices):
                                s = f"{convHash}{i}"
                                facetMappings[filteredI] = compress(s, "Base64")
                    else:
                        for childI, child in enumerate(sorted(cluster.children, key=lambda childKey: -childKey.numConversations)):
                            childPath = f"{curPath}.{childI}"
                            childHash = f"{curHash},{childPath}"
                            getClusterUrlMapping(cluster=child, curPath=childPath, curHash=childHash)
                getClusterUrlMapping(cluster=rootCluster, curPath=rootClusterPath, curHash=rootClusterHash)
                if verbose: print(rootClusterI)
        allFacetMappings.append(facetMappings)
    return allFacetMappings

