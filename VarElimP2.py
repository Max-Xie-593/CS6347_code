import numpy
from typing import Dict,List,Set,Tuple,Optional,Text,MutableSequence,Sequence
from math import floor,log10
from numpy import prod,dot
from copy import copy
from sys import argv, float_info
from itertools import chain
from collections import Counter
from random import uniform


class CliqueTable(object):

    class Clique(object):

        def __init__(self, nodes: List[int], domain: Dict[int,List[int]]):
            self.nodes: List[int] = nodes
            self.domain: Dict[int, List[int]] = domain

        def __str__(self):
            return str(self.nodes) + " - " + str(self.domain)

        def __len__(self) -> int:
            return len(self.nodes)

        def __bool__(self) -> bool:
            return any((self.nodes,self.domain))

    def __init__(self, nodes: List[int], domain: Dict[int,List[int]], tableVals: List[float], evid: Dict[int,int]):
        cli: CliqueTable.Clique = self.Clique(nodes,domain)
        (self.clique, newAssignments) = self._instantiateEvidence(cli,self._computeStride(cli),evid)
        self.tableVals: List[float] = self._filterValueTable(newAssignments,tableVals) if tableVals else []
        self.stride: List[int] = self._computeStride(self.clique)

    def _computeStride(self,cli:"CliqueTable.Clique") -> List[int]:
        if not cli.nodes:
            return [0]
        y = 1
        strid: List[int] = []
        for x in range(len(cli.nodes) - 1,-1,-1):
            strid.insert(0,y)
            y *= len(cli.domain[cli.nodes[x]])
        return strid

    def getIndex(self,assignment:List[int]) -> int:
        return dot(list(self.stride),assignment)

    def _getOneVarAssignment(self,strideNum:int,cardNum:int,index:int) -> int:
        return floor(index / strideNum) % cardNum

    def _getAllVarsAssignment(self,cli:"CliqueTable.Clique", stride: List[int], evid: Optional[Tuple[int,int]] = None) -> List[List[int]]:
        allAssignments: List[List[int]] = [[self._getOneVarAssignment(stride[y],len(cli.domain[cli.nodes[y]]),x) for y in range(len(stride))] for x in range(prod([len(cards) for cards in cli.domain.values()]))]
        if evid:
            evidIndex = cli.nodes.index(evid[0])
            for i, combo in enumerate(allAssignments):
                if combo[evidIndex] != evid[1]:
                    allAssignments[i] = []
        return allAssignments

    def getAllAssignments(self,evid: Optional[Tuple[int,int]] = None) -> List[List[int]]:
        return self._getAllVarsAssignment(self.clique,self.stride,evid)

    def _instantiateEvidence(self,cli:"CliqueTable.Clique", stride: List[int], evid: Dict[int,int]) -> Tuple["CliqueTable.Clique",List[List[int]]]:
        if not cli:
            return (cli,[])
        if not evid:
            return (cli,self._getAllVarsAssignment(cli,stride))
        newNodes: List[int] = list(cli.nodes)
        newDomain: Dict[int,List[int]] = dict(cli.domain)
        evidFilter: List[List[int]] = self._getAllVarsAssignment(cli,stride)
        for key, val in evid.items():
            newNodes.remove(key)
            newDomain[key] = [val]
            newAssignments: List[List[int]] = self._getAllVarsAssignment(cli,stride,(key,val))
            for i, newCombo in enumerate(newAssignments):
                if not newCombo:
                    evidFilter[i] = None
        return (CliqueTable.Clique(newNodes,newDomain),evidFilter)

    def _filterValueTable(self,assignments: List[List[int]],tableVals: List[float]) -> List[float]:
        if all(assignments):
            return tableVals
        newTableVals: List[float] = list(tableVals)
        for i, assignment in enumerate(assignments):
            if not assignment:
                newTableVals[i] = None
        return list(filter(None,newTableVals))

    def __str__(self):
        return ''.join(["Stride: ", str(self.stride), "\nClique: ",str(self.clique),"\nValues: ",str(self.tableVals),"\n"])


class MarNet(object):

    def __init__(self, clis: List[CliqueTable], nodeList: Dict[int,int]):
        self.cliques: List[CliqueTable] = clis
        self.nodeList: Dict[int,int] = nodeList
        self.networkDegree: Dict[int, int] = self._getNodeDegree(clis)

    def __str__(self):
        return '\n'.join([str(c) for c in self.cliques])

    def _computeZ(self) -> float:
        return sum([log10(x.tableVals[0]) for x in self.cliques])

    def _getNodeDegree(self, cliques: List[CliqueTable]) -> Dict[int,int]:
        numOccurenceforVars: Dict[int,int] = {}
        for c in cliques:
            for node in c.clique.nodes:
                if node not in numOccurenceforVars:
                    numOccurenceforVars[node] = 0
                numOccurenceforVars[node] += 1
        return numOccurenceforVars

    def _getMinDegree(self):
        return min(self.networkDegree,key=self.networkDegree.get)

    def _getMaxDegree(self):
        return max(self.networkDegree,key=self.networkDegree.get)

    def _getCliqueWithVar(self, commonVar: int) -> List[CliqueTable]:
        return [c for c in self.cliques if commonVar in c.clique.nodes]

    def _productCliques(self):
        cliquesCommonVar: List[CliqueTable] = self._getCliqueWithVar(self._getMinDegree())
        testSet: Set[int] = set()
        testList: List[Set[int]] = [set(x.clique.nodes) for x in cliquesCommonVar]
        for x in testList:
            testSet = testSet.union(x)
        newClique: CliqueTable.Clique = CliqueTable.Clique(sorted(list(testSet)),{n: list(range(size)) for n, size in self.nodeList.items() if n in sorted(list(testSet))})
        newCliqueTable: CliqueTable = CliqueTable(sorted(list(testSet)),{n: list(range(size)) for n, size in self.nodeList.items() if n in sorted(list(testSet))},[1.0 for x in range(prod([len(cards) for cards in newClique.domain.values()]))],{})
        for x in cliquesCommonVar:
            for assign, val in self._makenewCombosfromOld(x,newCliqueTable).items():
                newCliqueTable.tableVals[newCliqueTable.getIndex(list(assign))] *= self._threshold(val)
                self._threshold(newCliqueTable.tableVals[newCliqueTable.getIndex(list(assign))])
        for x in cliquesCommonVar:
            self.cliques.remove(x)
        self._sumOut(newCliqueTable,self._getMinDegree())

    def _makenewCombosfromOld(self,oldC: CliqueTable, newC: CliqueTable) -> Dict[Tuple[int,...],float]:
        comboHold: Dict[Tuple[int,...],float] = {}
        oldAssignments: List[List[int]] = oldC.getAllAssignments()
        newAssignments: List[List[int]] = newC.getAllAssignments()
        for old in oldAssignments:
            for new in newAssignments:
                if self._subset(oldC.clique.nodes,newC.clique.nodes,old,new):
                    comboHold[tuple(new)] = oldC.tableVals[oldC.getIndex(old)]
        return comboHold

    def _subset(self,oldNodes: List[int], newNodes: List[int], oldassignment: List[int], newassignment: List[int]) -> bool:
        if not(set(oldNodes) <= set(newNodes)):
            return False
        for a, var in enumerate(oldNodes):
            checkIndex = newNodes.index(var)
            if oldassignment[a] != newassignment[checkIndex]:
                return False
        return True

    def _sumOut(self,cli: CliqueTable,var: int):
        newVarList: List[int] = copy(cli.clique.nodes)
        newVarList.remove(var)
        if not newVarList:
            self.cliques.append(CliqueTable(newVarList,{},[sum(cli.tableVals)],{}))
        else:
            newClique: CliqueTable.Clique = CliqueTable.Clique(sorted(list(newVarList)),{n: list(range(size)) for n, size in self.nodeList.items() if n in sorted(list(newVarList))})
            newCliqueTable: CliqueTable = CliqueTable(sorted(list(newVarList)),{n: list(range(size)) for n, size in self.nodeList.items() if n in sorted(list(newVarList))},[0.0 for x in range(prod([len(cards) for cards in newClique.domain.values()]))],{})
            oldAssignments: List[List[int]] = cli.getAllAssignments()
            newAssignments: List[List[int]] = newCliqueTable.getAllAssignments()
            for new in newAssignments:
                for old in oldAssignments:
                    if self._subset(newVarList,cli.clique.nodes,new,old):
                        newCliqueTable.tableVals[newCliqueTable.getIndex(new)] += self._threshold(cli.tableVals[cli.getIndex(old)])
                        self._threshold(newCliqueTable.tableVals[newCliqueTable.getIndex(new)])
            self.cliques.append(newCliqueTable)
        self.networkDegree = self._getNodeDegree(self.cliques)

    def _threshold(self, x: float) -> float:
        if x < 1e-10:
            return 0.0
        elif x == float('inf'):
            return float_info.max
        return x

    def bucketElim(self) -> float:
        while any(self.networkDegree):
            self._productCliques()
        return self._computeZ()


class SampleMarNet(MarNet):

    def __init__(self, clis: List[CliqueTable], nodeList: Dict[int,int]):
        super().__init__(clis, nodeList)

    def _getMinOrdering(self) -> Sequence[int]:
        scopes: List[List[int]] = [x.clique.nodes for x in self.cliques]
        nodeNeighbors: Dict[int,Set[int]] = {node: set(chain(*(x for x in scopes if node in x))) - {node} for node in set(chain(*(y for y in scopes)))}
        ordering: MutableSequence[int] = []
        while any(nodeNeighbors):
            minNode = min(nodeNeighbors,key=lambda x: len(nodeNeighbors[x]))
            ordering.append(minNode)
            for node,neighbors in nodeNeighbors.items():
                if minNode in neighbors:
                    nodeNeighbors[node] = (neighbors | nodeNeighbors[minNode]) - {minNode,node}
            nodeNeighbors.pop(minNode)
        return ordering

    def _getTreeDecomp(self) -> List[Set[int]]:
        scopes: List[List[int]] = [x.clique.nodes for x in self.cliques]
        treeDecompost: List[Set[int]] = []
        for o in self._getMinOrdering():
            tn: List[List[int]] = [x for x in scopes if o in x]
            cluster: Set[int] = set(sorted(list(set([x for y in tn for x in y]))))
            treeDecompost.append(cluster)
            scopes = [c for c in scopes if o not in c]
            scopes.append(list(cluster - {o}))
        return treeDecompost

    def _getwCutSet(self, width: int) -> Set[int]:
        cutSet: Set[int] = set()
        treeDecomp: List[Set[int]] = self._getTreeDecomp()
        while max([len(x) for x in treeDecomp]) > width + 1:
            counts: Dict[int,int] = Counter(list(chain(*treeDecomp)))
            maxVar: int = max(counts,key=counts.get)
            cutSet.add(maxVar)
            treeDecomp = [set([x for x in y if x != maxVar]) for y in treeDecomp]
        return cutSet

    def _getUniformRandomAssignment(self, cutSet: Set[int]) -> Dict[int,int]:
        return {node: int(uniform(0,self.nodeList.get(node))) for node in cutSet}

    def _getAdaptiveAssignment(self, sampleS: List[float], sampleZ: List[float], sampleE: List[Dict[int,int]]) -> List[float]:
        n: int = len(sampleS)
        w: List[float] = [sampleZ[x] + sampleS[x] for x in range(n)]
        totalW: float = sum(w)
        return [sum((int(y == sampleE[i]) * w[i]) / totalW for i in range(n)) for y in sampleE]

    def sampleElim(self,runs: int, width: int, adapt=False) -> float:
        sampleEvidences: List[Dict[int,int]] = []
        sampleZVals: List[float] = []
        sampleSpaces: List[float] = []
        nodeListCopy: Dict[int,int] = copy(self.nodeList)
        cliquesCopy: List[CliqueTable] = copy(self.cliques)
        cutset: Set[int] = self._getwCutSet(width)
        for x in range(runs):
            print(x)
            randomEvid: Dict[int,int] = self._getUniformRandomAssignment(cutset)
            sampleEvidences.append(randomEvid)
            newCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in randomEvid.items() if node in x.clique.nodes}) for x in cliquesCopy]
            testNet: MarNet = MarNet(newCliqueTable,nodeListCopy)
            zVal: float = testNet.bucketElim()
            sampleZVals.append(zVal)
            sampleSize: float = prod([float(y) for x,y in nodeListCopy.items() if x in randomEvid])
            sampleSpaces.append(log10(sampleSize))
            if (x + 1) % 100 == 0 and (x - 1) > 1 and adapt:
                sampleSpaces = self._getAdaptiveAssignment(sampleSpaces,sampleZVals,sampleEvidences)
        return sum([sampleZVals[x] + sampleSpaces[x] for x in range(runs)]) / runs


def readFiles(networkFile: Text, evidFile: Text) -> Tuple[List[CliqueTable],Dict[int,int]]:
    with open(networkFile) as f:
        netLines: List[Text] = [line.strip() for line in f.readlines() if line.strip()]
    with open(evidFile) as g:
        evidLine: List[int] = [int(a) for a in g.readline().strip().split()]
    nodes: Dict[int,int] = {nodeName: int(size) for nodeName, size in enumerate(netLines[2].split())}
    numCliques: int = int(netLines[3])
    cliqueEnd: int = 4 + numCliques
    allCliques: List[List[int]] = [[int(i) for i in line.split()[1:]] for line in netLines[4:cliqueEnd]]
    cliqueVals: List[List[float]] = [[float(i) for i in netLines[x + 1].split()] for x in range(cliqueEnd,len(netLines),2)]
    evidDict: Dict[int,int] = {evidLine[i]:evidLine[i + 1] for i in range(1,len(evidLine),2)} if len(evidLine) != 1 else {}
    graphModel: List[CliqueTable] = [CliqueTable(cli,{n: list(range(size)) for n, size in nodes.items() if n in cli},cliqueVals[i],{evidNode: evidVal for evidNode, evidVal in evidDict.items() if evidNode in cli}) for i, cli in enumerate(allCliques)]
    return (graphModel,nodes)


def readPRFile(prFile: Text) -> float:
    with open(prFile) as f:
        lines: List[Text] = [line.strip() for line in f.readlines() if line.strip()]
    return float(lines[1])


def main():
    (net, nodeList) = readFiles(argv[1],argv[2])
    # exactVal: float = readPRFile(argv[3])
    adapt: bool = False
    if len(argv) > 3:
        adapt = (argv[3] == 'adapt')
    numpy.seterr(all='ignore')
    network: SampleMarNet = SampleMarNet(net,nodeList)
    estimatevalue: float = network.sampleElim(100,3,adapt)
    print(estimatevalue)
    # print((exactVal - estimatevalue) / exactVal)
    # print("%.4f" % network.bucketElim())


if __name__ == "__main__":
    main()