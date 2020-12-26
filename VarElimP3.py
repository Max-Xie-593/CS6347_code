from typing import Dict,List,Tuple,Optional,Text,Sequence,Set,MutableSequence
from math import floor,log10
from numpy import prod,dot,abs
from copy import copy,deepcopy
from sys import argv
from itertools import chain
from random import uniform,sample,shuffle

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
        allAssignments: List[List[int]] = [
            [
                self._getOneVarAssignment(
                    stride[y],
                    len(cli.domain[cli.nodes[y]]),
                    x
                )
                for y in range(len(stride))
            ]
            for x in range(
                prod(
                    [
                        len(cards)
                        for cards in cli.domain.values()
                    ]
                )
            )
        ]
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
        return ''.join(
            [
                "Stride: ",
                str(self.stride),
                "\nClique: ",
                str(self.clique),
                "\nValues: ",
                str(self.tableVals),
                "\n"
            ]
        )


class BayesNet(object):

    def __init__(self, clis: List[CliqueTable], nodeList: Dict[int,int]):
        self.cliques: List[CliqueTable] = clis
        self.nodeList: Dict[int,int] = nodeList

    def __str__(self):
        return '\n'.join([str(c) for c in self.cliques])

    def computeLikelihood(self) -> float:
        return sum([log10(x.tableVals[0]) for x in self.cliques])

    def learnParams(self,trainingdata: List[Dict[int,int]]):
        for x in self.cliques:
            nodes: List[int] = x.clique.nodes
            child: int = nodes[-1]
            newVals: List[float] = [0.0 for y in range(len(x.tableVals))]
            for assignment in x.getAllAssignments():
                assignmentsforClique: List[Dict[int,int]] = [{var: assignment for var,assignment in x.items() if var in nodes} for x in trainingdata]
                assignmentswithIndex: Dict[int,int] = {nodes:assignment[i] for i,nodes in enumerate(nodes)}
                numerator: int = 1
                for dataline in assignmentsforClique:
                    numerator += (assignmentswithIndex == dataline)
                denominator: int = 2
                del(assignmentswithIndex[child])
                for dataline in assignmentsforClique:
                    del(dataline[child])
                    denominator += (assignmentswithIndex == dataline)
                newVals[x.getIndex(assignment)] = round(float(numerator / denominator),6)
            x.tableVals = newVals


class Learner(object):

    def __init__(self, net: BayesNet, training: List[Dict[int,int]], missing: List[Dict[int,int]], test: List[Dict[int,int]]):
        self.network: BayesNet = net
        self.learnedNetwork: BayesNet = deepcopy(net)
        self.trainingData: List[Dict[int,int]] = training
        self.missingData: List[Dict[int,int]] = missing
        self.testData: List[Dict[int,int]] = test

    def testFOD(self) -> float:
        nodeListCopy: Dict[int,int] = copy(self.network.nodeList)
        self.learnedNetwork.learnParams(self.trainingData)
        originalLikelihoods: List[float] = [0.0 for i in range(len(self.testData))]
        learnedLikelihoods: List[float] = [0.0 for i in range(len(self.testData))]

        for i,testAssignment in enumerate(self.testData):
            print(i)
            originalCliquesCopy: List[CliqueTable] = deepcopy(self.network.cliques)
            learnedCliquesCopy: List[CliqueTable] = deepcopy(self.learnedNetwork.cliques)
            newOriginalCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in testAssignment.items() if node in x.clique.nodes}) for x in originalCliquesCopy]
            newLearnedCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in testAssignment.items() if node in x.clique.nodes}) for x in learnedCliquesCopy]
            newOriginalBayesNet: BayesNet = BayesNet(newOriginalCliqueTable,nodeListCopy)
            newLearnedBayesNet: BayesNet = BayesNet(newLearnedCliqueTable,nodeListCopy)
            originalLikelihoods[i] = newOriginalBayesNet.computeLikelihood()
            learnedLikelihoods[i] = newLearnedBayesNet.computeLikelihood()
        return sum(abs([originalLikelihoods[i] - learnedLikelihoods[i] for i in range(len(originalLikelihoods))]))

    def testPOD(self) -> float:
        nodeListCopy: Dict[int,int] = copy(self.network.nodeList)
        originalLikelihoods: List[float] = [0.0 for i in range(len(self.testData))]
        learnedLikelihoods: List[float] = [0.0 for i in range(len(self.testData))]

        for i,testAssignment in enumerate(self.testData):
            print(i)
            originalCliquesCopy: List[CliqueTable] = deepcopy(self.network.cliques)
            learnedCliquesCopy: List[CliqueTable] = deepcopy(self.learnedNetwork.cliques)
            newOriginalCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in testAssignment.items() if node in x.clique.nodes}) for x in originalCliquesCopy]
            newLearnedCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in testAssignment.items() if node in x.clique.nodes}) for x in learnedCliquesCopy]
            newOriginalBayesNet: BayesNet = BayesNet(newOriginalCliqueTable,nodeListCopy)
            newLearnedBayesNet: BayesNet = BayesNet(newLearnedCliqueTable,nodeListCopy)
            originalLikelihoods[i] = newOriginalBayesNet.computeLikelihood()
            learnedLikelihoods[i] = newLearnedBayesNet.computeLikelihood()
        return sum(abs([originalLikelihoods[i] - learnedLikelihoods[i] for i in range(len(originalLikelihoods))]))

    def _completeMissingData(self):
        newTrainingData: List[Dict[int,int]] = []
        for i, missingassignments in enumerate(self.missingData):
            unsignedNodes: List[int] = list(missingassignments.keys())
            unsignedNodesDomains: Dict[int,List[int]] = {x: [0,1] for x in unsignedNodes}
            # print(unsignedNodesDomains)
            holdClique: CliqueTable = CliqueTable(unsignedNodes,unsignedNodesDomains,[],{})
            # print(holdClique)
            for x in holdClique.getAllAssignments():
                newAssignment: Dict[int,int] = {nodes:x[i] for i,nodes in enumerate(unsignedNodes)}
                originaltraindata: Dict[int,int] = deepcopy(self.trainingData[i])
                newTrainingData.append({**originaltraindata,**newAssignment})
        self.trainingData = newTrainingData
        # print(len(self.trainingData))
        # print(len(self.missingData))

    def _computeLikelihood(self) -> List[float]:
        nodeListCopy: Dict[int,int] = copy(self.network.nodeList)
        originalLikelihoods: List[float] = [0.0 for i in range(len(self.trainingData))]
        for i,y in enumerate(self.trainingData):
            originalCliquesCopy: List[CliqueTable] = deepcopy(self.network.cliques)
            newOriginalCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in y.items() if node in x.clique.nodes}) for x in originalCliquesCopy]
            newOriginalBayesNet: BayesNet = BayesNet(newOriginalCliqueTable,nodeListCopy)
            originalLikelihoods[i] = newOriginalBayesNet.computeLikelihood()
        return originalLikelihoods

    def _computeLikelihoodT(self) -> List[float]:
        nodeListCopy: Dict[int,int] = copy(self.network.nodeList)
        originalLikelihoods: List[float] = [0.0 for i in range(len(self.testData))]
        for i,y in enumerate(self.testData):
            originalCliquesCopy: List[CliqueTable] = deepcopy(self.network.cliques)
            newOriginalCliqueTable: List[CliqueTable] = [CliqueTable(x.clique.nodes,x.clique.domain,x.tableVals,{node: nodeVal for node,nodeVal in y.items() if node in x.clique.nodes}) for x in originalCliquesCopy]
            newOriginalBayesNet: BayesNet = BayesNet(newOriginalCliqueTable,nodeListCopy)
            originalLikelihoods[i] = newOriginalBayesNet.computeLikelihood()
        return originalLikelihoods

    def _computeEss(self) -> Tuple[List[List[float]],float]:
        n = len(self.trainingData)
        count: List[int] = [1 for l in range(n)]
        tableLengths: List[List[float]] = [[0.0 for y in x.tableVals] for x in self.network.cliques]
        likelihoods: List[float] = self._computeLikelihood()
        total: float = 0.0
        total_count: int = 0
        for i in range(n):
            total += likelihoods[i]
            total_count += 1
            for m in range(len(tableLengths)):
                assignment: Dict[int,int] = {node:val for node, val in self.trainingData[i].items() if node in self.network.cliques[m].clique.nodes}
                orderedAssignment: List[int] = [assignment[x] for x in self.network.cliques[m].clique.nodes]
                index: int = self.network.cliques[m].getIndex(orderedAssignment)
                tableLengths[m][index] += (count[i] * likelihoods[i])
        return (tableLengths, total)

    def _normalizenetwork(self, essVals: List[List[float]]) -> List[List[float]]:
        for i,f in enumerate(self.network.cliques):
            childDomainSize: int = len(f.clique.domain[f.clique.nodes[-1]])
            for j in range(len(f.tableVals) // 2):
                total_sum: float = sum([essVals[i][childDomainSize * j + k] for k in range(childDomainSize)])
                if total_sum == 0.0:
                    for k in range(childDomainSize):
                        essVals[i][childDomainSize * j + k] = f.tableVals[childDomainSize * j + k]
                else:
                    for k in range(childDomainSize):
                        essVals[i][childDomainSize * j + k] = essVals[i][childDomainSize * j + k] / total_sum + log10(1e-100)
                    total_sum = sum([essVals[i][childDomainSize * j + k] for k in range(childDomainSize)])
                    for k in range(childDomainSize):
                        essVals[i][childDomainSize * j + k] = (essVals[i][childDomainSize * j + k] / total_sum)
        return essVals

    def EMPOD(self) -> float:
        startVals: List[List[float]] = [[uniform(0,1) for y in x.tableVals] for x in self.network.cliques]
        for i,x in enumerate(self.learnedNetwork.cliques):
            x.tableVals = startVals[i]
        self._completeMissingData()
        for t in range(20):
            print(t)
            (essVals, _) = self._computeEss()
            newEssVals: List[List[float]] = self._normalizenetwork(essVals)
            for i,x in enumerate(self.learnedNetwork.cliques):
                x.tableVals = newEssVals[i]
        return self.testPOD()
            

class BayesMixLearner(Learner):

    def __init__(self, net, training, missing, test, p: int):
        super().__init__(net, training, missing, test)
        self.k_Networks: List[BayesNet] = self._generate_k_dags(p)
        self.k_Learners: List[Learner] = [Learner(x,training,missing,test) for x in self.k_Networks]
        self.p_vals: List[float] = self._normalizeP(p)
        self.k: int = p

    def testMBayes(self) -> float:
        self._learn_params()
        hold: List[List[float]] = []
        for a in self.k_Learners:
            hold.append(a._computeLikelihoodT())
        weightedHold: List[float] = [0.0 for x in range(len(hold[0]))]
        for i,x in enumerate(hold):
            for j,y in enumerate(x):
                weightedHold[j] += (y * self.p_vals[i])
        originalnetVals: List[float] = self._computeLikelihoodT()
        return sum(abs([originalnetVals[i] - weightedHold[i] for i in range(len(originalnetVals))]))

    def _normalizeP(self, k: int) -> List[float]:
        p: List[float] = [uniform(0,1) for x in range(k)]
        total_p: float = sum(p)
        p = [(x / total_p) + log10(1e-100) for x in p]
        total_p = sum(p)
        return [(x / total_p) for x in p]

    def _difNormalizeP(self, p: List[float]) -> List[float]:
        total_p: float = sum(p)
        p = [(x / total_p) + log10(1e-100) for x in p]
        total_p = sum(p)
        return [(x / total_p) for x in p]

    def _generate_k_dags(self, k: int) -> List[BayesNet]:
        nodeListCopy: Dict[int,int] = copy(self.network.nodeList)
        k_random_networks: List[BayesNet] = [BayesNet([],nodeListCopy) for _ in range(k)]

        for i in range(k):
            nodes: List[int] = list(nodeListCopy.keys())
            used_nodes: List[int] = []
            shuffle(nodes)
            top: int = nodes.pop()
            newCliques: List[CliqueTable] = [CliqueTable([top],{top:list(range(nodeListCopy[top]))},[0.0 for x in range(nodeListCopy[top])],{})]
            used_nodes.append(top)
            while len(nodes) != 0:
                node: int = nodes.pop()
                num_parents: int = int(uniform(0,4)) % len(used_nodes)
                parents: List[int] = sample(used_nodes,num_parents)
                newcliqueNodes: List[int] = parents + [node]
                newdomain: Dict[int,List[int]] = {x:list(range(nodeListCopy[x])) for x in newcliqueNodes} 
                newCliques.append(CliqueTable(newcliqueNodes,newdomain,[0.0 for x in range(prod([len(y) for y in newdomain.values()]))],{}))
                used_nodes.append(node)
            k_random_networks[i].cliques = newCliques
        return k_random_networks

    def _learn_params(self):
        for z in self.k_Learners:
            tableLengths: List[List[float]] = [[uniform(0,1) for y in x.tableVals] for x in z.network.cliques]
            for i,x in enumerate(z.network.cliques):
                x.tableVals = tableLengths[i]
        for t in range(20):
            print(t)
            k_likelihoods: List[float] = []
            for x in self.k_Learners:
                (essVals, likelihood) = x._computeEss()
                newEssVals: List[List[float]] = x._normalizenetwork(essVals)
                k_likelihoods.append(likelihood)
                for i,z in enumerate(x.network.cliques):
                    z.tableVals = newEssVals[i]
            self.p_vals = self._difNormalizeP([k_likelihoods[i] for i in range(self.k)])


def readFiles(networkFile: Text, trainFile: Text, testfile: Text) -> Tuple[List[CliqueTable],Dict[int,int],List[Dict[int,int]],List[Dict[int,Text]],List[Dict[int,int]]]:

    def readTestData(testData: List[Text]) -> List[Dict[int,int]]:
        return [
            {
                nodeName: int(assignment)
                for nodeName, assignment in enumerate(i.split())
                if assignment != '?'
            }
            for i in testData[1:]
        ]

    def readTrainingData(pod: List[Text]) -> Tuple[List[Dict[int,int]],List[Dict[int,Text]]]:
        return (
            [
                {
                    nodeName: int(assignment)
                    for nodeName, assignment in enumerate(i.split())
                    if assignment != '?'
                }
                for i in pod[1:]
            ],
            [
                {
                    nodeName: assignment
                    for nodeName, assignment in enumerate(i.split())
                    if assignment == '?'
                }
                for i in pod[1:]
            ]
        )

    def grabValsforAllCliques(cliqueValues: List[Text], numberOfCliques: int) -> List[List[float]]:
        index: int = 0
        allCliqueVals: List[List[float]] = []
        while len(allCliqueVals) < numberOfCliques:
            numVals: int = int(cliqueValues[index])
            cliqueVals: List[List[float]] = []
            numValsHold: int = len(cliqueVals)
            index += 1
            while numValsHold < numVals:
                vals: List[float] = [float(i) for i in cliqueValues[index].split()]
                cliqueVals.append(vals)
                numValsHold += len(vals)
                index += 1
            allCliqueVals.append(list(chain.from_iterable(cliqueVals)))
        return allCliqueVals

    with open(networkFile) as f:
        netLines: List[Text] = [line.replace('\t',' ').strip() for line in f.readlines() if line.strip()]
    with open(trainFile) as f:
        trainlines: List[Text] = [line.strip() for line in f.readlines() if line.strip()]
    with open(testfile) as f:
        testlines: List[Text] = [line.strip() for line in f.readlines() if line.strip()]
    nodes: Dict[int,int] = {nodeName: int(size) for nodeName, size in enumerate(netLines[2].split())}
    numCliques: int = int(netLines[3])
    cliqueEnd: int = 4 + numCliques
    allCliques: List[List[int]] = [[int(i) for i in line.split()[1:]] for line in netLines[4:cliqueEnd]]
    cliqueVals: List[List[float]] = grabValsforAllCliques(netLines[cliqueEnd:],numCliques)
    graphModel: List[CliqueTable] = [
        CliqueTable(
            cli,
            {
                n: list(range(size))
                for n, size in nodes.items()
                if n in cli
            },
            cliqueVals[i],
            {}
        )
        for i, cli in enumerate(allCliques)
    ]
    (trainingAssignments, missingAssignments) = readTrainingData(trainlines)
    testAssignments: List[Dict[int,int]] = readTestData(testlines)
    return (graphModel,nodes,trainingAssignments,missingAssignments,testAssignments)


def main():
    (net, nodeList, trainingdata, missingdata, testdata) = readFiles(argv[1],argv[2],argv[3])
    network: BayesNet = BayesNet(net,nodeList)
    if argv[4] == '1':
        learner: Learner = Learner(network,trainingdata,missingdata,testdata)
        print(learner.testFOD())
    elif argv[4] == '2':
        learner: Learner = Learner(network,trainingdata,missingdata,testdata)
        print(learner.EMPOD())
    elif argv[4] == '3':
        mixLearner: BayesMixLearner = BayesMixLearner(network,trainingdata,missingdata,testdata,2)
        print(mixLearner.testMBayes())


if __name__ == "__main__":
    main()