/// \ingroup base
/// \class MergeTreeBase
/// \author Mathieu Pont (mathieu.pont@lip6.fr)
/// \date 2021.
///
/// \b Related \b publication \n
/// "Wasserstein Distances, Geodesics and Barycenters of Merge Trees" \n
/// Mathieu Pont, Jules Vidal, Julie Delon, Julien Tierny.\n
/// Proc. of IEEE VIS 2021.\n
/// IEEE Transactions on Visualization and Computer Graphics, 2021

#pragma once

#include <AssignmentSolver.h>
#include <FTMNode.h>
#include <FTMTree.h>
#include <FTMTreePPUtils.h>
#include <FTMTreeUtils.h>

#include "MergeTreeUtils.h"

#include <optional>
#include <functional>

namespace ttk {

  class MergeTreeBase : virtual public Debug {
  protected:
    //MA_mditz
    bool MA_mditz = false;
    bool acceleration_ = false;
    bool parallelFor = false;
    bool statsTest = false;
    bool useThresholdCBD_ = true;
    bool globalThreshold_ = true;
    double thresholdOfCBD_ = 5.;
    bool cbdDebug = false;
    bool shortTreeStats = true;
    bool postprocess_ = true;
    bool sortForestSolverInput = false;

    int assignmentSolverID_ = 0;
    bool epsilon1UseFarthestSaddle_ = false;
    double epsilonTree1_ = 0;
    double epsilonTree2_ = 0;
    double epsilon2Tree1_ = 100;
    double epsilon2Tree2_ = 100;
    double epsilon3Tree1_ = 100;
    double epsilon3Tree2_ = 100;
    double persistenceThreshold_ = 0;
    bool barycenterMergeTree_ = false;
    bool useMinMaxPair_ = true;
    bool deleteMultiPersPairs_ = false;

    bool branchDecomposition_ = true;
    int wassersteinPower_ = 2;
    bool normalizedWasserstein_ = true;
    bool keepSubtree_ = false;
    double nonMatchingWeight_ = 1.0;

    bool distanceSquaredRoot_ = true; // squared root
    bool useFullMerge_ = false;

    bool isPersistenceDiagram_ = false;
    bool convertToDiagram_ = false;

    // Double input
    double mixtureCoefficient_ = 0.5;
    bool useDoubleInput_ = false;

    // Old
    bool parallelize_ = true;
    int nodePerTask_ = 32;
    bool cleanTree_ = true;


    // Clean correspondence
    std::vector<std::vector<int>> treesNodeCorr_;

  public:
    MergeTreeBase() {
      this->setDebugMsgPrefix(
        "MergeTreeBase"); // inherited from Debug: prefix will be printed
                          // at the beginning of every msg
    }

    void setPostprocess(bool postproc) {
      postprocess_ = postproc;
    }

    //MA_mditz
    void setMA_mditz(bool b){
      MA_mditz = b;
    }

    //MA_mditz
    void setAcceleration(bool b) {
      acceleration_ = b;
    }

    //MA_mditz
    void setStatsTest(bool b){
      statsTest = b;
    }

    //MA_mditz
    void setParallelFor(bool b){
      parallelFor = b;
    }

    //MA_mditz
    void setUseThresholdCBD(bool b){
      useThresholdCBD_ = b;
    }

    //MA_mditz
    void setGlobalThreshold(bool b){
      globalThreshold_ = b;
    }

    //MA_mditz
    void setThresholdOfCBD(double b){
      thresholdOfCBD_ = b;
    }

    void setAssignmentSolver(int assignmentSolver) {
      assignmentSolverID_ = assignmentSolver;
    }

    void setEpsilon1UseFarthestSaddle(bool b) {
      epsilon1UseFarthestSaddle_ = b;
    }

    void setEpsilonTree1(double epsilon) {
      epsilonTree1_ = epsilon;
    }

    void setEpsilonTree2(double epsilon) {
      epsilonTree2_ = epsilon;
    }

    void setEpsilon2Tree1(double epsilon) {
      epsilon2Tree1_ = epsilon;
    }

    void setEpsilon2Tree2(double epsilon) {
      epsilon2Tree2_ = epsilon;
    }

    void setEpsilon3Tree1(double epsilon) {
      epsilon3Tree1_ = epsilon;
    }

    void setEpsilon3Tree2(double epsilon) {
      epsilon3Tree2_ = epsilon;
    }

    void setPersistenceThreshold(double pt) {
      persistenceThreshold_ = pt;
    }

    void setParallelize(bool para) {
      parallelize_ = para;
    }

    void setNodePerTask(int npt) {
      nodePerTask_ = npt;
    }

    void setBranchDecomposition(bool useBD) {
      branchDecomposition_ = useBD;
    }

    void setNormalizedWasserstein(bool normalizedWasserstein) {
      normalizedWasserstein_ = normalizedWasserstein;
    }

    void setKeepSubtree(bool keepSubtree) {
      keepSubtree_ = keepSubtree;
    }

    void setNonMatchingWeight(double weight) {
      nonMatchingWeight_ = weight;
    }

    void setBarycenterMergeTree(bool imt) {
      barycenterMergeTree_ = imt;
    }

    void setDistanceSquaredRoot(bool distanceSquaredRoot) {
      distanceSquaredRoot_ = distanceSquaredRoot;
    }

    void setUseMinMaxPair(bool useMinMaxPair) {
      useMinMaxPair_ = useMinMaxPair;
    }

    void setDeleteMultiPersPairs(bool deleteMultiPersPairsT) {
      deleteMultiPersPairs_ = deleteMultiPersPairsT;
    }

    void setCleanTree(bool clean) {
      cleanTree_ = clean;
    }

    void setIsPersistenceDiagram(bool isPD) {
      isPersistenceDiagram_ = isPD;
    }

    std::vector<std::vector<int>> getTreesNodeCorr() {
      return treesNodeCorr_;
    }

    // ------------------------------------------------------------------------
    // Double Input
    // ------------------------------------------------------------------------
    double mixDistancesMinMaxPairWeight(bool isFirstInput) {
      return (
        mixtureCoefficient_ == 0.0 or mixtureCoefficient_ == 1.0
          ? (isFirstInput ? mixtureCoefficient_ : (1.0 - mixtureCoefficient_))
          : (isFirstInput ? 1.0 / std::pow(mixDistancesWeight(isFirstInput), 2)
                          : 0.0));
    }

    double mixDistancesWeight(bool isFirstInput) {
      return (isFirstInput ? std::min(mixtureCoefficient_ * 2, 1.0)
                           : std::min(-mixtureCoefficient_ * 2 + 2, 1.0));
    }

    template <class dataType>
    double mixDistances(dataType distance1, dataType distance2) {
      return mixDistancesWeight(true) * distance1
             + mixDistancesWeight(false) * distance2;
    }

    template <class dataType>
    void
      mixDistancesMatrix(std::vector<std::vector<dataType>> &distanceMatrix,
                         std::vector<std::vector<dataType>> &distanceMatrix2) {
      for(unsigned int i = 0; i < distanceMatrix.size(); ++i)
        for(unsigned int j = 0; j < distanceMatrix[i].size(); ++j)
          distanceMatrix[i][j] = mixDistances<dataType>(
            distanceMatrix[i][j], distanceMatrix2[i][j]);
    }

    // ------------------------------------------------------------------------
    // Tree Preprocessing
    // ------------------------------------------------------------------------
    // Epsilon 1 processing
    template <class dataType>
    void mergeSaddle(ftm::FTMTree_MT *tree,
                     double epsilon,
                     std::vector<std::vector<ftm::idNode>> &treeNodeMerged,
                     bool mergeByPersistence = false) {
      bool fullMerge = (epsilon == 100);
      fullMerge &= useFullMerge_;

      treeNodeMerged.clear();
      treeNodeMerged.resize(tree->getNumberOfNodes());

      if(mergeByPersistence)
        ftm::computePersistencePairs<dataType>(
          tree); // need to have the pairing (if merge by persistence)

      // Compute epsilon value
      dataType maxValue = tree->getValue<dataType>(0);
      dataType minValue = tree->getValue<dataType>(0);
      for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i) {
        if(!tree->isRoot(i) and !tree->isLeaf(i)) {
          dataType iValue = tree->getValue<dataType>(i);
          if(mergeByPersistence) {
            maxValue = (maxValue < iValue) ? iValue : maxValue;
            minValue = (minValue > iValue) ? iValue : minValue;
          } else {
            ftm::idNode const parent = tree->getParentSafe(i);
            dataType parentValue = tree->getValue<dataType>(parent);
            dataType tempMax = std::max(iValue, parentValue);
            dataType tempMin = std::min(iValue, parentValue);
            if((tempMax - tempMin) > (maxValue - minValue)) {
              maxValue = tempMax;
              minValue = tempMin;
            }
          }
        }
      }
      double const epsilonOri = epsilon;
      epsilon = (maxValue - minValue) * epsilon / 100;

      // For Farthest Saddle option
      if(epsilon1UseFarthestSaddle_)
        epsilon = tree->getMaximumPersistence<dataType>() * epsilonOri / 100;
      bool isJT = tree->isJoinTree<dataType>();
      auto isFarthest = [&](ftm::idNode a, ftm::idNode b) {
        return (isJT
                and tree->getValue<dataType>(a) > tree->getValue<dataType>(b))
               or (not isJT
                   and tree->getValue<dataType>(a)
                         < tree->getValue<dataType>(b));
      };
      std::vector<ftm::idNode> farthestSaddle(tree->getNumberOfNodes());
      for(unsigned int i = 0; i < farthestSaddle.size(); ++i)
        farthestSaddle[i] = i;

      // --- Merge saddle
      // Create stack
      std::stack<int> nodeStack;
      std::queue<ftm::idNode> queue;
      queue.emplace(tree->getRoot());
      while(!queue.empty()) {
        ftm::idNode const node = queue.front();
        queue.pop();
        nodeStack.emplace(node);
        std::vector<ftm::idNode> children;
        tree->getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }
      // Iterate through nodes
      while(!nodeStack.empty()) {
        ftm::idNode const nodeId = nodeStack.top();
        nodeStack.pop();
        if(!tree->isRoot(nodeId) and !tree->isLeaf(nodeId)) {
          ftm::idNode const parentNodeId = tree->getParentSafe(nodeId);
          dataType nodeValue = tree->getValue<dataType>(nodeId);
          if(epsilon1UseFarthestSaddle_)
            nodeValue = tree->getValue<dataType>(farthestSaddle[nodeId]);
          dataType parentNodeValue = tree->getValue<dataType>(parentNodeId);
          dataType diffValue = std::max(nodeValue, parentNodeValue)
                               - std::min(nodeValue, parentNodeValue);
          if(diffValue <= epsilon) {
            ftm::idNode nodeIdToDelete, nodeIdToKeep;
            if(mergeByPersistence) {
              auto birthDeath1 = tree->getBirthDeath<dataType>(nodeId);
              auto birthDeath2 = tree->getBirthDeath<dataType>(parentNodeId);
              dataType pers1
                = std::get<1>(birthDeath1) - std::get<0>(birthDeath1);
              dataType pers2
                = std::get<1>(birthDeath2) - std::get<0>(birthDeath2);
              nodeIdToDelete = (pers1 > pers2) ? parentNodeId : nodeId;
              nodeIdToKeep = (pers1 > pers2) ? nodeId : parentNodeId;
              if(nodeIdToDelete == parentNodeId)
                nodeStack.emplace(nodeId);
            } else {
              nodeIdToDelete = nodeId;
              nodeIdToKeep = parentNodeId;
            }
            // Manage nodeMerged vector of vector
            for(auto node : treeNodeMerged[nodeIdToDelete]) {
              treeNodeMerged[nodeIdToKeep].push_back(node);
              if(isFarthest(farthestSaddle[nodeIdToKeep],
                            tree->getNode(node)->getOrigin()))
                farthestSaddle[nodeIdToKeep] = tree->getNode(node)->getOrigin();
            }
            treeNodeMerged[nodeIdToKeep].push_back(
              tree->getNode(nodeIdToDelete)->getOrigin());
            if(isFarthest(farthestSaddle[nodeIdToKeep], nodeIdToDelete))
              farthestSaddle[nodeIdToKeep] = nodeIdToDelete;
            treeNodeMerged[nodeIdToDelete].clear();
            // Delete node
            tree->deleteNode(nodeIdToDelete);
          }
        }
      }

      if(fullMerge) {
        auto root = tree->getRoot();
        tree->getNode(root)->setOrigin(root);
      }
    }

    // Epsilon 2 and 3 processing
    template <class dataType>
    void persistenceMerging(ftm::FTMTree_MT *tree,
                            double epsilon2,
                            double epsilon3 = 100) {
      bool fullMerge = (epsilon2 == 0);
      fullMerge &= useFullMerge_;
      epsilon2 /= 100;
      epsilon3 /= 100;
      dataType maxPers = tree->getMaximumPersistence<dataType>();

      std::queue<ftm::idNode> queue;
      queue.emplace(tree->getRoot());
      while(!queue.empty()) {
        ftm::idNode const node = queue.front();
        queue.pop();
        ftm::idNode const nodeParent = tree->getParentSafe(node);
        if(!tree->isRoot(node)) {
          const double nodePers = tree->getNodePersistence<dataType>(node);
          const double nodeParentPers
            = tree->getNodePersistence<dataType>(nodeParent);
          if(nodePers / nodeParentPers > epsilon2
             and nodePers / maxPers < epsilon3)
            tree->setParent(node, tree->getParentSafe(nodeParent));
        }
        std::vector<ftm::idNode> children;
        tree->getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }

      if(fullMerge) {
        auto root = tree->getRoot();
        if(tree->getNode(root)->getOrigin() != (int)root) {
          tree->setParent(tree->getNode(root)->getOrigin(), root);
          tree->getNode(root)->setOrigin(root);
        }
      }
    }

    template <class dataType>
    void keepMostImportantPairs(ftm::FTMTree_MT *tree, int n, bool useBD) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, dataType>> pairs;
      tree->getPersistencePairsFromTree(pairs, useBD);
      n = std::max(n, 2); // keep at least 2 pairs
      int const index = std::max((int)(pairs.size() - n), 0);
      dataType threshold = std::get<2>(pairs[index]) * (1.0 - 1e-6)
                           / tree->getMaximumPersistence<dataType>() * 100.0;
      persistenceThresholding<dataType>(tree, threshold);
    }

    template <class dataType>
    void persistenceThresholding(ftm::FTMTree_MT *tree,
                                 double persistenceThresholdT,
                                 std::vector<ftm::idNode> &deletedNodes) {
      ftm::idNode const treeRoot = tree->getRoot();
      dataType maxPers = tree->getMaximumPersistence<dataType>();
      dataType threshold = persistenceThresholdT / 100 * maxPers;
<<<<<<< HEAD
      //std::cout <<"[PersistenceThreshold] float: " << threshold << "\n";
=======
>>>>>>> 069d7453f01a67d1aea1dbb8c050202f86addf89

      dataType secondMax = tree->getSecondMaximumPersistence<dataType>();
      bool keepOneZeroPersistencePair = (secondMax == 0 or maxPers == 0);
      if(threshold >= secondMax)
        threshold = (1.0 - 1e-6) * secondMax;

      for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i) {
        if(tree->isRoot(i))
          continue;
        dataType nodePers = tree->getNodePersistence<dataType>(i);
        if(nodePers == 0 and keepOneZeroPersistencePair
           and tree->getParentSafe(i) == treeRoot) {
          keepOneZeroPersistencePair = false;
          continue;
        }
        if((nodePers == 0 or nodePers <= threshold
            or not tree->isNodeOriginDefined(i))) {
          tree->deleteNode(i);
          deletedNodes.push_back(i);
          ftm::idNode const nodeOrigin = tree->getNode(i)->getOrigin();
          if(tree->isNodeOriginDefined(i)
             and tree->getNode(nodeOrigin)->getOrigin() == (int)i) {
            tree->deleteNode(nodeOrigin);
            deletedNodes.push_back(nodeOrigin);
          }
        }
      }
    }

    template <class dataType>
    void persistenceThresholding(ftm::FTMTree_MT *tree,
                                 std::vector<ftm::idNode> &deletedNodes) {
      persistenceThresholding<dataType>(
        tree, persistenceThreshold_, deletedNodes);
    }

    template <class dataType>
    void persistenceThresholding(ftm::FTMTree_MT *tree,
                                 double persistenceThresholdT) {
      std::vector<ftm::idNode> deletedNodes;
      persistenceThresholding<dataType>(
        tree, persistenceThresholdT, deletedNodes);
    }

    template <class dataType>
    void persistenceThresholding(ftm::FTMTree_MT *tree) {
      std::vector<ftm::idNode> deletedNodes;
      persistenceThresholding<dataType>(
        tree, persistenceThreshold_, deletedNodes);
    }

    template <class dataType>
    void verifyOrigins(ftm::FTMTree_MT *tree) {
      for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i)
        if(not tree->isNodeAlone(i) and not tree->isNodeOriginDefined(i)) {
          std::stringstream ss;
          std::vector<ftm::idNode> children;
          tree->getChildren(i, children);
          ss << i << " has no origin (scalar=" << tree->getValue<dataType>(i)
             << ", noChildren=" << children.size()
             << ", parent=" << tree->getParentSafe(i) << ")";
          printMsg(ss.str());
          if(!tree->isRoot(i))
            tree->deleteNode(i);
          else {
            std::stringstream ss2;
            ss2 << "the root has no origin!";
            printErr(ss2.str());
          }
        }
    }

    template <class dataType>
    void preprocessTree(ftm::FTMTree_MT *tree,
                        bool deleteInconsistentNodes = true) {
      if(deleteInconsistentNodes) {
        // Manage inconsistent critical points
        // Critical points with same scalar value than parent
        for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i)
          if(!tree->isNodeAlone(i) and !tree->isRoot(i)
             and tree->getValue<dataType>(tree->getParentSafe(i))
                   == tree->getValue<dataType>(i)) {
            /*printMsg("[preprocessTree] " + std::to_string(i)
                     + " has same scalar value than parent (will be
               deleted).");*/
            tree->deleteNode(i);
          }
        // Valence 2 nodes
        for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i)
          if(tree->getNode(i)->getNumberOfUpSuperArcs() == 1
             and tree->getNode(i)->getNumberOfDownSuperArcs() == 1) {
            /*printMsg("[preprocessTree] " + std::to_string(i)
                     + " has 1 up arc and 1 down arc (will be deleted).");*/
            tree->deleteNode(i);
          }
      }

      // Compute persistence pairs
      if(not isPersistenceDiagram_ or convertToDiagram_) {
        auto pairs = ftm::computePersistencePairs<dataType>(tree);
        // Verify pairs
        verifyOrigins<dataType>(tree);
      }
    }

    template <class dataType> 
    std::vector<char> globalThresholdingCBD(std::shared_ptr<ftm::MergeTree<dataType>> &CBD, std::vector<ftm::idNode> &dataMap, int additionalChoices, ftm::MergeTree<dataType>* inputMTptr){
      unsigned int numNodes = CBD->tree.getNumberOfNodes();
      std::vector<ftm::idNode> leaves;
      inputMTptr->tree.getLeavesFromTree(leaves);
      int numLeaves = leaves.size();
      std::vector<dataType> leavesValues(numLeaves);

      std::transform(
        leaves.begin(), 
        leaves.end(), 
        leavesValues.begin(),
        [&inputMTptr](int id) {
          return inputMTptr->tree.template getNodePersistence<dataType>(id);
        }
      );

      std::partial_sort(leavesValues.begin(), leavesValues.begin() + additionalChoices, leavesValues.end(), std::greater<dataType>{});
      //std::sort(
      //  leavesValues.begin(),
      //  leavesValues.end(),
      //  std::greater<dataType>{}
      //);

      //leavesValues.erase( std::unique( leavesValues.begin(), leavesValues.end() ), leavesValues.end() );
      //Top additionalChoices entries, duplicates irrelevant; additionalChoices-th entry might be wrong
      dataType lastRankedVal = additionalChoices > 0 ? leavesValues[additionalChoices] : std::numeric_limits<dataType>::max();
       
      std::vector<char> stay(numNodes, true);

      
      std::vector<ftm::idNode> subtrees;
      std::vector<ftm::idNode> branches;
      for (unsigned int i = 0; i < numNodes; ++i) {
        if (CBD->tree.getNode(i)->getIsSubtree()) {
          subtrees.push_back(i);
        } 
      }
      

      //#pragma omp parallel for schedule(dynamic, 64)
      for (ftm::idNode i : subtrees) {
        std::vector<ftm::idNode> iChildren;
        CBD->tree.getChildren(i, iChildren);
      
        int numChildren = iChildren.size();
        std::vector<std::pair<ftm::idNode, dataType>> iValues(numChildren);
        
        std::transform(
          iChildren.begin(), 
          iChildren.end(),
          iValues.begin(),
          [&dataMap,&inputMTptr](int id) {
            return std::make_pair(id, inputMTptr->tree.template getNodePersistence<dataType>(dataMap[id]));
          }
        );

        unsigned int maxChoices = numChildren < (additionalChoices+1) ? numChildren : (additionalChoices +1);
        //std::sort(
        //  iValues.begin(),
        //  iValues.end(),
        //  [](auto &left, auto &right) {
        //    return left.second > right.second;
        //});

        std::partial_sort(iValues.begin(), iValues.begin() + maxChoices , iValues.end(), [](auto &left, auto &right) {
          return left.second > right.second;
        });
        
        for (int j = numChildren-1; j >= 0; --j) {
          if (iValues[j].second >= lastRankedVal) break;
          stay[iValues[j].first] = false;
        }
        stay[iValues[0].first] = true;
        CBD->tree.getNode(i)->setOrigin(iValues[0].first);
        //std::cout << i << " 's Origin is set to " << iValues[0].first << " \n";
      }

      return stay;
    }
    
    template <class dataType> 
    std::vector<char> localThresholdingCBD(std::shared_ptr<ftm::MergeTree<dataType>> &CBD, std::vector<ftm::idNode> &dataMap, int additionalChoices, ftm::MergeTree<dataType>* inputMTptr){
    unsigned int numNodes = CBD->tree.getNumberOfNodes();
    std::vector<ftm::idNode> subtrees;
    for (unsigned int i = 0; i < numNodes; ++i) {
      if (CBD->tree.getNode(i)->getIsSubtree()) {
        subtrees.push_back(i);
      }
    }
    std::vector<char> stay(numNodes, true);
    //#pragma omp parallel for schedule(dynamic, 64)
    for (ftm::idNode i : subtrees) {
      std::vector<ftm::idNode> iChildren;
      CBD->tree.getChildren(i, iChildren);
    
      size_t numChildren = iChildren.size();
      std::vector<std::pair<ftm::idNode, dataType>> iValues(numChildren);
      
      std::transform(
        iChildren.begin(), 
        iChildren.end(),
        iValues.begin(),
        [&dataMap,&inputMTptr](int id) {
          return std::make_pair(id, inputMTptr->tree.template getNodePersistence<dataType>(dataMap[id]));
        }
      );

      std::sort(iValues.begin(), iValues.end(), [](auto &left, auto &right) {
        return left.second > right.second;
      });


      for (unsigned int j = additionalChoices; j < numChildren; ++j) {
        stay[iValues[j].first] = false;
      }
      stay[iValues[0].first] = true;
    }
    return stay;
    }

    template <class dataType> 
    std::shared_ptr<ftm::MergeTree<dataType>> transformToThresholdCBD(std::shared_ptr<ftm::MergeTree<dataType>> &CBD, std::vector<ftm::idNode> &dataMap, int additionalChoices,ftm::MergeTree<dataType>* inputMTptr){
      unsigned int numNodes = CBD->tree.getNumberOfNodes();   
      
      std::vector<char> stay = globalThreshold_ ? globalThresholdingCBD(CBD,dataMap, additionalChoices,inputMTptr) : localThresholdingCBD(CBD, dataMap, additionalChoices, inputMTptr);
      //Consequence deletes
      std::vector<char> reachable(numNodes, 0);
      reachable[0] = 1;  // root

      std::queue<ftm::idNode> traverseQueue;
      traverseQueue.push(0);
      while (!traverseQueue.empty()) {
        ftm::idNode currNode = traverseQueue.front();
        traverseQueue.pop();
        reachable[currNode] = 1;

        std::vector<ftm::idNode> res;
        CBD->tree.getChildren(currNode, res);
        for (ftm::idNode c : res) {
          if (reachable[c] != 1 && stay[c] == 1) {
            traverseQueue.push(c);
          }
        }
      }
    
      /*
      while (!frontier.empty()) {
          std::vector<ftm::idNode> next;

          #pragma omp parallel
          {
              std::vector<int> local;
              #pragma omp for schedule(dynamic, 64) nowait
              for (int k = 0; k < (int)frontier.size(); ++k) {
                  ftm::idNode p = frontier[k];

                  std::vector<ftm::idNode> ch;
                  CBD->tree.getChildren(p, ch);

                  for (auto c : ch) {
                      if (!stay[c]) continue;        
                      char expected = 0;
                      if (__atomic_compare_exchange_n(&reachable[c], &expected, (char)1,
                                                      false,
                                                      __ATOMIC_RELAXED,
                                                      __ATOMIC_RELAXED)) {
                          local.push_back(c);       
                      }
                  }
              }
              #pragma omp critical
              next.insert(next.end(), local.begin(), local.end());
          }
          frontier.swap(next);
      }
      
      */


      
      unsigned int numTCBD = 0;
      //#pragma omp parallel for schedule(static) reduction(+:numTCBD)
      for (unsigned int i = 0; i < numNodes; ++i){
        //This is irrelevant as we expect that reachable includes all stay nodes, but just in case 
        stay[i] = stay[i] && reachable[i];
        if(stay[i])
          numTCBD++;
      }

      if (cbdDebug) { 
        std::cout << "Stay: \n";
        for (unsigned int i = 0; i < numNodes; ++i){
          std::cout << "(" << i <<", "  << bool(stay[i]) << "); ";
        }
      }
      
          
      auto thresholdCBDptr = std::make_shared<ftm::MergeTree<dataType>>(ttk::ftm::createEmptyMergeTree<dataType>(numTCBD));
      
      std::vector<ftm::idNode> oldToNew(numNodes);

      std::vector<dataType>scalarsTCBD = std::vector<dataType>(numTCBD);

      for ( unsigned int i = dataMap.size() - 1 ; i <=  dataMap.size() -1 ; i-- ) {
        if(!stay[i]){
          dataMap.erase(dataMap.begin() + i);
        }
      }
      
      
      unsigned int numAddedNodes = 0;
      for(unsigned int nId = 0; nId < CBD->tree.getNumberOfNodes(); ++nId){
            if (!stay[nId]) continue;
            thresholdCBDptr->tree.makeNode(numAddedNodes);

            ttk::ftm::Node* newNode = thresholdCBDptr->tree.getNode(numAddedNodes);
            ttk::ftm::Node* oldNode = CBD->tree.getNode(nId);
            oldToNew[nId] = numAddedNodes;
            newNode->setIsSubtree(oldNode->getIsSubtree());

            
            
            
            
            if (!oldNode->getIsSubtree() || sortForestSolverInput) {
              newNode->setOrigin(oldToNew[oldNode->getOrigin()]); //Should be fine because the only relevant origins are the direct unique parent and by inout CBD's construction they should appear beforehand
            }
            if (sortForestSolverInput) {
              newNode->setDataMap(oldNode->getDataMap());
            }
            
            dataType val = CBD->tree.template getValue<dataType>(nId);
            scalarsTCBD[numAddedNodes] = val;
            ++numAddedNodes;
      }

      //Temporary as Origin must be stored for both branch nodes and subtrees, meaning oldToNew must be fully set... This is usually unnecessary
      if (sortForestSolverInput) {
        for(unsigned int nId = 0; nId < CBD->tree.getNumberOfNodes(); ++nId){
          if (!stay[nId]) continue;
          ttk::ftm::Node* oldNode = CBD->tree.getNode(nId);
          ttk::ftm::Node* newNode = thresholdCBDptr->tree.getNode(oldToNew[nId]);

          newNode->setOrigin(oldToNew[oldNode->getOrigin()]);
        }
      }
      

      //Add all arcs of cciIH 
      for(unsigned int aId = 0; aId < CBD->tree.getNumberOfSuperArcs(); ++aId){ //(By conventions fine traversal of arcs)
        ttk::ftm::SuperArc* currArc = CBD->tree.getSuperArc(aId);
        auto upNodeId = currArc->getUpNodeId();
        auto downNodeId = currArc->getDownNodeId();

        if (stay[upNodeId] && stay[downNodeId] ) {
          thresholdCBDptr->tree.makeSuperArc(oldToNew[downNodeId], oldToNew[upNodeId]);              
        }
      }

      thresholdCBDptr->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsTCBD);
      thresholdCBDptr->scalars->values = (void *)(thresholdCBDptr->scalarsValues->data());
      thresholdCBDptr->scalars->size = thresholdCBDptr->scalarsValues->size();
      
      return thresholdCBDptr;
    }

  
    template <class dataType> 
    std::pair<std::shared_ptr<ftm::MergeTree<dataType>>,std::vector<ftm::idNode>> computeCompleteBranchDecomposition(
      ftm::MergeTree<dataType>* inputMTptr, ftm::idNode pId, ftm::idNode cId ) {
      
      using DataMap = std::vector<ftm::idNode>;
      bool useDataMap = postprocess_ || useThresholdCBD_ || sortForestSolverInput;
      //=====================================================
      //Bottom up  through Edge Recursion of Input MergeTree
      //=====================================================
      //Conventions for the result CBD for easier handling throughout the algorithm: 
      //- Subtree node resembling the entire tree are at entry 0
      //- Subtree nodes store the higher(closer to root) node's scalar to have an origin for the branch nodes. 
      //- Branch nodes store the lowest node's (the leaf's) scalar
        
      
      std::vector<ftm::idNode> ccs; 
      inputMTptr->tree.getChildren(cId, ccs);
      unsigned int numCcs = ccs.size();

      

      for(unsigned int i = 0; i < numCcs; ++i){
      }    


      //  Base case
      //  O parent p
      //  |                      
      //  O child c

      if(numCcs ==  0) {
        ftm::MergeTree<dataType> in2 = ttk::ftm::createEmptyMergeTree<dataType>(2);
        std::shared_ptr<ftm::MergeTree<dataType>> completeBDptr = std::make_shared<ftm::MergeTree<dataType>>(in2);

        completeBDptr->tree.makeNode(0); 
        completeBDptr->tree.makeNode(1);

        completeBDptr->tree.makeSuperArc(1, 0);
        completeBDptr->tree.getNode(0)->setOrigin(1);  
        completeBDptr->tree.getNode(1)->setOrigin(0);
        
        completeBDptr->tree.getNode(0)->setIsSubtree(true);

        std::vector<dataType> scalarsCBD;
        scalarsCBD.push_back(inputMTptr->tree.template getValue<dataType>(pId));
        scalarsCBD.push_back(inputMTptr->tree.template getValue<dataType>(cId));

        completeBDptr->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBDptr->scalars->values = (void *)(completeBDptr->scalarsValues->data());
        completeBDptr->scalars->size = completeBDptr->scalarsValues->size();

        DataMap dataMap;
        if (useDataMap) {
          dataMap = DataMap(2);
          dataMap[0] = pId;
          dataMap[1] = cId;
          //Temporary
          if (sortForestSolverInput) {
            completeBDptr->tree.getNode(0)->setDataMap(pId);
            completeBDptr->tree.getNode(1)->setDataMap(cId);
          }
        }
        

        return {completeBDptr, dataMap};
      }
      else{ 
        
        //  Step case
        //      O parent p
        //      |
        //      O child c
        //   ( ... ) 
        //   O     O children ccs

        //---------------------------------------------------------
        //Recurse
        std::vector<std::shared_ptr<ftm::MergeTree<dataType>>> recursion_results(numCcs);
        std::vector<DataMap> recursion_DataMaps(numCcs);
        
        #ifdef TTK_ENABLE_OPENMP4
        #pragma omp parallel for if(parallelFor)
        #endif
        for(unsigned int i = 0; i < numCcs; ++i){
          auto result = computeCompleteBranchDecomposition<dataType>(inputMTptr, cId, ccs[i]);
          recursion_results[i] = result.first;
          recursion_DataMaps[i] = result.second;          
        }    
        
        //---------------------------------------------------------
        //Computing the number of nodes in the result
        
                        
        unsigned int numNodesResult = 1; //main subtree node
        for(unsigned int i = 0; i < numCcs; ++i){
          std::shared_ptr<ftm::MergeTree<dataType>> currCBD = recursion_results[i];
          unsigned int currNum = currCBD->scalars->size;
          numNodesResult += currNum; //add on of recursive result
          std::vector<ftm::idNode> cs;
          currCBD->tree.getChildren(0,cs);
          numNodesResult += cs.size(); //add extended main branches (c - l while l some leaf) from recursion result to p - l          
        }
        //---------------------------------------------------------
        //Initialize CBD result
        ftm::MergeTree<dataType> in2 = ttk::ftm::createEmptyMergeTree<dataType>(numNodesResult);
        std::shared_ptr<ftm::MergeTree<dataType>> completeBDptr = std::make_shared<ftm::MergeTree<dataType>>(in2);
        
        *completeBDptr = in2;

        //Remember scalars throughout merging and later steps to set at the end.
        std::vector<dataType>scalarsCBD = std::vector<dataType>(numNodesResult);
        

        //Make main subtree node
        completeBDptr->tree.makeNode(0); 
        scalarsCBD[0] = inputMTptr->tree.template getValue<dataType>(pId);
        completeBDptr->tree.getNode(0)->setIsSubtree(true);

        DataMap dataMap;
        if(useDataMap){
          dataMap = DataMap(numNodesResult);
          dataMap[0] = pId;
          //Temporary
          if (sortForestSolverInput) {
            completeBDptr->tree.getNode(0)->setDataMap(pId);
          }
        }
        

        //---------------------------------------------------------
        //Merge recursion results

        unsigned int numAddedNodes = 1;

        //References to main subtree nodes of the recursion results in merged CBDs
        std::vector<ftm::idNode> mainSTNodes = std::vector<ftm::idNode>(numCcs);

        for(unsigned int i = 0; i < numCcs; ++i){
          //cciIH := CBD of Merge Tree rooted at [c,cc_i] through induction hypothesis 
          std::shared_ptr<ftm::MergeTree<dataType>> cciIH = recursion_results[i];
          DataMap ldmIH = recursion_DataMaps[i];

          unsigned int currOffset = numAddedNodes;
          mainSTNodes[i] = currOffset;
        
          //Add all nodes, their origins, scalar values and subtree bool of cciIH
          for(unsigned int nId = 0; nId < cciIH->tree.getNumberOfNodes(); ++nId){//(By conventions fine traversal of arcs)
            unsigned int newId = currOffset + nId;
            completeBDptr->tree.makeNode(newId);

            ttk::ftm::Node* newNode = completeBDptr->tree.getNode(newId);
            ttk::ftm::Node* oldNode = cciIH->tree.getNode(nId);
            newNode->setOrigin(oldNode->getOrigin() + currOffset); //Get origin of node, set it with offset
            newNode->setIsSubtree(oldNode->getIsSubtree());

            numAddedNodes += 1;
            dataType val = cciIH->tree.template getValue<dataType>(nId);

            scalarsCBD[newId] = val;
            if(useDataMap){ 
              dataMap[newId] = ldmIH[nId];
              //Temporary
              if (sortForestSolverInput) {
                completeBDptr->tree.getNode(newId)->setDataMap(nId);
              }
            }
          }

          //Add all arcs of cciIH 
          for(unsigned int aId = 0; aId < cciIH->tree.getNumberOfSuperArcs(); ++aId){ //(By conventions fine traversal of arcs)
            ttk::ftm::SuperArc* currArc = cciIH->tree.getSuperArc(aId);
            completeBDptr->tree.makeSuperArc(currArc->getDownNodeId()+ currOffset , currArc->getUpNodeId()+ currOffset);
          }
        }
        //---------------------------------------------------------
        //Fill up the CBD 
        // - Add Extended main branches (c-l) of the cciIHs to (p-l)
        // - Add arc [Main subtree node, p-l]
        // - Add arcs of (c-l) to (p-l) (all adjacent subtree nodes to (c-l) are also adjacent to (p-l))
        // - Add arcs [p-l, mainSTNodes[j]] with j not i

        for(unsigned int i = 0; i < numCcs; ++i){
          std::vector<ftm::idNode> mainbranches;
          completeBDptr->tree.getChildren(mainSTNodes[i],mainbranches);

          for(ftm::idNode currMainBranch : mainbranches){
            unsigned int currExtendedMainBranch = numAddedNodes;

            //Add (p-l) as a node
            completeBDptr->tree.makeNode(currExtendedMainBranch);
            scalarsCBD[currExtendedMainBranch] = scalarsCBD[currMainBranch];
            completeBDptr->tree.getNode(currExtendedMainBranch)->setOrigin(0); 
            numAddedNodes += 1;

            if(useDataMap)
              dataMap[currExtendedMainBranch] = dataMap[currMainBranch];

            //Add arc [Main subtree node, p-l]
            completeBDptr->tree.makeSuperArc(currExtendedMainBranch, 0);
            
            //Add arcs of (c-l) to (p-l)
            std::vector<ftm::idNode> adjSubtreesCL;
            completeBDptr->tree.getChildren(currMainBranch, adjSubtreesCL);
            for(ftm::idNode adjTree : adjSubtreesCL){
              completeBDptr->tree.makeSuperArc(adjTree,currExtendedMainBranch);
            }

            //Add arcs [p-l, mainSTNodes[j]] with j not i
            for(unsigned int j = 0; j < numCcs; ++j){
              if(i != j){
                completeBDptr->tree.makeSuperArc(mainSTNodes[j], currExtendedMainBranch);
              }
            }
          } 
        }
        //---------------------------------------------------------
        completeBDptr->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBDptr->scalars->values = (void *)(completeBDptr->scalarsValues->data());
        completeBDptr->scalars->size = completeBDptr->scalarsValues->size();
        return {completeBDptr, dataMap};
      }
    }

    template <class dataType> 
    std::shared_ptr<ftm::MergeTree<dataType>> computeCompleteBranchDecomposition(ftm::MergeTree<dataType>* inputMTptr, std::vector<ftm::idNode> &dataMap){
      std::vector<ftm::idNode> children;
      inputMTptr->tree.getChildren(inputMTptr->tree.getRoot(),children);

      auto result = computeCompleteBranchDecomposition<dataType>(inputMTptr, inputMTptr->tree.getRoot(), children[0]);
      auto CBD = result.first;
      dataMap = result.second;

      if (useThresholdCBD_) {
        std::vector<ftm::idNode> leaves;
        inputMTptr->tree.getLeavesFromTree(leaves);


        // number of leaves decremented as most persistent branch either way a choice
        unsigned int additionalChoices = static_cast<unsigned int>(((leaves.size()-1) * thresholdOfCBD_) / 100.0);
        if (cbdDebug) {
          std::cout << "Additional Choices:" << additionalChoices << "\n"
                    << "Leaves size: " << leaves.size() << "\n"
                    << "thresholdOfCBD: " << thresholdOfCBD_ << "\n";
        }
        CBD = transformToThresholdCBD<dataType>(CBD, dataMap, additionalChoices, inputMTptr);
      }
      
      return CBD;
      
    }

    template <class dataType> 
    ftm::FTMTree_MT *computeBranchDecomposition(
      ftm::FTMTree_MT *tree,
      std::vector<std::vector<ftm::idNode>> &treeNodeMerged) {
      ftm::FTMTree_MT *treeNew = tree;

      ftm::idNode const root = treeNew->getRoot();

      // Manage when there is only one pair
      if(tree->isThereOnlyOnePersistencePair()) {
        ftm::idNode const rootOrigin = treeNew->getNode(root)->getOrigin();
        treeNew->getNode(rootOrigin)->setOrigin(rootOrigin);
        return treeNew;
      }

      // Manage multi persistence pairing
      std::vector<std::vector<ftm::idNode>> treeMultiPers;
      tree->getMultiPersOriginsVectorFromTree(treeMultiPers);

      // General case
      std::vector<bool> nodeDone(tree->getNumberOfNodes(), false);
      std::queue<ftm::idNode> queueNodes;
      queueNodes.emplace(root);
      while(!queueNodes.empty()) {
        ftm::idNode const node = queueNodes.front();
        queueNodes.pop();
        ftm::idNode const nodeOrigin = treeNew->getNode(node)->getOrigin();
        if(node == nodeOrigin
           or treeNew->getNodeLevel(node) > treeNew->getNodeLevel(nodeOrigin))
          continue;

        // Init vector with all origins
        std::vector<std::tuple<ftm::idNode, int>> vecOrigins;
        for(auto nodeMergedOrigin : treeNodeMerged[node]) {
          vecOrigins.emplace_back(nodeMergedOrigin, 0);
          for(auto multiPersOrigin :
              treeMultiPers[tree->getNode(nodeMergedOrigin)->getOrigin()])
            vecOrigins.emplace_back(multiPersOrigin, 1);
        }
        if(not tree->isNodeMerged(node))
          for(auto multiPersOrigin : treeMultiPers[node])
            vecOrigins.emplace_back(multiPersOrigin, 1);
        vecOrigins.emplace_back(nodeOrigin, 2);

        bool splitRoot = (vecOrigins.size() != 1 and treeNew->isRoot(node));
        splitRoot = false; // disabled

        // Process each origin
        for(auto stackTuple : vecOrigins) {
          ftm::idNode const nodeOriginT = std::get<0>(stackTuple);
          int const nodeOriginTID = std::get<1>(stackTuple);
          if(nodeDone[nodeOriginT]
             and nodeDone[tree->getNode(nodeOriginT)->getOrigin()])
            continue;
          nodeDone[nodeOriginT] = true;
          nodeDone[tree->getNode(nodeOriginT)->getOrigin()] = true;

          // Manage new parent
          ftm::idNode newParent = node;
          // - if merged node
          if(nodeOriginTID == 0) {
            newParent = treeNew->getNode(nodeOriginT)->getOrigin();
            treeNew->setParent(newParent, treeNew->getParentSafe(node));
            // - if multi pers node or nodeOrigin and splitRoot
          } else if(nodeOriginTID == 1 or (nodeOriginTID == 2 and splitRoot)) {
            newParent = nodeOriginT;
          }

          // Set nodes in the branch as childrens of the node
          ftm::idNode parentNodeOrigin = treeNew->getParentSafe(nodeOriginT);
          while(parentNodeOrigin != node) {
            ftm::idNode const oldParentNodeOrigin
              = treeNew->getParentSafe(parentNodeOrigin);
            treeNew->setParent(parentNodeOrigin, newParent);
            parentNodeOrigin = oldParentNodeOrigin;
          }

          if(nodeOriginTID == 1 or (nodeOriginTID == 2 and splitRoot))
            treeNew->setParent(newParent, treeNew->getParentSafe(node));
          else // if(nodeOriginTID != 1) // if not a multi pers node
            // Delete the other node of the pair
            treeNew->deleteNode(nodeOriginT);
          if(nodeOriginTID == 2 and splitRoot)
            tree->getNode(node)->setOrigin(node);

          // Push childrens of the node to the stack to process them
          std::vector<ftm::idNode> childrenNode;
          treeNew->getChildren(newParent, childrenNode);
          for(ftm::idNode const children : childrenNode)
            if(!treeNew->isLeaf(children))
              queueNodes.emplace(children);
        }
      }

      // Verify inconsistency
      // verifyBranchDecompositionInconsistency<dataType>(treeNew);

      return treeNew;
    }

    template <class dataType>
    void dontUseMinMaxPair(ftm::FTMTree_MT *tree) {
      ftm::idNode const treeRoot = tree->getRoot();
      // Full merge case, search for the origin
      if(tree->getNode(treeRoot)->getOrigin() == (int)treeRoot) {
        ftm::idNode const nodeIdToDelete
          = tree->getMergedRootOrigin<dataType>();
        if(nodeIdToDelete != treeRoot
           and not tree->isNodeIdInconsistent(nodeIdToDelete)) {
          if(tree->isThereOnlyOnePersistencePair())
            tree->getNode(nodeIdToDelete)->setOrigin(nodeIdToDelete);
          else
            tree->deleteNode(nodeIdToDelete);
        }
        // Classic case
      } else {
        ftm::idNode const rootOrigin = tree->getNode(treeRoot)->getOrigin();
        if(tree->isThereOnlyOnePersistencePair())
          tree->getNode(rootOrigin)->setOrigin(rootOrigin);
        else
          tree->deleteNode(rootOrigin);
      }

      tree->getNode(treeRoot)->setOrigin(treeRoot);
    }

    void verifyPairsTree(ftm::FTMTree_MT *tree) {
      int cptBug = 0;
      for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i) {
        if(not tree->isNodeOriginDefined(i)) {
          std::stringstream ss;
          ss << i << " _ " << tree->getNode(i)->getOrigin() << " / "
             << tree->getNumberOfNodes();
          printMsg(ss.str());
          if(tree->isNodeAlone(i))
            printMsg("alone");
          cptBug++;
        }
      }
      std::stringstream ss;
      ss << cptBug;
      printMsg(ss.str());
    }

    template <class dataType>
    void deleteMultiPersPairs(ftm::FTMTree_MT *tree, bool useBD) {
      auto multiPersOrigins = tree->getMultiPersOrigins<dataType>(useBD);
      for(auto origin : multiPersOrigins)
        tree->deleteNode(origin);
    }

    template <class dataType>
    void MA_mditz_print(ftm::MergeTree<dataType>& exp){
      for (unsigned int i = 0; i < exp.tree.getNumberOfNodes(); ++i){
        std::vector<ftm::idNode> cs;
        std::vector<ftm::idNode> ps;
        exp.tree.getChildren(i, cs);
        exp.tree.getParents_DAG(i, ps);

        if(cs.empty() && ps.empty())
          continue;

        std::cout << "Node " << i ; 
        std::cout << "\n---Scalar: "  << exp.tree.template getValue<dataType>(i);
        std::cout << "\n---Is subtree: " << exp.tree.getNode(i)->getIsSubtree() ;
        //Temporary
        if(!exp.tree.getNode(i)->getIsSubtree() || sortForestSolverInput){
          std::cout << "\n---Origin: " << exp.tree.getNode(i)->getOrigin();
          std::cout << "\n---Origin Scalar: " << exp.tree.template getValue<dataType>(exp.tree.getNode(i)->getOrigin());
        }
        std::cout << "\n---Children: ";
        for (ftm::idNode c: cs){
          std::cout << c << ", ";
        }
        std::cout << "\n    Parents: ";
        
        for (ftm::idNode c: ps){
          std::cout << c << ", ";
        }
        std::cout << "\n";
      }
      std::cout << std::endl;
    }

    template <class dataType>
    void preprocessingPipeline(ftm::MergeTree<dataType> &mTree,
                               double epsilonTree,
                               double epsilon2Tree,
                               double epsilon3Tree,
                               bool branchDecompositionT,
                               bool useMinMaxPairT,
                               bool cleanTreeT,
                               double persistenceThreshold,
                               std::vector<int> &nodeCorr,
                               std::vector<ftm::idNode> &dataMap, //mditz
                               ftm::MergeTree<dataType> &preprocessed_MT,
                               bool deleteInconsistentNodes = true,
                               bool removeMergedSaddles = false,
                               int treeIdx = -1
                              ) {
      
      Timer t_proc;
      std::stringstream ss;

      ftm::FTMTree_MT *tree = &(mTree.tree);
      size_t inputTreeSize = tree->getRealNumberOfNodes();
      
      preprocessTree<dataType>(tree, deleteInconsistentNodes);
      
      // - Delete null persistence pairs and persistence thresholding
      persistenceThresholding<dataType>(tree, persistenceThreshold);
<<<<<<< HEAD
      //std::cout << "[PersistenceThreshold] Percentage: " << persistenceThreshold<< "\n";
=======
>>>>>>> 069d7453f01a67d1aea1dbb8c050202f86addf89
      
      // - Merge saddle points according epsilon
      std::vector<std::vector<ftm::idNode>> treeNodeMerged(
        tree->getNumberOfNodes());
      if(not isPersistenceDiagram_ or convertToDiagram_) {
        if(epsilonTree != 0) {
          mergeSaddle<dataType>(tree, epsilonTree, treeNodeMerged);
          if(removeMergedSaddles) {
            for(unsigned int j = 0; j < treeNodeMerged.size(); j++) {
              for(auto k : treeNodeMerged[j]) {
                auto nodeToDelete = tree->getNode(k)->getOrigin();
                tree->getNode(k)->setOrigin(j);
                tree->getNode(nodeToDelete)->setOrigin(-1);
              }
            }
          }
        }
      }
      size_t preprocTreeSize = tree->getRealNumberOfNodes();

      if (statsTest){
        std::cout << "[StatsTest] MTn: " << tree->getRealNumberOfNodes() << "\n";
        std::cout << "[StatsTest] MTm: " << tree->getRealNumberOfSuperArcs() << "\n";
      }

        if (cbdDebug) {
           std::cout << "\n\n========================================\n"
          << "     Preprocessed Merge Tree     \n"
          << "========================================\n";
          MA_mditz_print(mTree);
          std::cout << "========================================\n";
        }
      
      if(MA_mditz){
        
        preprocessed_MT = ftm::copyMergeTree(mTree);
        /*
        if (cbdDebug) {
          std::cout << "\n\n========================================\n"
          << "     getNodePersistences     \n"
          << "========================================\n";

          for (unsigned int i = 0; i< tree->getNumberOfNodes(); i++) {
            std::cout << "Node " << i << ": " << mTree.tree.template getNodePersistence<dataType>(i) << "\n";
          }
        }
        */

        mTree = *computeCompleteBranchDecomposition<dataType>(&mTree, dataMap);
        if (shortTreeStats) {
          /*
          std::vector<ftm::idNode> subtrees;
          for (unsigned int i = 0; i < tree->getNumberOfNodes(); ++i) {
            if (tree->getNode(i)->getIsSubtree()) {
              subtrees.push_back(i);
            } 
          }
          */
          ss << "[" <<treeIdx <<"]:|MTinput|:" << inputTreeSize << ";|MTprocessed|:" << preprocTreeSize<< ";|CBD|:" << tree->getNumberOfNodes()<<";|R(CBD)|:"<<tree->getNumberOfSuperArcs(); 
          std::cout << ss.str() << std::endl;
          ss.str("");
          ss.clear();
          
        }

        if (cbdDebug) {
          std::string title = useThresholdCBD_ ? "Thresholded CBD with " + std::to_string(thresholdOfCBD_) + "\% delta" : "Complete Branch Decomposition";
           std::cout << "\n\n========================================\n"
          << "     "<< title <<"     \n"
          << "========================================\n";
          MA_mditz_print(mTree);
          std::cout << "========================================\n";
        }
        

        tree = &(mTree.tree);
        if (statsTest){
          std::cout << "[StatsTest] CBDn: " << tree->getRealNumberOfNodes() << "\n";
          std::cout << "[StatsTest] CBDm: " << tree->getRealNumberOfSuperArcs() << "\n";
        }
        
      }
      
      if(not MA_mditz and branchDecompositionT
         and (not isPersistenceDiagram_ or convertToDiagram_)){
        tree = computeBranchDecomposition<dataType>(tree, treeNodeMerged);
        
        if (shortTreeStats) {
          
          ss << "[" <<treeIdx <<"]:|MT|:" << preprocTreeSize << ";|BDT|:" << tree->getRealNumberOfNodes() << "\n";
          std::cout << ss.str() ;
        }
        if (cbdDebug) {
          std::cout << "BDT:\n";
          MA_mditz_print(mTree);
        }
        
        if (statsTest){
          std::cout << "[StatsTest] BDTn: " << tree->getRealNumberOfNodes() << "\n";
          std::cout << "[StatsTest] BDTm: " << tree->getRealNumberOfSuperArcs() << "\n";
        }
        
      }
      
      // - Delete multi pers pairs
      if(deleteMultiPersPairs_ and not MA_mditz)
        deleteMultiPersPairs<dataType>(tree, branchDecompositionT);

      
      // - Remove min max pair
      // verifyPairsTree(tree);
      if(not useMinMaxPairT and not MA_mditz)
        dontUseMinMaxPair<dataType>(tree);

      
      // - Epsilon 2 and 3 processing
      if(branchDecompositionT and not isPersistenceDiagram_ and not MA_mditz)
        persistenceMerging<dataType>(tree, epsilon2Tree, epsilon3Tree);

      // - Tree cleaning (remove unused nodes)
      if(cleanTreeT and not MA_mditz) {
        ftm::cleanMergeTree<dataType>(mTree, nodeCorr, branchDecompositionT);
        tree = &(mTree.tree);
        reverseNodeCorr(tree, nodeCorr);
      }
      
      if (branchDecompositionT && not MA_mditz && sortForestSolverInput) {
        for (unsigned int i = 0; i < tree->getNumberOfNodes(); i++) {
          tree->getNode(i)->setDataMap(nodeCorr[i]);
        }
      }

      // - Root number verification
      if(tree->getNumberOfRoot() != 1)
        printErr("preprocessingPipeline tree->getNumberOfRoot() != 1");

      // - Time printing
      // verifyPairsTree(tree);
      auto t_preproc_time = t_proc.getElapsedTime();
      ss << "TIME PREPROC.   = " << t_preproc_time;
      printMsg(ss.str(), debug::Priority::VERBOSE);
      ss.str("");
      ss.clear();
      ss << "[Time] PreProcTime: " << t_preproc_time <<"\n";
      std::cout << ss.str() ;
    }

    
    //No persistenceThreshold and dataMap+PreprocessSave
    template <class dataType>
    void preprocessingPipeline(ftm::MergeTree<dataType> &mTree,
                               double epsilonTree,
                               double epsilon2Tree,
                               double epsilon3Tree,
                               bool branchDecompositionT,
                               bool useMinMaxPairT,
                               bool cleanTreeT,
                               std::vector<int> &nodeCorr,
                               bool deleteInconsistentNodes = true,
                               bool removeMergedSaddles = false,
                               int treeIdx = -1
                                ) {
      ftm::MergeTree<dataType> dummyTree;
      std::vector<ftm::idNode> localDataMap;                        
      preprocessingPipeline<dataType>(
        mTree, epsilonTree, epsilon2Tree, epsilon3Tree, branchDecompositionT,
        useMinMaxPairT, cleanTreeT, persistenceThreshold_, nodeCorr,localDataMap, dummyTree,
        deleteInconsistentNodes, removeMergedSaddles, treeIdx);
    }

    //No persistenceThreshold
    template <class dataType>
    void preprocessingPipeline(ftm::MergeTree<dataType> &mTree,
                              double epsilonTree,
                              double epsilon2Tree,
                              double epsilon3Tree,
                              bool branchDecompositionT,
                              bool useMinMaxPairT,
                              bool cleanTreeT,
                              std::vector<int> &nodeCorr,        // No double before this!
                              std::vector<ftm::idNode> &dataMap,
                              ftm::MergeTree<dataType> &preprocessed_MT,
                              bool deleteInconsistentNodes = true,
                              bool removeMergedSaddles = false) {
        preprocessingPipeline<dataType>(
            mTree, epsilonTree, epsilon2Tree, epsilon3Tree, branchDecompositionT,
            useMinMaxPairT, cleanTreeT, this->persistenceThreshold_, nodeCorr, dataMap, preprocessed_MT,
            deleteInconsistentNodes, removeMergedSaddles);
    }

    //No dataMap+PreprocessSave
    template <class dataType>
    void preprocessingPipeline(ftm::MergeTree<dataType> &mTree,
                              double epsilonTree,
                              double epsilon2Tree,
                              double epsilon3Tree,
                              bool branchDecompositionT,
                              bool useMinMaxPairT,
                              bool cleanTreeT,
                              double persistenceThreshold,
                              std::vector<int> &nodeCorr,        
                              bool deleteInconsistentNodes = true,
                              bool removeMergedSaddles = false) {
        std::vector<ftm::idNode> localDataMap;                        
        ftm::MergeTree<dataType> dummy;
        preprocessingPipeline<dataType>(
            mTree, epsilonTree, epsilon2Tree, epsilon3Tree, branchDecompositionT,
            useMinMaxPairT, cleanTreeT, persistenceThreshold, nodeCorr, localDataMap, dummy,
            deleteInconsistentNodes, removeMergedSaddles);
    }

    void reverseNodeCorr(ftm::FTMTree_MT *tree, std::vector<int> &nodeCorr) {
      std::vector<int> newNodeCorr(tree->getNumberOfNodes());
      for(unsigned int i = 0; i < nodeCorr.size(); ++i)
        if(nodeCorr[i] >= 0 && nodeCorr[i] < (int)newNodeCorr.size())
          newNodeCorr[nodeCorr[i]] = i;
      nodeCorr = newNodeCorr;
    }

    template <class dataType>
    void mtFlattening(ftm::MergeTree<dataType> &mt) {
      ftm::FTMTree_MT *tree = &(mt.tree);
      ttk::ftm::computePersistencePairs<dataType>(tree);
      persistenceThresholding<dataType>(tree);
      std::vector<std::vector<ftm::idNode>> treeNodeMerged;
      mergeSaddle<dataType>(tree, 100.0, treeNodeMerged);
      computeBranchDecomposition<dataType>(tree, treeNodeMerged);
    }

    template <class dataType>
    void mtsFlattening(std::vector<ftm::MergeTree<dataType>> &mts) {
      for(auto &mt : mts)
        mtFlattening(mt);
    }

    double getSizeLimitMetric(std::vector<ftm::FTMTree_MT *> &trees) {
      std::array<double, 3> stats;
      getTreesStats(trees, stats);
      auto meanNodes = stats[0];
      unsigned int const n = trees.size();
      return meanNodes * n;
    }

    // ------------------------------------------------------------------------
    // Tree Postprocessing
    // ------------------------------------------------------------------------
    template <class dataType>
    void copyMinMaxPair(ftm::MergeTree<dataType> &mTree1,
                        ftm::MergeTree<dataType> &mTree2,
                        bool setOrigins = false) {
      // Get min max pair
      ftm::FTMTree_MT *tree1 = &(mTree1.tree);
      ftm::idNode root = tree1->getRoot();
      dataType newMax = tree1->getValue<dataType>(root);
      ftm::idNode rootOrigin = tree1->getNode(root)->getOrigin();
      dataType newMin = tree1->getValue<dataType>(rootOrigin);

      // Update tree
      ftm::idNode root2 = mTree2.tree.getRoot();
      std::vector<dataType> newScalarsVector;
      ftm::getTreeScalars<dataType>(mTree2, newScalarsVector);
      newScalarsVector[root2] = newMax;

      auto root2Origin = mTree2.tree.getNode(root2)->getOrigin();
      if(root2Origin == (int)root2)
        root2Origin = mTree2.tree.template getMergedRootOrigin<dataType>();
      if(mTree2.tree.isNodeIdInconsistent(root2Origin))
        newScalarsVector.push_back(newMin);
      else
        newScalarsVector[root2Origin] = newMin;

      // Set new scalars
      ftm::setTreeScalars<dataType>(mTree2, newScalarsVector);

      // Create root origin if not already there
      ftm::FTMTree_MT *treeNew = &(mTree2.tree);
      if(mTree2.tree.isNodeIdInconsistent(root2Origin)) {
        root2Origin = treeNew->getNumberOfNodes();
        treeNew->makeNode(root2Origin);
      }

      // Manage new origins
      if(setOrigins) {
        treeNew->getNode(root2Origin)->setOrigin(root2);
        treeNew->getNode(root2)->setOrigin(root2Origin);
      }
    }

    template <class dataType>
    std::tuple<int, dataType> fixMergedRootOrigin(ftm::FTMTree_MT *tree) {
      if(not tree->isFullMerge())
        return std::make_tuple(-1, -1);

      // Get node of the min max pair
      int maxIndex = tree->getMergedRootOrigin<dataType>();

      // Link node of the min max pair with the root
      ftm::idNode const treeRoot = tree->getRoot();
      dataType oldOriginValue
        = tree->getValue<dataType>(tree->getNode(maxIndex)->getOrigin());
      tree->getNode(maxIndex)->setOrigin(treeRoot);

      return std::make_tuple(maxIndex, oldOriginValue);
    }

    // TODO fix bug when one multi pers. pairs is moved up with epsilon 2 and 3
    // but not its brothers
    template <class dataType>
    void branchDecompositionToTree(ftm::FTMTree_MT *tree) {
      ftm::idNode const treeRoot = tree->getRoot();

      // Get original tree message
      std::stringstream const oriPrintTree = tree->printTree();
      std::stringstream const oriPrintPairs
        = tree->printPairsFromTree<dataType>(true);
      std::stringstream const oriPrintMultiPers
        = tree->printMultiPersPairsFromTree<dataType>(true);

      // One pair case
      if(tree->isThereOnlyOnePersistencePair()) {
        ftm::idNode const treeRootOrigin = tree->getNode(treeRoot)->getOrigin();
        tree->getNode(treeRootOrigin)->setOrigin(treeRoot);
        return;
      }

      // Manage full merge and dontuseMinMaxPair_
      bool const isFM = tree->isFullMerge();
      if(isFM) {
        ftm::idNode const mergedRootOrigin
          = tree->getMergedRootOrigin<dataType>();
        if(not tree->isNodeIdInconsistent(mergedRootOrigin)
           and mergedRootOrigin != treeRoot)
          tree->getNode(treeRoot)->setOrigin(mergedRootOrigin);
        else {
          printErr("branchDecompositionToTree mergedRootOrigin inconsistent");
        }
      }

      // Some functions
      bool isJT = tree->isJoinTree<dataType>();
      auto comp = [&](const std::tuple<ftm::idNode, dataType> &a,
                      const std::tuple<ftm::idNode, dataType> &b) {
        return isJT ? std::get<1>(a) > std::get<1>(b)
                    : std::get<1>(a) < std::get<1>(b);
      };
      auto getIndexNotMultiPers = [&](int index, ftm::FTMTree_MT *treeT,
                                      std::vector<ftm::idNode> &children) {
        while(index >= 0 and treeT->isMultiPersPair(children[index]))
          --index;
        return index;
      };

      // Branch Decomposition To Tree
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> nodeParent;
      std::queue<ftm::idNode> queue;
      queue.emplace(treeRoot);
      while(!queue.empty()) {
        ftm::idNode node = queue.front();
        queue.pop();
        auto nodeOrigin = tree->getNode(node)->getOrigin();
        if(tree->isLeaf(node)) {
          if(tree->isNodeAlone(nodeOrigin)) {
            if(not isFM)
              nodeParent.emplace_back(nodeOrigin, node);
            else
              nodeParent.emplace_back(node, nodeOrigin);
          } else if(tree->isMultiPersPair(node)) {
            nodeParent.emplace_back(node, nodeOrigin);
          }
          continue;
        }

        // Get children and sort them by scalar values
        std::vector<ftm::idNode> childrenOri;
        tree->getChildren(node, childrenOri);
        std::vector<ftm::idNode> children = childrenOri;
        std::vector<std::tuple<ftm::idNode, dataType>> childrenScalars;
        for(unsigned int i = 0; i < children.size(); ++i) {
          if(isFM and (int) children[i] != nodeOrigin)
            children[i] = tree->getNode(children[i])->getOrigin();
          childrenScalars.push_back(std::make_tuple(
            children[i], tree->getValue<dataType>(children[i])));
        }
        std::sort(std::begin(childrenScalars), std::end(childrenScalars), comp);
        children.clear();
        for(unsigned int i = 0; i < childrenScalars.size(); ++i)
          children.push_back(std::get<0>(childrenScalars[i]));

        // Get new parent of children
        for(unsigned int i = 1; i < children.size(); ++i) {
          if(tree->isMultiPersPair(children[i]))
            continue;
          int const index = getIndexNotMultiPers(i - 1, tree, children);
          if(index >= 0)
            nodeParent.emplace_back(children[i], children[index]);
        }

        bool const multiPersPair
          = tree->getNode(nodeOrigin)->getOrigin() != (int)node;
        if(not multiPersPair) {
          if(not isFM) {
            int const index
              = getIndexNotMultiPers(children.size() - 1, tree, children);
            nodeParent.emplace_back(nodeOrigin, children[index]);
          } else
            nodeParent.emplace_back(children[0], node);
        } else {
          // std::cout << "branchDecompositionToTree multiPersPair" <<
          // std::endl;
          nodeParent.emplace_back(children[0], nodeOrigin);
          int index = getIndexNotMultiPers(children.size() - 1, tree, children);
          if(index < 0) { // should not be possible
            printErr("[branchDecompositionToTree] index < 0");
            index = 0;
          }
          nodeParent.emplace_back(node, children[index]);
        }

        // Push children to the queue
        for(auto child : childrenOri)
          queue.emplace(child);
      }

      // Set new parents for each node
      for(auto nodeParentT : nodeParent)
        tree->setParent(std::get<0>(nodeParentT), std::get<1>(nodeParentT));

      // Verify that the tree is correct
      for(unsigned int i = 0; i < tree->getNumberOfNodes(); ++i)
        if(tree->getNode(i)->getNumberOfDownSuperArcs() == 1
           and tree->getNode(i)->getNumberOfUpSuperArcs() == 1) {
          printMsg(oriPrintPairs.str());
          printMsg(oriPrintMultiPers.str());
          printMsg(oriPrintTree.str());
          printMsg(tree->printTree().str());
          std::stringstream ss;
          auto iOrigin = tree->getNode(i)->getOrigin();
          ss << i << " _ " << iOrigin;
          if(tree->getNode(iOrigin)->getOrigin() != int(i))
            ss << " _ " << tree->getNode(iOrigin)->getOrigin() << " _ "
               << tree->getNode(tree->getNode(iOrigin)->getOrigin())
                    ->getOrigin();
          printMsg(ss.str());
          printErr("[branchDecompositionToTree] 1 up arc and 1 down arc");
        }
    }

    // For not branch decomposition tree
    template <class dataType>
    void putBackMergedNodes(ftm::FTMTree_MT *tree) {
      bool isJT = tree->isJoinTree<dataType>();
      std::queue<ftm::idNode> queue;
      queue.emplace(tree->getRoot());
      while(!queue.empty()) {
        ftm::idNode const node = queue.front();
        queue.pop();
        ftm::idNode const nodeOrigin = tree->getNode(node)->getOrigin();
        if(!tree->isLeaf(node)) {
          std::vector<ftm::idNode> children;
          tree->getChildren(node, children);
          std::vector<ftm::idNode> lowestNodes;
          ftm::idNode branchOrigin = nodeOrigin;
          for(auto child : children) {
            ftm::idNode lowestNode = tree->getLowestNode<dataType>(child);
            lowestNodes.push_back(lowestNode);
            ftm::idNode const lowestNodeOrigin
              = tree->getNode(lowestNode)->getOrigin();
            if(not tree->isNodeAlone(lowestNodeOrigin)
               and lowestNodeOrigin != node)
              branchOrigin = lowestNode;
          }
          for(size_t i = 0; i < children.size(); ++i) {
            ftm::idNode lowestNodeOrigin
              = tree->getNode(lowestNodes[i])->getOrigin();
            if(branchOrigin == lowestNodes[i] or lowestNodeOrigin == node)
              continue;
            dataType lowestNodeOriginVal
              = tree->getValue<dataType>(lowestNodeOrigin);
            ftm::idNode branchOriginT = branchOrigin;
            ftm::idNode const branchRoot
              = tree->getNode(branchOrigin)->getOrigin();
            while(branchRoot != branchOriginT) {
              dataType val
                = tree->getValue<dataType>(tree->getParentSafe(branchOriginT));
              if((val > lowestNodeOriginVal and isJT)
                 or (val < lowestNodeOriginVal and not isJT))
                break;
              branchOriginT = tree->getParentSafe(branchOriginT);
            }
            tree->setParent(
              lowestNodeOrigin, tree->getParentSafe(branchOriginT));
            tree->setParent(branchOriginT, lowestNodeOrigin);
            tree->setParent(children[i], lowestNodeOrigin);
          }
        }
        std::vector<ftm::idNode> children;
        tree->getChildren(node, children);
        for(auto child : children)
          queue.emplace(child);
      }
    }

    template <class dataType>
    void postprocessingPipeline(ftm::FTMTree_MT *tree) {
      // if(not branchDecomposition_ or not useMinMaxPair)
      // fixMergedRootOrigin<dataType>(tree);
      if(tree->isFullMerge()) {
        auto mergedRootOrigin = tree->getMergedRootOrigin<dataType>();
        if(not tree->isNodeIdInconsistent(mergedRootOrigin))
          tree->getNode(tree->getRoot())->setOrigin(mergedRootOrigin);
        else
          printErr(
            "[postprocessingPipeline] mergedRootOrigin inconsistent id.");
      }
      if(branchDecomposition_) {
        if(not isPersistenceDiagram_ and tree->getRealNumberOfNodes() != 0)
          branchDecompositionToTree<dataType>(tree);
      } else
        putBackMergedNodes<dataType>(tree);
    }

    //Not fully implemented. Instead of converting, we use saved MT from preprocessing for now.
    template <class dataType>
    ftm::MergeTree<dataType> completeBDToMergeTree( ftm::FTMTree_MT *CBD1,
                                                    ftm::FTMTree_MT *CBD2,
                                                    std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>  &outputMatching,
                                                    std::vector<ftm::idNode> dataMap1,
                                                    std::vector<ftm::idNode> dataMap2
                                                    ) {
      //Matching corresponds to a subtree in the CBD, may not be full BDT        
      unsigned int numNodesCBD1 = dataMap1.size();
      unsigned int numNodesCBD2 = dataMap2.size();                                     
      std::vector<ftm::idNode> matchingBDT1;
      std::transform(outputMatching.begin(), outputMatching.end(), matchingBDT1.begin(),[](auto m) {
          return std::get<0>(m);
        }
      );
      std::remove_if(matchingBDT1.begin(), matchingBDT1.end(), [numNodesCBD1](auto id){return id >= numNodesCBD1;});
      size_t mT1Size = matchingBDT1.size(); // Branch consists of unique end points (*2), matching includes subtree nodes (/2)

      std::vector<ftm::idNode> matchingBDT2;
      std::transform(outputMatching.begin(), outputMatching.end(), matchingBDT2.begin(),[](auto m) {
          return std::get<1>(m);
        }
      );
      std::remove_if(matchingBDT2.begin(), matchingBDT2.end(), [numNodesCBD2](auto id){return id >= numNodesCBD2;});

      ftm::MergeTree<dataType> mT1 = ttk::ftm::createEmptyMergeTree<dataType>(mT1Size);
      return mT1;

    }

    // ------------------------------------------------------------------------
    // Output Matching
    // ------------------------------------------------------------------------
    template <class dataType>
    void convertCompleteBranchDecompositionMatching(std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &outputMatching,
                                                    std::vector<ftm::idNode> &dataMap1,
                                                    std::vector<ftm::idNode> &dataMap2,
                                                    ftm::FTMTree_MT *CBD1, 
                                                    ftm::FTMTree_MT *CBD2,
                                                    ftm::FTMTree_MT *MT1, 
                                                    ftm::FTMTree_MT *MT2
                                                    ) {

      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> toAdd;

      unsigned int numNodesCBD1 = dataMap1.size();
      unsigned int numNodesCBD2 = dataMap2.size();

      for(auto mTuple : outputMatching) {
        ftm::idNode node1 = std::get<0>(mTuple);
        ftm::idNode node2 = std::get<1>(mTuple);

        bool mapped1 = node1 < numNodesCBD1;
        bool mapped2 = node2 < numNodesCBD2;

        //Else assignment is irrelevant, but getNode should not be called for non mapped
        bool isSubtree1 = mapped1 ? CBD1->getNode(node1)->getIsSubtree() : true;
        bool isSubtree2 = mapped2 ? CBD2->getNode(node2)->getIsSubtree() : true;

        ftm::idNode  MTnode1;
        ftm::idNode MTnode2;

        if (mapped1) MTnode1 = dataMap1[node1];
        if (mapped2) MTnode2 = dataMap2[node2];

         
        double cost = std::get<2>(mTuple);
        if (mapped1 && mapped2 && !isSubtree1 && !isSubtree2) toAdd.emplace_back(MTnode1, MTnode2 ,cost);

        ftm::idNode  branchOrigin1;
        if (mapped1 && !isSubtree1) {
          branchOrigin1 = dataMap1[CBD1->getParent(node1)];
          MT1->getNode(branchOrigin1)->setOrigin(MTnode1);
          MT1->getNode(MTnode1)->setOrigin(branchOrigin1);
        }

        ftm::idNode  branchOrigin2;
        if (mapped2 && !isSubtree2) {
          branchOrigin2 = dataMap2[CBD2->getParent(node2)];
          MT2->getNode(branchOrigin2)->setOrigin(MTnode2);
          MT2->getNode(MTnode2)->setOrigin(branchOrigin2);
        }

        if (mapped1 && mapped2 && !isSubtree1 && !isSubtree2) toAdd.emplace_back(branchOrigin1, branchOrigin2 ,cost);
      }
      outputMatching.clear();
      outputMatching.insert(outputMatching.end(), toAdd.begin(), toAdd.end());

      if (cbdDebug) {
        std::cout << "\nMatching converted from CBD to Merge Tree with size "<< outputMatching.size()<< " : ";
        for (auto t : outputMatching) {
          std::cout << "(" <<std::to_string(std::get<0>(t)) <<";"<<std::to_string(std::get<1>(t)) <<"), ";
        }

        std::cout << "\ndataMap1 with size "<< dataMap1.size()<< " : ";
        int i=0;
        for (auto t : dataMap1) {
          std::cout << "(" <<i <<";"<< t <<"), ";
          i++;
        }

        std::cout << "\ndataMap2 with size "<< dataMap2.size()<< " : ";
        i=0;
        for (auto t : dataMap2) {
          std::cout << "(" <<i <<";"<< t <<"), ";
          i++;
        }
        /*
        std::cout << "\nOrigins1 with size "<< MT1->getNumberOfNodes()<< " : ";
        for(unsigned int nId = 0; nId < MT1->getNumberOfNodes(); ++nId){
          std::cout << "(" << nId <<";"<< MT1->getNode(nId)->getOrigin() <<"), ";
        }

        std::cout << "\nOrigins2 with size "<< MT2->getNumberOfNodes()<< " : ";
        for(unsigned int nId = 0; nId < MT2->getNumberOfNodes(); ++nId){
          std::cout << "(" << nId <<";"<< MT2->getNode(nId)->getOrigin() <<"), ";
        }
        */
      }
    }

    template <class dataType>
    void convertBranchDecompositionMatching(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
        &outputMatching) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> toAdd;
      for(auto mTuple : outputMatching) {
        ftm::idNode const node1 = std::get<0>(mTuple);
        ftm::idNode const node2 = std::get<1>(mTuple);
        double const cost = std::get<2>(mTuple);
        ftm::idNode const node1Origin = tree1->getNode(node1)->getOrigin();
        ftm::idNode const node2Origin = tree2->getNode(node2)->getOrigin();

        int const node1Level = tree1->getNodeLevel(node1);
        int const node1OriginLevel = tree1->getNodeLevel(node1Origin);
        int const node2Level = tree2->getNodeLevel(node2);
        int const node2OriginLevel = tree2->getNodeLevel(node2Origin);

        ftm::idNode const node1Higher
          = (node1Level > node1OriginLevel) ? node1 : node1Origin;
        ftm::idNode const node1Lower
          = (node1Level > node1OriginLevel) ? node1Origin : node1;
        ftm::idNode const node2Higher
          = (node2Level > node2OriginLevel) ? node2 : node2Origin;
        ftm::idNode const node2Lower
          = (node2Level > node2OriginLevel) ? node2Origin : node2;

        if(((tree1->isRoot(node1Higher) and tree1->isFullMerge())
            or (tree2->isRoot(node2Higher) and tree2->isFullMerge())))
          continue;

        if(!tree1->isNodeAlone(node1Higher)
           and !tree2->isNodeAlone(node2Higher))
          toAdd.emplace_back(node1Higher, node2Higher, cost);
        if(!tree1->isNodeAlone(node1Lower) and !tree2->isNodeAlone(node2Lower))
          toAdd.emplace_back(node1Lower, node2Lower, cost);
      }
      outputMatching.clear();
      outputMatching.insert(outputMatching.end(), toAdd.begin(), toAdd.end());
    }

    template <class dataType>
    void convertBranchDecompositionMatching(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &outputMatching) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
        realOutputMatching(outputMatching.size());
      for(size_t i = 0; i < outputMatching.size(); ++i) {
        const auto &tup{outputMatching[i]};
        realOutputMatching[i] = {std::get<0>(tup), std::get<1>(tup), 0.0};
      }

      convertBranchDecompositionMatching<dataType>(
        tree1, tree2, realOutputMatching);

      outputMatching.clear();
      for(auto tup : realOutputMatching)
        outputMatching.emplace_back(std::get<0>(tup), std::get<1>(tup));
    }

    template <class dataType>
    void identifyRealMatching(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &outputMatching,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, bool>> &realMatching) {
      for(std::tuple<ftm::idNode, ftm::idNode> mTuple : outputMatching) {
        ftm::idNode tree1Node = std::get<0>(mTuple);
        ftm::idNode tree2Node = std::get<1>(mTuple);
        dataType relabelCostVal
          = relabelCostOnly<dataType>(tree1, tree1Node, tree2, tree2Node);
        dataType deleteInsertCostVal = deleteCost<dataType>(tree1, tree1Node)
                                       + insertCost<dataType>(tree2, tree2Node);
        bool isRealMatching = (relabelCostVal <= deleteInsertCostVal);
        realMatching.emplace_back(tree1Node, tree2Node, isRealMatching);
      }
    }

    // ------------------------------------------------------------------------
    // Edit Costs
    // ------------------------------------------------------------------------
    template <class dataType>
    dataType computeDistance(
      dataType x1, dataType x2, dataType y1, dataType y2, double power = 2) {
      if(power <= 0)
        return std::max(
          std::abs((double)(x1 - y1)), std::abs((double)(x2 - y2)));
      else
        return std::pow(std::abs((double)(x1 - y1)), power)
               + std::pow(std::abs((double)(x2 - y2)), power);
    }

    template <class dataType>
    dataType deleteCost(ftm::FTMTree_MT *tree, ftm::idNode nodeId) {
      dataType cost = 0;
      dataType newMin = 0.0, newMax = 1.0;
      // Get birth/death
      auto birthDeath
        = normalizedWasserstein_
            ? getNormalizedBirthDeath<dataType>(tree, nodeId, newMin, newMax)
            : tree->getBirthDeath<dataType>(nodeId);
      dataType birth = std::get<0>(birthDeath);
      dataType death = std::get<1>(birthDeath);
      dataType projec = (birth + death) / 2;
      // Compute delete cost
      cost = computeDistance<dataType>(
        birth, death, projec, projec, wassersteinPower_);
      // Divide cost by two if not branch decomposition and not merged
      /*if(! branchDecomposition_ and ! tree->isNodeMerged(nodeId))
        cost /= 2;*/
      cost *= nonMatchingWeight_;

      return cost;
    }

    template <class dataType>
    dataType insertCost(ftm::FTMTree_MT *tree, ftm::idNode nodeId) {
      return deleteCost<dataType>(tree, nodeId);
    }

    template <class dataType>
    dataType relabelCostOnly(ftm::FTMTree_MT *tree1,
                             ftm::idNode nodeId1,
                             ftm::FTMTree_MT *tree2,
                             ftm::idNode nodeId2) {
      dataType cost = 0;
      dataType newMin = 0.0, newMax = 1.0;
      // Get birth/death of the first tree
      auto birthDeath1
        = normalizedWasserstein_
            ? getNormalizedBirthDeath<dataType>(tree1, nodeId1, newMin, newMax)
            : tree1->getBirthDeath<dataType>(nodeId1);
      dataType birth1 = std::get<0>(birthDeath1);
      dataType death1 = std::get<1>(birthDeath1);
      // Get birth/death of the second tree
      auto birthDeath2
        = normalizedWasserstein_
            ? getNormalizedBirthDeath<dataType>(tree2, nodeId2, newMin, newMax)
            : tree2->getBirthDeath<dataType>(nodeId2);
      dataType birth2 = std::get<0>(birthDeath2);
      dataType death2 = std::get<1>(birthDeath2);
      // Compute relabel cost
      cost = computeDistance<dataType>(
        birth1, death1, birth2, death2, wassersteinPower_);
      // Divide cost by two if not branch decomposition and not merged
      /*bool merged = isNodeMerged(tree1, nodeId1) or isNodeMerged(tree2,
      nodeId2); if(! branchDecomposition_ and ! merged) cost /= 2;*/

      return cost;
    }

    template <class dataType>
    dataType relabelCost(ftm::FTMTree_MT *tree1,
                         ftm::idNode nodeId1,
                         ftm::FTMTree_MT *tree2,
                         ftm::idNode nodeId2) {
      // Full merge case and only one persistence pair case
      if(tree1->getNode(nodeId1)->getOrigin() == (int)nodeId1
         or tree2->getNode(nodeId2)->getOrigin() == (int)nodeId2)
        return 0;

      // Compute relabel cost
      dataType cost = relabelCostOnly<dataType>(tree1, nodeId1, tree2, nodeId2);

      if(keepSubtree_) {
        // Compute deleteInsert cost
        dataType deleteInsertCost = deleteCost<dataType>(tree1, nodeId1)
                                    + insertCost<dataType>(tree2, nodeId2);
        if(deleteInsertCost < cost)
          cost = deleteInsertCost;
      }

      return cost;
    }

    // ------------------------------------------------------------------------
    // Utils
    // ------------------------------------------------------------------------
    void getParamNames(std::vector<std::string> &paramNames) {
      paramNames = std::vector<std::string>{"epsilon1",
                                            "epsilon2",
                                            "epsilon3",
                                            "persistenceThreshold",
                                            "branchDecomposition",
                                            "normalizedWasserstein",
                                            "keepSubtree",
                                            "isPersistenceDiagram",
                                            "deleteMultiPersPairs",
                                            "epsilon1UseFarthestSaddle",
                                            "mixtureCoefficient"};
    }

    double getParamValueFromName(std::string &paramName) {
      double value = 0.0;
      if(paramName == "epsilon1")
        value = epsilonTree1_;
      else if(paramName == "epsilon2")
        value = epsilon2Tree1_;
      else if(paramName == "epsilon3")
        value = epsilon3Tree1_;
      else if(paramName == "persistenceThreshold")
        value = persistenceThreshold_;
      else if(paramName == "branchDecomposition")
        value = branchDecomposition_;
      else if(paramName == "normalizedWasserstein")
        value = normalizedWasserstein_;
      else if(paramName == "keepSubtree")
        value = keepSubtree_;
      else if(paramName == "isPersistenceDiagram")
        value = isPersistenceDiagram_;
      else if(paramName == "deleteMultiPersPairs")
        value = deleteMultiPersPairs_;
      else if(paramName == "epsilon1UseFarthestSaddle")
        value = epsilon1UseFarthestSaddle_;
      else if(paramName == "mixtureCoefficient")
        value = mixtureCoefficient_;
      return value;
    }

    void setParamValueFromName(std::string &paramName, double value) {
      if(paramName == "epsilon1")
        epsilonTree1_ = value;
      else if(paramName == "epsilon2")
        epsilon2Tree1_ = value;
      else if(paramName == "epsilon3")
        epsilon3Tree1_ = value;
      else if(paramName == "persistenceThreshold")
        persistenceThreshold_ = value;
      else if(paramName == "branchDecomposition")
        branchDecomposition_ = value;
      else if(paramName == "normalizedWasserstein")
        normalizedWasserstein_ = value;
      else if(paramName == "keepSubtree")
        keepSubtree_ = value;
      else if(paramName == "isPersistenceDiagram")
        isPersistenceDiagram_ = value;
      else if(paramName == "deleteMultiPersPairs")
        deleteMultiPersPairs_ = value;
      else if(paramName == "epsilon1UseFarthestSaddle")
        epsilon1UseFarthestSaddle_ = value;
      else if(paramName == "mixtureCoefficient")
        mixtureCoefficient_ = value;
    }

    void getTreesStats(std::vector<ftm::FTMTree_MT *> &trees,
                       std::array<double, 3> &stats) {
      double avgNodes = 0, avgNodesT = 0;
      double avgDepth = 0;
      for(unsigned int i = 0; i < trees.size(); ++i) {
        auto noNodesT = trees[i]->getNumberOfNodes();
        auto noNodes = trees[i]->getRealNumberOfNodes();
        avgNodes += noNodes;
        avgNodesT += noNodesT;
        avgDepth += trees[i]->getTreeDepth();
      }
      avgNodes /= trees.size();
      avgNodesT /= trees.size();
      avgDepth /= trees.size();

      stats = {avgNodes, avgNodesT, avgDepth};
    }

    void printTreesStats(std::vector<ftm::FTMTree_MT *> &trees) {
      std::array<double, 3> stats;
      getTreesStats(trees, stats);
      int avgNodes = stats[0], avgNodesT = stats[1];
      double const avgDepth = stats[2];
      std::stringstream ss;
      ss << trees.size() << " trees average [node: " << avgNodes << " / "
         << avgNodesT << ", depth: " << avgDepth << "]";
      printMsg(ss.str(), debug::Priority::PERFORMANCE);
    }

    template <class dataType>
    void printTreesStats(std::vector<ftm::MergeTree<dataType>> &trees) {
      std::vector<ftm::FTMTree_MT *> treesT;
      ftm::mergeTreeToFTMTree<dataType>(trees, treesT);
      printTreesStats(treesT);
    }

    template <class dataType>
    void printTableVector(std::vector<std::vector<dataType>> &table) {
      std::streamsize const ssize = std::cout.precision();
      std::stringstream ss;
      ss << "      ";
      for(unsigned int j = 0; j < table[0].size(); ++j)
        ss << j - 1 << "    ";
      printMsg(ss.str());
      ss.str("");
      ss.clear();
      for(unsigned int i = 0; i < table.size(); ++i) {
        ss << std::setw(3) << std::setfill('0') << std::internal << i - 1
           << " | ";
        for(unsigned int j = 0; j < table[0].size(); ++j) {
          ss << std::fixed << std::setprecision(2) << table[i][j] << " ";
        }
        printMsg(ss.str());
        printMsg("");
      }
      std::cout.precision(ssize);
      printMsg(debug::Separator::L2);
    }

    template <class dataType>
    void printTable(dataType *table, int nRows, int nCols) {
      std::vector<std::vector<dataType>> vec(nRows, std::vector<dataType>());
      for(int i = 0; i < nRows; ++i)
        for(int j = 0; j < nCols; ++j)
          vec[i].push_back(table[i * nCols + j]);
      printTableVector<dataType>(vec);
    }

    void printMatching(std::vector<MatchingType> &matchings) {
      printMsg(debug::Separator::L2);
      for(const auto &mTuple : matchings) {
        std::stringstream ss;
        ss << std::get<0>(mTuple) << " - " << std::get<1>(mTuple) << " - "
           << std::get<2>(mTuple);
        printMsg(ss.str());
      }
      printMsg(debug::Separator::L2);
    }

    void printMatching(
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &matchings) {
      printMsg(debug::Separator::L2);
      for(auto mTuple : matchings) {
        std::stringstream ss;
        ss << std::get<0>(mTuple) << " - " << std::get<1>(mTuple);
        printMsg(ss.str());
      }
      printMsg(debug::Separator::L2);
    }

    void printMatching(
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &matchings) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> matchingsT(
        matchings.size());
      for(size_t i = 0; i < matchings.size(); ++i) {
        const auto &tup{matchings[i]};
        matchingsT[i] = {std::get<0>(tup), std::get<1>(tup), 0.0};
      }
      printMatching(matchingsT);
    }

    template <class dataType>
    void printPairs(
      std::vector<std::tuple<SimplexId, SimplexId, dataType>> &treePairs) {
      for(auto pair : treePairs) {
        std::stringstream const ss;
        ss << std::get<0>(pair) << " _ " << std::get<1>(pair) << " _ "
           << std::get<2>(pair);
        printMsg(ss.str());
      }
      printMsg(debug::Separator::L2);
    }

    template <class dataType>
    void printOutputMatching(
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &outputMatching,
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      bool computeCosts = true) {
      dataType cost = 0;
      std::vector<bool> tree1Done(tree1->getNumberOfNodes(), false);
      std::vector<bool> tree2Done(tree2->getNumberOfNodes(), false);
      std::stringstream ss;
      for(std::tuple<ftm::idNode, ftm::idNode> matching : outputMatching) {
        ftm::idNode node0 = std::get<0>(matching);
        ftm::idNode node1 = std::get<1>(matching);
        ftm::idNode node0Origin = tree1->getNode(node0)->getOrigin();
        ftm::idNode node1Origin = tree2->getNode(node1)->getOrigin();
        ss << node0 << " - " << node1 << " _ [ ";
        ss << "f(" << node0 << ")=" << tree1->getValue<dataType>(node0)
           << " _ ";
        ss << "g(" << node1 << ")=" << tree2->getValue<dataType>(node1) << " ]"
           << " _ [ ";
        ss << "f(" << node0Origin
           << ")=" << tree1->getValue<dataType>(node0Origin) << " _ ";
        ss << "g(" << node1Origin
           << ")=" << tree2->getValue<dataType>(node1Origin) << " ] ";

        if(computeCosts) {
          dataType tempCost = relabelCost<dataType>(tree1, node0, tree2, node1);
          dataType tempCost2
            = relabelCostOnly<dataType>(tree1, node0, tree2, node1);
          ss << "cost = " << tempCost << " (" << tempCost2 << ")" << std::endl;
          cost += tempCost;
        } else
          ss << std::endl;
        tree1Done[node0] = true;
        tree2Done[node1] = true;
      }

      for(unsigned int i = 0; i < tree1->getNumberOfNodes(); ++i)
        if(not tree1Done[i] and not tree1->isNodeAlone(i)) {
          ftm::idNode nodeOrigin = tree1->getNode(i)->getOrigin();
          ss << "T1 " << i << " _ [ f(" << i
             << ") = " << tree1->getValue<dataType>(i);
          ss << "_ f(" << nodeOrigin
             << ") = " << tree1->getValue<dataType>(nodeOrigin);
          ss << "]";
          if(computeCosts) {
            dataType tempCost = deleteCost<dataType>(tree1, i);
            ss << " _ cost = " << tempCost << std::endl;
            cost += tempCost;
          } else
            ss << std::endl;
        }
      for(unsigned int i = 0; i < tree2->getNumberOfNodes(); ++i)
        if(not tree2Done[i] and not tree2->isNodeAlone(i)) {
          ftm::idNode nodeOrigin = tree2->getNode(i)->getOrigin();
          ss << "T2 " << i << " _ [ g(" << i
             << ") = " << tree2->getValue<dataType>(i);
          ss << "_ g(" << nodeOrigin
             << ") = " << tree2->getValue<dataType>(nodeOrigin);
          ss << "]";
          if(computeCosts) {
            dataType tempCost = deleteCost<dataType>(tree2, i);
            ss << " _ cost = " << tempCost << std::endl;
            cost += tempCost;
          } else
            ss << std::endl;
        }
      if(computeCosts)
        ss << "total cost = " << cost << " (" << std::sqrt(cost) << ")"
           << std::endl;

      printMsg(ss.str());
      printMsg(debug::Separator::L2);
    }
  }; // MergeTreeBase class

} // namespace ttk
