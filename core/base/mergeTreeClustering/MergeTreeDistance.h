/// \ingroup base
/// \class MergeTreeDistance
/// \author Mathieu Pont (mathieu.pont@lip6.fr)
/// \date 2021.
///
/// This module defines the %MergeTreeDistance class that computes distance
/// between two merge trees.
///
/// \b Related \b publication \n
/// "Wasserstein Distances, Geodesics and Barycenters of Merge Trees" \n
/// Mathieu Pont, Jules Vidal, Julie Delon, Julien Tierny.\n
/// Proc. of IEEE VIS 2021.\n
/// IEEE Transactions on Visualization and Computer Graphics, 2021
///
/// \b Related \b publication \n
/// "Edit Distance between Merge Trees" \n
/// R. Sridharamurthy, T. B. Masood, A. Kamakshidasan and V. Natarajan. \n
/// IEEE Transactions on Visualization and Computer Graphics, 2020.

#pragma once

// ttk common includes
#include <Debug.h>

#include "MergeTreeBase.h"
#include <AssignmentAuction.h>
#include <AssignmentExhaustive.h>
#include <AssignmentMunkres.h>
#include <AssignmentSolver.h>

namespace ttk {

  /**
   * The MergeTreeDistance class provides methods to compute distance
   * between two merge trees.
   */
  class MergeTreeDistance : virtual public Debug, public MergeTreeBase {

  private:
    double t_assignment_time_ = 0;

    bool preprocess_ = true;
    
    bool saveTree_ = false;
    bool onlyEmptyTreeDistance_ = false;

    bool isCalled_ = false;

    double auctionEpsilon_ = -1;
    double auctionEpsilonDiviser_ = 0;
    int auctionRound_ = -1;

    double minMaxPairWeight_ = 1.0;

    //MA_mditz Debugging
    bool sortForestSolverInput = false;
    int nodesDone = 0;
    double itParentTime = 0;
    double computeTime = 0;
    /*
    int newTaskNumber = 0;
    std::vector<int> taskOfNode1;
    std::vector<int> taskOfNode2;
    */

    int numSinglePairing = 0;
    int sizeSinglePairing = 0;
    double timeSinglePairing =0;
    int maxdegree1SinglePairing = 0;
    int maxdegree2SinglePairing = 0;

    int numAssignment = 0;
    int sizeAssignment = 0;
    double timeAssignment = 0;
    int maxdegree1Assignment = 0;
    int maxdegree2Assignment = 0;
    // Just to get some stats about run
    // std::map<int, int> assignmentProblemSize, assignmentProblemIter;

    bool testing_ = true;
    

    // Parallel version data
    std::vector<std::vector<ftm::idNode>> tree2LevelToNode_;
    std::vector<int> tree1Level_, tree2Level_;
    //MA_mditz
    std::vector<std::tuple<int,int>> tree1Range_, tree2Range_; 

  public:
    MergeTreeDistance() {
      this->setDebugMsgPrefix(
        "MergeTreeDistance"); // inherited from Debug: prefix will be printed at
                              // the beginning of every msg
#ifdef TTK_ENABLE_OPENMP4
      omp_set_nested(1);
#endif
    }
    ~MergeTreeDistance() override = default;

    void setIsCalled(bool ic) {
      isCalled_ = ic;
    }

    void setPreprocess(bool preproc) {
      preprocess_ = preproc;
    }

    void setTesting(bool test) {
      testing_ = test;
    }

    void setSaveTree(bool save) {
      saveTree_ = save;
    }

    void setAuctionEpsilon(double aucEpsilon) {
      auctionEpsilon_ = aucEpsilon;
    }

    void setAuctionEpsilonDiviser(double aucEpsilonDiviser) {
      auctionEpsilonDiviser_ = aucEpsilonDiviser;
    }

    void setAuctionNoRounds(double aucNoRounds) {
      auctionRound_ = aucNoRounds;
    }

    void setOnlyEmptyTreeDistance(double only) {
      onlyEmptyTreeDistance_ = only;
    }

    void setMinMaxPairWeight(double weight) {
      minMaxPairWeight_ = weight;
    }

    /**
     * Implementation of the algorithm.
     */

    // ------------------------------------------------------------------------
    // Assignment Problem
    // ------------------------------------------------------------------------
    template <class dataType>
    void runAssignmentProblemSolver(std::vector<std::vector<dataType>> &costMatrix,
                                 std::vector<MatchingType> &matchings) {
      AssignmentSolver<dataType> *assignmentSolver;
      AssignmentExhaustive<dataType> solverExhaustive;
      AssignmentMunkres<dataType> solverMunkres;
      AssignmentAuction<dataType> solverAuction;

      int const nRows = costMatrix.size() - 1;
      int const nCols = costMatrix[0].size() - 1;
      int const max_dim = std::max(nRows, nCols);
      int const min_dim = std::min(nRows, nCols);

      int assignmentSolverID = assignmentSolverID_;
      if((min_dim <= 2 and max_dim <= 2) or (min_dim <= 1 and max_dim <= 6))
        assignmentSolverID = 1;

      switch(assignmentSolverID) {
        case 1:
          solverExhaustive = AssignmentExhaustive<dataType>();
          assignmentSolver = &solverExhaustive;
          break;
        case 2:
          solverMunkres = AssignmentMunkres<dataType>();
          assignmentSolver = &solverMunkres;
          break;
        case 0:
        default:
          solverAuction = AssignmentAuction<dataType>();
          solverAuction.setEpsilon(auctionEpsilon_);
          solverAuction.setEpsilonDiviserMultiplier(auctionEpsilonDiviser_);
          solverAuction.setNumberOfRounds(auctionRound_);
          assignmentSolver = &solverAuction;
      }
      assignmentSolver->setInput(costMatrix);
      assignmentSolver->setBalanced(false);
      assignmentSolver->run(matchings);
    }

    template <class dataType>
    void createCostMatrix(std::vector<std::vector<dataType>> &treeTable,
                          std::vector<ftm::idNode> &children1,
                          std::vector<ftm::idNode> &children2,
                          std::vector<std::vector<dataType>> &costMatrix) {
      unsigned int nRows = children1.size(), nCols = children2.size();
      for(unsigned int i = 0; i < nRows; ++i) {
        int const forestTableI = children1[i] + 1;
        for(unsigned int j = 0; j < nCols; ++j) {
          int const forestTableJ = children2[j] + 1;
          // Cost of assigning i and j
          costMatrix[i][j] = treeTable[forestTableI][forestTableJ];
          //MA_mditz Sollte angepasst werden
          /*
          if(not checkEntry(children1[i],children2[j],tree1,tree2)
             and not keepSubtree_ and not MA_mditz)
            printErr("different levels!"); // should be impossible
          */
        }
          
        // Cost of not assigning i
        costMatrix[i][nCols] = treeTable[forestTableI][0];
      }
      for(unsigned int j = 0; j < nCols; ++j) {
        int const forestTableJ = children2[j] + 1;
        // Cost of not assigning j
        costMatrix[nRows][j] = treeTable[0][forestTableJ];
      }
      costMatrix[nRows][nCols] = 0;
    }

    template <class dataType>
    dataType postprocessAssignment(
      std::vector<MatchingType> &matchings,
      std::vector<ftm::idNode> &children1,
      std::vector<ftm::idNode> &children2,
      std::vector<std::tuple<int, int>> &forestAssignment) {
      dataType cost = 0;
      for(const auto &mTuple : matchings) {
        cost += std::get<2>(mTuple);
        /* Before mditz
        if(std::get<0>(mTuple) >= (int)children1.size()
           || std::get<1>(mTuple) >= (int)children2.size() )
          continue;
        int const tableId1 = children1[std::get<0>(mTuple)] + 1;
        int const tableId2 = children2[std::get<1>(mTuple)] + 1;
        */

        bool erased1 = std::get<0>(mTuple) >= (int)children1.size();
        bool erased2 = std::get<1>(mTuple) >= (int)children2.size();
        if (erased1 || erased2 ) {
          if (MA_mditz) {
            int const tableId1 = erased1? 0 : children1[std::get<0>(mTuple)] + 1;
            int const tableId2 = erased2? 0 : children2[std::get<1>(mTuple)] + 1;

            forestAssignment.emplace_back(tableId1, tableId2);
          }
          continue;
        }
        int const tableId1 = children1[std::get<0>(mTuple)] + 1;
        int const tableId2 = children2[std::get<1>(mTuple)] + 1;

        forestAssignment.emplace_back(tableId1, tableId2);
      }
      return cost;
    }

    template <class dataType>
    dataType forestAssignmentProblem(
      ftm::FTMTree_MT *ttkNotUsed(tree1),
      ftm::FTMTree_MT *ttkNotUsed(tree2),
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<ftm::idNode> &children1,
      std::vector<ftm::idNode> &children2,
      std::vector<std::tuple<int, int>> &forestAssignment) {
      // --- Create cost matrix
      int nRows = children1.size(), nCols = children2.size();
      std::vector<std::vector<dataType>> costMatrix(
        nRows + 1, std::vector<dataType>(nCols + 1));
      createCostMatrix(treeTable, children1, children2, costMatrix);

      // assignmentProblemSize[costMatrix.size()*costMatrix[0].size()]++;

      // --- Solve assignment problem
      std::vector<MatchingType> matchings;
      runAssignmentProblemSolver(costMatrix, matchings);

      // --- Postprocess matching to create output assignment
      dataType cost = postprocessAssignment<dataType>(
        matchings, children1, children2, forestAssignment);

      return cost;
    }

    template <class dataType>
    void computeForestsDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      int i,
      int j,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      std::vector<ftm::idNode> &children1,
      std::vector<ftm::idNode> &children2) {

      //In case of CBD computation and both subtree nodes, compute all choice pairs
      if (MA_mditz and (tree1->getNode(i-1)->getIsSubtree() and tree2->getNode(j-1)->getIsSubtree())){

        std::tuple<dataType, ftm::idNode, ftm::idNode> term_pairChoices = computeTermSinglePairing_MA_mditz<dataType>(children1,children2,treeTable);
        forestTable[i][j] = std::get<0>(term_pairChoices);
        if(postprocess_)
          forestBackTable[i][j] = {std::make_tuple(std::get<1>(term_pairChoices), std::get<2>(term_pairChoices))};

        return;
      } 
        
        
      if(children1.size() != 0 && children2.size() != 0) {
        dataType forestTerm3;

        // Term 3
        Timer t_assignment;
        std::vector<std::tuple<int, int>> forestAssignment;
        
        if (sortForestSolverInput) {
          std::sort(children1.begin(), children1.end(), [tree1](auto &left, auto &right) {
            return tree1->getValue<dataType>(left) < tree1->getValue<dataType>(right);
          });
          std::sort(children2.begin(), children2.end(), [tree2](auto &left, auto &right) {
            return tree2->getValue<dataType>(left) < tree2->getValue<dataType>(right);
          });
        }

        forestTerm3 = forestAssignmentProblem<dataType>(
          tree1, tree2, treeTable, children1, children2, forestAssignment);
        
        
        if(not parallelize_ or (statsTest and omp_get_num_threads() == 1)){
          std::cout << "[StatsTest] AssInst: " << children1.size()*children2.size() << "\n"; 
          t_assignment_time_ += t_assignment.getElapsedTime();
          sizeAssignment += children1.size()*children2.size();
          numAssignment += 1;
          maxdegree1Assignment = std::max<int>(maxdegree1Assignment,children1.size());
          maxdegree2Assignment = std::max<int>(maxdegree2Assignment,children2.size());
          
        }
        if(not keepSubtree_) {
          // Compute table value
          forestTable[i][j] = forestTerm3;
          // Add backtracking information
          
          /*
          std::cout <<"\n" << i << ", " << j << " hat forestTerm3 mit cost " << forestTerm3 << " und Assignment: ";
          for(auto a : forestAssignment)
            std::cout << "(" << std::get<0>(a) <<", "<< std::get<1>(a) << "), ";
          */
          
          
          if (postprocess_) 
            forestBackTable[i][j] = forestAssignment;
          
        } else {
          dataType forestTerm1, forestTerm2;
          std::tuple<dataType, ftm::idNode> forestCoTerm1, forestCoTerm2;

          // Term 1
          forestCoTerm1
            = computeTerm1_2<dataType>(children2, i, forestTable, true);
          forestTerm1 = forestTable[0][j] + std::get<0>(forestCoTerm1);

          // Term2
          forestCoTerm2
            = computeTerm1_2<dataType>(children1, j, forestTable, false);
          forestTerm2 = forestTable[i][0] + std::get<0>(forestCoTerm2);

          // Compute table value
          forestTable[i][j]
            = std::min(std::min(forestTerm1, forestTerm2), forestTerm3);
          if (!postprocess_) {
            return;
          }
          // Add backtracking information
          if(forestTable[i][j] == forestTerm3) {
            forestBackTable[i][j] = forestAssignment;
          } else if(forestTable[i][j] == forestTerm2) {
            forestBackTable[i][j].push_back(
              std::make_tuple(std::get<1>(forestCoTerm2), j));
          } else {
            forestBackTable[i][j].push_back(
              std::make_tuple(i, std::get<1>(forestCoTerm1)));
          }
        }
      } else {
        // If one of the forest is empty we get back to equation 8 or 10
        forestTable[i][j] = (children1.size() == 0) ? forestTable[0][j] : forestTable[i][0];
      }
    }

    // ------------------------------------------------------------------------
    // Edit Distance Dynamic Programming Equations
    // ------------------------------------------------------------------------
    template <class dataType>
    void computeForestToEmptyDistance(
      ftm::FTMTree_MT *tree1,
      ftm::idNode nodeI,
      int i,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable, 
      std::vector<std::vector<std::vector<std::tuple<int, int>>>> &forestBackTable) {

      std::vector<std::tuple<int, int>> backTrack; 
      bool cbdSubtree = MA_mditz and tree1->getNode(nodeI)->getIsSubtree();
      std::vector<ftm::idNode> children;
      tree1->getChildren(nodeI, children);
      
      ftm::idNode cbdSubtreeMin; 
      if(cbdSubtree){
        cbdSubtreeMin = children[0]+1;
        forestTable[i][0] = treeTable[cbdSubtreeMin][0];
      }
      else
        forestTable[i][0] = 0;

      
      for(ftm::idNode const child : children){
        if(cbdSubtree){
          if (forestTable[i][0] > treeTable[child + 1][0]) {
            forestTable[i][0] =  treeTable[child + 1][0];
            cbdSubtreeMin = child + 1;
          }  
        }
        else{
          forestTable[i][0] += treeTable[child + 1][0];
        }
      }

      if (cbdSubtree) {
        backTrack.emplace_back(cbdSubtreeMin, 0);
      }
      if (postprocess_) 
        forestBackTable[i][0] = backTrack;
    }

    template <class dataType>
    void computeSubtreeToEmptyDistance(
      ftm::FTMTree_MT *tree1,
      ftm::idNode nodeI,
      int i,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable) {
      if(MA_mditz and tree1->getNode(nodeI)->getIsSubtree())
        treeTable[i][0] = forestTable[i][0];
      else
        treeTable[i][0] = forestTable[i][0] + deleteCost<dataType>(tree1, nodeI);
    }

    template <class dataType>
    void computeEmptyToForestDistance(
      ftm::FTMTree_MT *tree2,
      ftm::idNode nodeJ,
      int j,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable, 
      std::vector<std::vector<std::vector<std::tuple<int, int>>>> &forestBackTable) {

      std::vector<std::tuple<int, int>> backTrack; 
      bool cbdSubtree = MA_mditz and tree2->getNode(nodeJ)->getIsSubtree();
      std::vector<ftm::idNode> children;
      tree2->getChildren(nodeJ, children);
      
      ftm::idNode cbdSubtreeMin;
      if(cbdSubtree){
        cbdSubtreeMin = children[0]+1;
        forestTable[0][j] = treeTable[0][cbdSubtreeMin];
      }  else
        forestTable[0][j] = 0;

      for(ftm::idNode const child : children)
        if(cbdSubtree){
          if (forestTable[0][j] > treeTable[0][child + 1]) {
            forestTable[0][j] =  treeTable[0][child + 1];
            cbdSubtreeMin = child + 1;
          }
        }
        else{
          forestTable[0][j] += treeTable[0][child + 1];
        }
      if (cbdSubtree) {
        backTrack.emplace_back(0, cbdSubtreeMin);
      }
      if (postprocess_) 
        forestBackTable[0][j] = backTrack;
    }

    template <class dataType>
    void computeEmptyToSubtreeDistance(
      ftm::FTMTree_MT *tree2,
      ftm::idNode nodeJ,
      int j,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable) {
      if(MA_mditz and tree2->getNode(nodeJ)->getIsSubtree())
        treeTable[0][j] = forestTable[0][j];
      else
        treeTable[0][j] = forestTable[0][j] + insertCost<dataType>(tree2, nodeJ);
    }

    // Compute first or second term of forests and subtrees distance
    template <class dataType>
    std::tuple<dataType, ftm::idNode>
      computeTerm1_2(std::vector<ftm::idNode> &childrens,
                     int ind,
                     std::vector<std::vector<dataType>> &table,
                     bool computeTerm1) {
      dataType tempMin = (childrens.size() == 0)
                           ? ((computeTerm1) ? table[ind][0] : table[0][ind])
                           : std::numeric_limits<dataType>::max();
      ftm::idNode bestIdNode = 0;
      for(ftm::idNode children : childrens) {
        children += 1;
        dataType temp;
        if(computeTerm1) {
          temp = table[ind][children] - table[0][children];
        } else {
          temp = table[children][ind] - table[children][0];
        }
        if(temp < tempMin) {
          tempMin = temp;
          bestIdNode = children;
        }
      }
      return std::make_tuple(tempMin, bestIdNode);
    }

    //MA_mditz
    template <class dataType>
    std::tuple<dataType, ftm::idNode, ftm::idNode>
      computeTermSinglePairing_MA_mditz(  std::vector<ftm::idNode> &children1,
                                    std::vector<ftm::idNode> &children2,
                                    std::vector<std::vector<dataType>> &treeTable){
      if(not MA_mditz)
        this->printErr("Computing Single Pairing Term while MA_mditz is false");
      dataType tempMin = std::numeric_limits<dataType>::max();
      ftm::idNode bestIdNode1 = 0;
      ftm::idNode bestIdNode2 = 0;
      for(ftm::idNode child1 : children1) {
        for(ftm::idNode child2 : children2) {
          dataType temp = treeTable[child1 + 1][child2 +1];
          if(temp < tempMin) {
            tempMin = temp;
            bestIdNode1 = child1;
            bestIdNode2 = child2;
          }
        }
      }
      return std::make_tuple(tempMin, bestIdNode1 +1, bestIdNode2 +1);
    }

    template <class dataType>
    void computeSubtreesDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      int i,
      int j,
      ftm::idNode nodeI,
      ftm::idNode nodeJ,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<ftm::idNode> &children1,
      std::vector<ftm::idNode> &children2) {
      bool subtreesCBD = MA_mditz and (tree1->getNode(nodeI)->getIsSubtree() and tree2->getNode(nodeJ)->getIsSubtree());

      dataType treeTerm3;
      // Term 3
      treeTerm3  = forestTable[i][j] + (subtreesCBD ? 0 : relabelCost<dataType>(tree1, nodeI, tree2, nodeJ));
      

      if(not keepSubtree_) {
        // Compute table value
        treeTable[i][j] = treeTerm3;
        // Add backtracking information
        if (postprocess_) 
          treeBackTable[i][j] = std::make_tuple(i, j);
        
      } else {
        dataType treeTerm1, treeTerm2;
        std::tuple<dataType, ftm::idNode> treeCoTerm1, treeCoTerm2;

        // Term 1
        treeCoTerm1 = computeTerm1_2<dataType>(children2, i, treeTable, true);
        treeTerm1 = treeTable[0][j] + std::get<0>(treeCoTerm1);

        // Term 2
        treeCoTerm2 = computeTerm1_2<dataType>(children1, j, treeTable, false);
        treeTerm2 = treeTable[i][0] + std::get<0>(treeCoTerm2);

        // Compute table value
        treeTable[i][j] = std::min(std::min(treeTerm1, treeTerm2), treeTerm3);

        if (!postprocess_) {
          return;
        }
        // Add backtracking information
        if(treeTable[i][j] == treeTerm3 ) {
          treeBackTable[i][j] = std::make_tuple(i, j);
        } else if(treeTable[i][j] == treeTerm2) {
          treeBackTable[i][j] = std::make_tuple(std::get<1>(treeCoTerm2), j);
        } else {
          treeBackTable[i][j] = std::make_tuple(i, std::get<1>(treeCoTerm1));
        }
      }
    }

    // --------------------------------------------------------------------------------
    // Output Matching
    // --------------------------------------------------------------------------------
    template <class dataType>
    void computeMatching(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>> &outputMatching,
      int startR,
      int startC) {
      unsigned int deletionIDTree1 = -1; //2*tree1->getNumberOfNodes();
      unsigned int deletionIDTree2 = -1; // 2*tree2->getNumberOfNodes();

      outputMatching.clear();
      std::queue<std::tuple<int, int, bool, bool>> backQueue;
      //Additional for deletion, cbd
      backQueue.emplace(startR, startC, true, false);
      while(!backQueue.empty()) {
        std::tuple<int, int, bool, bool> elem = backQueue.front();
        backQueue.pop();
        int const i = std::get<0>(elem);
        int const j = std::get<1>(elem);
        bool useTreeTable = std::get<2>(elem);
        bool deletionSubtree = std::get<3>(elem);

        if(useTreeTable) {
          int const tupleI = std::get<0>(treeBackTable[i][j]);
          int const tupleJ = std::get<1>(treeBackTable[i][j]);
          if(tupleI != 0 && tupleJ != 0) {
            useTreeTable = (tupleI != i || tupleJ != j);
            backQueue.emplace(tupleI, tupleJ, useTreeTable, false);
            if(not useTreeTable) { // We have matched i and j
              ftm::idNode const tree1Node = tupleI - 1;
              ftm::idNode const tree2Node = tupleJ - 1;
              double cost = 0;
              dataType costT
                = relabelCost<dataType>(tree1, tree1Node, tree2, tree2Node);
              cost = static_cast<double>(costT);
              outputMatching.emplace_back(tree1Node, tree2Node, cost);
            }
          } 
        } else if (deletionSubtree) { //For Wasserstein on CBDs s.t. we can compute optimum BDT including deletion cases
          const bool is_i_zero = (i == 0);

          auto* primaryTree   = is_i_zero ? tree2 : tree1;
          auto  primaryIdx    = is_i_zero ? j - 1 : i - 1;
          auto  deleteNumTree = is_i_zero ? deletionIDTree1 : deletionIDTree2;
          if (is_i_zero) {
              outputMatching.emplace_back(deleteNumTree, primaryIdx, 0);
          } else {
              outputMatching.emplace_back(primaryIdx, deleteNumTree, 0);
          }

          if (primaryTree->getNode(primaryIdx)->getIsSubtree()) {
              auto tableVal = is_i_zero ? std::get<1>(forestBackTable[i][j][0]) 
                                             : std::get<0>(forestBackTable[i][j][0]);
              backQueue.emplace(is_i_zero ? 0 : tableVal, is_i_zero ? tableVal : 0, false, true);
          } else {
              std::vector<ftm::idNode> children;
              primaryTree->getChildren(primaryIdx, children);
              for (ftm::idNode c : children) {
                  backQueue.emplace(is_i_zero ? 0 : c +1, is_i_zero ? c +1 : 0, false, true);
              }
          }
        } else {
          for(std::tuple<int, int> forestBackElem : forestBackTable[i][j]) {
            int const tupleI = std::get<0>(forestBackElem);
            int const tupleJ = std::get<1>(forestBackElem);
            if(tupleI != 0 && tupleJ != 0) {
              useTreeTable = (tupleI != i && tupleJ != j);
              backQueue.emplace(tupleI, tupleJ, useTreeTable, false);
              //std::cout << "---- in not useTreetable emplaced: " << tupleI <<", " <<tupleJ << " with useTreeTable " << useTreeTable << "\n";
            } else if (MA_mditz) {
              outputMatching.emplace_back(tupleI == 0? deletionIDTree1 : tupleI, 
                                          tupleJ == 0? deletionIDTree2 : tupleJ, 
                                          0);
              backQueue.emplace(tupleI, tupleJ, false, true);
            } 
          }
        }
      }
    }

    // ------------------------------------------------------------------------
    // Main Functions
    // ------------------------------------------------------------------------
    template <class dataType>
    dataType
      computeDistance(ftm::FTMTree_MT *tree1,
                      ftm::FTMTree_MT *tree2,
                      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
                        &outputMatching) {
      // ---------------------
      // ----- Init dynamic programming tables
      // --------------------
      size_t const nRows = tree1->getNumberOfNodes() + 1;
      size_t const nCols = tree2->getNumberOfNodes() + 1;
      std::vector<std::vector<dataType>> treeTable(
        nRows, std::vector<dataType>(nCols));
      std::vector<std::vector<dataType>> forestTable(
        nRows, std::vector<dataType>(nCols));

      std::vector<std::vector<std::tuple<int, int>>> treeBackTable;
      std::vector<std::vector<std::vector<std::tuple<int, int>>>> forestBackTable;

      if (postprocess_) {
        // Backtracking tables (output matching)
        treeBackTable = std::vector<std::vector<std::tuple<int, int>>>(nRows, std::vector<std::tuple<int, int>>(nCols));
        forestBackTable = std::vector<std::vector<std::vector<std::tuple<int, int>>>>(nRows, std::vector<std::vector<std::tuple<int, int>>>(nCols));
      }
       
      int const indR = tree1->getRoot() + 1;
      int const indC = tree2->getRoot() + 1;
      if(MA_mditz and acceleration_){
        tree1->getAllNodeRangeLevel(tree1Range_);
        tree2->getAllNodeRangeLevel(tree2Range_);
      }
      if (!MA_mditz) {
        tree1->getAllNodeLevel(tree1Level_);
        tree2->getAllNodeLevel(tree2Level_);
        tree2->getLevelToNode(tree2LevelToNode_);
      }
      
      // ---------------------
      // ----- Compute edit distance
      // --------------------
      Timer EDtime;
      computeEditDistance(tree1, tree2, treeTable, forestTable, treeBackTable,
                          forestBackTable, nRows, nCols);
      
      dataType distance = treeTable[indR][indC];

      if(onlyEmptyTreeDistance_)
        distance = treeTable[indR][0];
      if(branchDecomposition_) {
        if(not useMinMaxPair_) {
          if(onlyEmptyTreeDistance_)
            distance -= deleteCost<dataType>(tree1, tree1->getRoot());
          else
            distance -= relabelCost<dataType>(
              tree1, tree1->getRoot(), tree2, tree2->getRoot());
        } else {
          if(minMaxPairWeight_ != 1.0) {
            auto cost = relabelCost<dataType>(
              tree1, tree1->getRoot(), tree2, tree2->getRoot());
            distance = distance - cost + minMaxPairWeight_ * cost;
          }
        }
      }
      /*
      std::cout <<"Tree and Forest Table:\n";
      for(unsigned int i = 0; i < nRows; i++){
        for(unsigned int j = 0; j < nCols; j++){
          std::cout << "("<<i <<";" << j <<"; S: "<< treeTable[i][j]<<"; F: " << forestTable[i][j]<<") "; 
        }
        std::cout << "\n";
      }
      */
      if(distanceSquaredRoot_)
        distance = std::sqrt(distance);
      
      // ---------------------
      // ----- Compute matching
      // --------------------
      
      if (postprocess_)
        computeMatching<dataType>(tree1, tree2, treeBackTable, forestBackTable,
                              outputMatching, indR, indC);

      
      return distance;
    }

    template <class dataType>
    dataType computeDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &outputMatching) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
        realOutputMatching;
      
      dataType res
        = computeDistance<dataType>(tree1, tree2, realOutputMatching);
      
      for(auto tup : realOutputMatching)
        outputMatching.emplace_back(std::get<0>(tup), std::get<1>(tup));
      
      return res;
    }

    template <class dataType>
    dataType execute(ftm::MergeTree<dataType> &mTree1,
                     ftm::MergeTree<dataType> &mTree2,
                     std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
                       &outputMatching, int tree1Idx = -1, int tree2Idx = -1) {

      if (cbdDebug) {
        std::cout << "========================================\n"
          << "        MergeTree Distance Configuration         \n"
          << "========================================\n"
          << std::boolalpha // Prints 'true'/'false' instead of '1'/'0'
          << "MA_mditz:          " << MA_mditz << "\n"
          << "acceleration_:     " << acceleration_ << "\n"
          << "parallelFor:       " << parallelFor << "\n"
          << "statsTest:         " << statsTest << "\n"
          << "useThresholdCBD_:  " << useThresholdCBD_ << "\n"
          << "globalThreshold_:  " << globalThreshold_ << "\n"
          << "thresholdOfCBD_:   " << thresholdOfCBD_ << "\n"
          << "branchDecomposition_:   " << branchDecomposition_ << "\n"
          << "========================================\n";
      }
      

      Memory m;
      Timer t_total;

      // ---------------------
      // ----- Testing
      // --------------------
      testing_ = false;

      // ---------------------
      // ----- Preprocessing
      // --------------------
      std::vector<ftm::idNode> dataMap1; 
      std::vector<ftm::idNode> dataMap2;

      ftm::MergeTree<dataType> mTree1Copy;
      ftm::MergeTree<dataType> mTree2Copy;
      if(true){//saveTree_) {
        mTree1Copy = ftm::copyMergeTree<dataType>(mTree1);
        mTree2Copy = ftm::copyMergeTree<dataType>(mTree2);
      }
      ftm::MergeTree<dataType> &mTree1Int = (saveTree_ ? mTree1Copy : mTree1);
      ftm::MergeTree<dataType> &mTree2Int = (saveTree_ ? mTree2Copy : mTree2);
      ftm::FTMTree_MT *tree1 = &(mTree1Int.tree);
      ftm::FTMTree_MT *tree2 = &(mTree2Int.tree);
      if(not isCalled_ and not isPersistenceDiagram_ and not MA_mditz) {
        verifyMergeTreeStructure<dataType>(tree1);
        verifyMergeTreeStructure<dataType>(tree2);
      }

      ftm::MergeTree<dataType> preprocessed_MT1_copy;
      ftm::MergeTree<dataType> preprocessed_MT2_copy;

      //mditz:This is only entered with MergeTreeClustering
      if(preprocess_) {
        treesNodeCorr_.resize(2);
        preprocessingPipeline<dataType>(
          mTree1Int, epsilonTree1_, epsilon2Tree1_, epsilon3Tree1_,
          branchDecomposition_, useMinMaxPair_, cleanTree_, treesNodeCorr_[0], dataMap1, preprocessed_MT1_copy, true, false);
        preprocessingPipeline<dataType>(
          mTree2Int, epsilonTree2_, epsilon2Tree2_, epsilon3Tree2_,
          branchDecomposition_, useMinMaxPair_, cleanTree_, treesNodeCorr_[1], dataMap2, preprocessed_MT2_copy,true, false);
      }
      tree1 = &(mTree1Int.tree);
      tree2 = &(mTree2Int.tree);

      if (cbdDebug) {
        std::cout << "\nAfter preprocessing dataMap1 with size "<< dataMap1.size()<< " : ";
        int i=0;
        for (auto t : dataMap1) {
          std::cout << "(" <<i <<";"<< t <<"), ";
          i++;
        }

        std::cout << "\nAfter preprocessing dataMap2 with size "<< dataMap2.size()<< " : ";
        i=0;
        for (auto t : dataMap2) {
          std::cout << "(" <<i <<";"<< t <<"), ";
          i++;
        }

      }
      if(tree1Idx != -1)
        std::cout <<"("<< tree1Idx << "," << tree2Idx <<") starts computeDistance "<<std::endl;
      // ---------------------
      // ----- Compute Distance
      // --------------------
      dataType distance
        = computeDistance<dataType>(tree1, tree2, outputMatching);
      std::cout <<"("<< tree1Idx << "," << tree2Idx <<") ends computeDistance "<<std::endl;
      if (cbdDebug) {
        std::cout << "\n========================================\n";
        std::cout << "Tree1:\n";
        MA_mditz_print(mTree1);
        std::cout << "========================================\n";
        std::cout << "\n\nTree2:\n";
        MA_mditz_print(mTree2);
        std::cout << "========================================\n";
        std::cout << "\n\nMatching after compute Distance "<< outputMatching.size()<< " : ";
        for (auto t : outputMatching) {
          std::cout << "(" <<std::to_string(std::get<0>(t)) <<";"<<std::to_string(std::get<1>(t)) <<";" << std::to_string(std::get<2>(t))<<"), ";
        }
      }
      // ---------------------
      // ----- Postprocessing
      // --------------------
      if(postprocess_) {
        if (MA_mditz) {
          convertCompleteBranchDecompositionMatching<dataType>(outputMatching, dataMap1, dataMap2, 
                                                            tree1, tree2,   //CBDs
                                                            &(preprocessed_MT1_copy.tree), &(preprocessed_MT2_copy.tree)); //MergeTrees

          mTree1Int = preprocessed_MT1_copy;
          mTree2Int = preprocessed_MT2_copy;
          
          tree1 = &(mTree1Int.tree);
          tree2 = &(mTree2Int.tree);  
          
        } else {
          postprocessingPipeline<dataType>(tree1);
          postprocessingPipeline<dataType>(tree2);
          if(branchDecomposition_)
          convertBranchDecompositionMatching<dataType>(
            tree1, tree2, outputMatching);
        }
      }
      if (cbdDebug) {
        std::cout << "\nMatching after postprocessing "<< outputMatching.size()<< " : ";
        for (auto t : outputMatching) {
          std::cout << "(" <<std::to_string(std::get<0>(t)) <<";"<<std::to_string(std::get<1>(t)) <<";" << std::to_string(std::get<2>(t))<<"), ";
        }
      }
      // std::cout << "TIME COMP.MATCH. = " << t_match_time << std::endl;
      printMsg("Total", 1, t_total.getElapsedTime(), this->threadNumber_,
               debug::LineMode::NEW, debug::Priority::INFO);
      printMsg(debug::Separator::L2);
      std::stringstream ss2;
      ss2 << "DISTANCE²       = "
          << (distanceSquaredRoot_ ? distance * distance : distance);
      printMsg(ss2.str());
      std::stringstream ss3;
      ss3 << "DISTANCE        = "
          << (distanceSquaredRoot_ ? distance : std::sqrt(distance));
      printMsg(ss3.str());
      printMsg(debug::Separator::L2);
      std::stringstream ss4;
      ss4 << "MEMORY          = " << m.getElapsedUsage();
      printMsg(ss4.str());
      printMsg(debug::Separator::L2);

      return distance;
    }

    template <class dataType>
    dataType execute(
      ftm::MergeTree<dataType> &tree1,
      ftm::MergeTree<dataType> &tree2,
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> &outputMatching, int tree1Idx = -1, int tree2Idx = -1) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, double>>
        realOutputMatching;
      dataType res = execute<dataType>(tree1, tree2, realOutputMatching,tree1Idx,tree2Idx);
      for(auto tup : realOutputMatching)
        outputMatching.emplace_back(std::get<0>(tup), std::get<1>(tup));
      return res;
    }

    template <class dataType>
    void computeEditDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      int nRows,
      int nCols) {
      
      Timer t_dyn;
      t_assignment_time_ = 0;
      itParentTime = 0;
      computeTime = 0;
      //MA_mditz
      nodesDone = 0;
      numSinglePairing = 0;
      sizeSinglePairing = 0;
      timeSinglePairing =0;
      maxdegree1SinglePairing = 0;
      maxdegree2SinglePairing = 0;

      numAssignment = 0;
      sizeAssignment = 0;
      timeAssignment = 0;
      maxdegree1Assignment = 0;
      maxdegree2Assignment = 0;

      if(parallelize_) {
        //MA_mditz
        /*
        taskOfNode1 = std::vector<int>(nRows -1, -1);
        taskOfNode2 = std::vector<int>(nCols -1, -1);
        newTaskNumber = 0;
        */
        
        parallelEditDistance(tree1, tree2, treeTable, forestTable,
                             treeBackTable, forestBackTable, nRows, nCols);
        
        if(statsTest){
          
          std::cout << "[StatsTest] NodesPerTask: "<< nodePerTask_<<"\n";
          std::cout << "[StatsTest] ParentItTime: " << itParentTime <<"\n";
          std::cout << "[StatsTest] DPEntryTime: " << computeTime <<"\n";
          std::cout << "[StatsTest] AssNum: "<< numAssignment <<"\n";
          std::cout << "[StatsTest] AssLoad: "<< sizeAssignment<<"\n";
          std::cout << "[StatsTest] AssTime: "<< t_assignment_time_<<"\n";
          std::cout << "[StatsTest] AssCBD1deg: "<< maxdegree1Assignment<<"\n";;
          std::cout << "[StatsTest] AssCBD2deg:  "<< maxdegree2Assignment <<"\n";
          /*
          std::cout << "Single pairing stats: ";
          std::cout << "\n  Num: "<< numSinglePairing;
          std::cout << "\n  Total load: "<< sizeSinglePairing;
          std::cout << "\n  Average load: "<< sizeSinglePairing/numSinglePairing;
          std::cout << "\n  Total time: "<< timeSinglePairing;
          std::cout << "\n  Average time: "<< timeSinglePairing/numSinglePairing;
          std::cout << "\n  Maxdegree 1st tree: "<< maxdegree1SinglePairing;
          std::cout << "\n  Maxdegree 2nd tree: "<< maxdegree2SinglePairing <<"\n\n";
          */
        }
      } else {
        // Distance T1 to empty tree
        //MA_mditz
        std::vector<std::vector<bool>> entryDone(nRows, std::vector<bool>(nCols, false));
        classicEditDistance(tree1, tree2, true, true, tree1->getRoot(),
                            tree2->getRoot(), treeTable, forestTable,
                            treeBackTable, forestBackTable, nRows, nCols, entryDone);
        if(onlyEmptyTreeDistance_ and not MA_mditz)
          return;
        // Distance T2 to empty tree
        classicEditDistance(tree1, tree2, false, true, tree1->getRoot(),
                            tree2->getRoot(), treeTable, forestTable,
                            treeBackTable, forestBackTable, nRows, nCols,entryDone);
        // Distance T1 to T2
        classicEditDistance(tree1, tree2, true, false, tree1->getRoot(),
                            tree2->getRoot(), treeTable, forestTable,
                            treeBackTable, forestBackTable, nRows, nCols,entryDone);
      }

      printMsg("Dynamic programing", 1, t_dyn.getElapsedTime(),
               this->threadNumber_, debug::LineMode::NEW,
               debug::Priority::INFO);
      if(not parallelize_)
        printMsg("Assignment problems", 1, t_assignment_time_,
                 this->threadNumber_, debug::LineMode::NEW,
                 debug::Priority::INFO);
    }

    //MA_mditz
    //If acceleration is enabled and distance does not keep subtrees, check if level ranges intersect if MA_mditz, otherwise Wasserstein level check
    bool checkEntry(ftm::idNode nodeI,ftm::idNode nodeJ,
                            ftm::FTMTree_MT *tree1, ftm::FTMTree_MT *tree2){
      if(keepSubtree_)
        return true;
      if (MA_mditz){
        bool acc = true;
        if(acceleration_){
          int maxlevel1 = std::get<1>(tree1Range_[nodeI]);
          int maxlevel2 = std::get<1>(tree2Range_[nodeJ]);
          int minlevel1 = std::get<0>(tree1Range_[nodeI]);
          int minlevel2 = std::get<0>(tree2Range_[nodeJ]);

          acc = not (maxlevel1 < minlevel2 or maxlevel2< minlevel1);
        }
        return acc and (tree1->getNode(nodeI)->getIsSubtree() == tree2->getNode(nodeJ)->getIsSubtree());
      }
      return tree1Level_[nodeI] == tree2Level_[nodeJ];

    }
    
    template <class dataType>
    void classicEditDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      bool processTree1,
      bool computeEmptyTree,
      ftm::idNode nodeI,
      ftm::idNode nodeJ,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      int nRows,
      int nCols, std::vector<std::vector<bool>> entryDone) {
      
      int const i = computeEmptyTree and not processTree1? 0: nodeI +1;
      int const j = computeEmptyTree and processTree1? 0: nodeJ +1;;
      if (entryDone[i][j]){
        return;
      }
      if(processTree1) {
        std::vector<ftm::idNode> childrens;
        tree1->getChildren(nodeI, childrens);
        for(auto children : childrens)
          classicEditDistance(tree1, tree2, processTree1, computeEmptyTree,
                              children, nodeJ, treeTable, forestTable,
                              treeBackTable, forestBackTable, nRows, nCols,entryDone);
      } else {
        std::vector<ftm::idNode> childrens;
        tree2->getChildren(nodeJ, childrens);
        for(auto children : childrens)
          classicEditDistance(tree1, tree2, processTree1, computeEmptyTree,
                              nodeI, children, treeTable, forestTable,
                              treeBackTable, forestBackTable, nRows, nCols,entryDone);
      }
      
      if(processTree1) {
        if(computeEmptyTree) {
          // --- Forest to empty tree distance
          computeForestToEmptyDistance(tree1, nodeI, i, treeTable, forestTable, forestBackTable);

          // --- Subtree to empty tree distance
          computeSubtreeToEmptyDistance(tree1, nodeI, i, treeTable, forestTable);
        } else
          classicEditDistance(tree1, tree2, false, false, nodeI,
                              tree2->getRoot(), treeTable, forestTable,
                              treeBackTable, forestBackTable, nRows, nCols,entryDone);
      } else {
        if(computeEmptyTree) {
          // --- Empty tree to forest distance
          computeEmptyToForestDistance(tree2, nodeJ, j, treeTable, forestTable, forestBackTable);

          // --- Empty tree to subtree distance
          computeEmptyToSubtreeDistance(tree2, nodeJ, j, treeTable, forestTable);
          //}else{
        //MA_mditz, brauch fixing sobald MA_mditz + keepSubtree_
        } else if(checkEntry(nodeI,nodeJ, tree1, tree2)) {
          nodesDone++;
          std::vector<ftm::idNode> children1;
          tree1->getChildren(nodeI, children1);
          std::vector<ftm::idNode> children2;
          tree2->getChildren(nodeJ, children2);
          // --- Forests distance
          computeForestsDistance(tree1, tree2, i, j, treeTable, forestTable,
                                 forestBackTable, children1, children2);

          // --- Subtrees distance
          computeSubtreesDistance(tree1, tree2, i, j, nodeI, nodeJ, treeTable,
                                  forestTable, treeBackTable, children1,
                                  children2);
        }
      }
      entryDone[i][j] = true;
    }

    // ------------------------------------------------------------------------
    // Parallel version
    // ------------------------------------------------------------------------
    template <class dataType>
    void parallelEditDistance(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      int ttkNotUsed(nRows),
      int ttkNotUsed(nCols)) {
      std::vector<int> tree1NodeChildSize, tree2NodeChildSize;
      for(unsigned int i = 0; i < tree1->getNumberOfNodes(); ++i) {
        std::vector<ftm::idNode> children;
        tree1->getChildren(i, children);
        tree1NodeChildSize.push_back(children.size());
      }
      for(unsigned int j = 0; j < tree2->getNumberOfNodes(); ++j) {
        std::vector<ftm::idNode> children;
        tree2->getChildren(j, children);
        tree2NodeChildSize.push_back(children.size());
      }

      // Get trees data
      std::vector<ftm::idNode> tree1Leaves;
      tree1->getLeavesFromTree(tree1Leaves);
      std::vector<ftm::idNode> tree2Leaves;
      tree2->getLeavesFromTree(tree2Leaves);


      // Distance T1 to empty tree
      Timer timeOfSteps;
      parallelEmptyTreeDistance_v2(tree1, true, tree1Leaves, tree1NodeChildSize,
                                   treeTable, forestTable, treeBackTable,
                                   forestBackTable);
      //std::cout << "T1 -> empty: " << timeOfSteps.getElapsedTime() << std::endl;
      if(onlyEmptyTreeDistance_)
        return;

      //double sub = timeOfSteps.getElapsedTime();
      // Distance T2 to empty tree
      parallelEmptyTreeDistance_v2(tree2, false, tree2Leaves,
                                   tree2NodeChildSize, treeTable, forestTable,
                                   treeBackTable, forestBackTable);
      //std::cout << "T2 -> empty: " << timeOfSteps.getElapsedTime() - sub << std::endl;
      //sub = timeOfSteps.getElapsedTime();
      // Distance T1 to T2
      parallelTreeDistance_v2(tree1, tree2, true, 0, tree1Leaves,
                              tree1NodeChildSize, tree2Leaves,
                              tree2NodeChildSize, treeTable, forestTable,
                              treeBackTable, forestBackTable, true);
      //std::cout << "T1 -> T2: " << timeOfSteps.getElapsedTime() - sub << std::endl;

      //MA_mditz
      /*
      std::vector<int> numNodesOfTask(newTaskNumber -1 , 0);
      std::vector<int> numNodesOfTaskComputed(newTaskNumber -1 , 0);
      for(int taskId : taskOfNode1){
        if(taskId != -1)
          numNodesOfTask[taskId] += 1;
      }
      for(int taskId : taskOfNode2){
        if(taskId != -1)
          numNodesOfTask[taskId] += 1;
      }

      for (int taskId = 0; taskId < newTaskNumber ; taskId++){
        std::cout << "Task " << taskId << " count" << numNodesOfTask[taskId]<<"\n";
      }
      */
    }

    // Forests and subtrees distances
    template <class dataType>
    void parallelTreeDistance_v2(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      bool isTree1,
      int i,
      std::vector<ftm::idNode> &tree1Leaves,
      std::vector<int> &tree1NodeChildSize,
      std::vector<ftm::idNode> &tree2Leaves,
      std::vector<int> &tree2NodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      bool firstCall = false) {
      ftm::idNode const nodeT = -1;
      ftm::FTMTree_MT *treeT = (isTree1) ? tree1 : tree2;
      std::vector<int> treeChildDone(treeT->getNumberOfNodes(), 0);
      std::vector<bool> treeNodeDone(treeT->getNumberOfNodes(), false);
      std::queue<ftm::idNode> treeQueue;
      //std::cout << i-1 << " in upper parallel in  tree 1?"<< isTree1<<std::endl;
      if(isTree1)
        for(ftm::idNode const leaf : tree1Leaves)
          treeQueue.emplace(leaf);
      else if(keepSubtree_ or MA_mditz)
        for(ftm::idNode const leaf : tree2Leaves)
          treeQueue.emplace(leaf);
      else if(tree1Level_[i - 1] < (int)tree2LevelToNode_.size())
        for(ftm::idNode const node : tree2LevelToNode_[tree1Level_[i - 1]])
          treeQueue.emplace(node);

      if(not isCalled_) // and firstCall)
        parallelTreeDistancePara(tree1, tree2, isTree1, i, tree1Leaves,
                                 tree1NodeChildSize, tree2Leaves,
                                 tree2NodeChildSize, treeTable, forestTable,
                                 treeBackTable, forestBackTable, firstCall,
                                 nodeT, treeChildDone, treeNodeDone, treeQueue);
      else
        parallelTreeDistanceTask(tree1, tree2, isTree1, i, tree1Leaves,
                                 tree1NodeChildSize, tree2Leaves,
                                 tree2NodeChildSize, treeTable, forestTable,
                                 treeBackTable, forestBackTable, nodeT,
                                 treeChildDone, treeNodeDone, treeQueue);

      
    }

    // (isCalled_=false)
    template <class dataType>
    void parallelTreeDistancePara(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      bool isTree1,
      int i,
      std::vector<ftm::idNode> &tree1Leaves,
      std::vector<int> &tree1NodeChildSize,
      std::vector<ftm::idNode> &tree2Leaves,
      std::vector<int> &tree2NodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      bool firstCall,
      ftm::idNode nodeT,
      std::vector<int> &treeChildDone,
      std::vector<bool> &treeNodeDone,
      std::queue<ftm::idNode> &treeQueue) {
#ifdef TTK_ENABLE_OPENMP4
#pragma omp parallel num_threads(this->threadNumber_) if(firstCall)
      {
#pragma omp single nowait
#endif
        parallelTreeDistanceTask(tree1, tree2, isTree1, i, tree1Leaves,
                                 tree1NodeChildSize, tree2Leaves,
                                 tree2NodeChildSize, treeTable, forestTable,
                                 treeBackTable, forestBackTable, nodeT,
                                 treeChildDone, treeNodeDone, treeQueue);
#ifdef TTK_ENABLE_OPENMP4
      } // pragma omp parallel
#endif

      TTK_FORCE_USE(firstCall);
    }


    template <class dataType>
    void parallelTreeDistanceTask(
      ftm::FTMTree_MT *tree1,
      ftm::FTMTree_MT *tree2,
      bool isTree1,
      int i,
      std::vector<ftm::idNode> &tree1Leaves,
      std::vector<int> &tree1NodeChildSize,
      std::vector<ftm::idNode> &tree2Leaves,
      std::vector<int> &tree2NodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      ftm::idNode nodeT,
      std::vector<int> &treeChildDone,
      std::vector<bool> &treeNodeDone,
      std::queue<ftm::idNode> &treeQueue) {
      //std::cout << "Num threads: " <<omp_get_num_threads()<< std::endl;
      int nodePerTask = nodePerTask_;
      
      while(!treeQueue.empty()) {
        
        std::queue<ftm::idNode> taskQueue;
        nodePerTask = nodePerTask > (int)treeQueue.size() ? treeQueue.size()
                                                          : nodePerTask;
        for(int j = 0; j < nodePerTask; ++j) {
          nodeT = treeQueue.front();
          treeQueue.pop();
          taskQueue.emplace(nodeT);
        }
        /*
        int taskNumber;
        
        #pragma omp atomic capture
        taskNumber = newTaskNumber++;
        */
#ifdef TTK_ENABLE_OPENMP4
#pragma omp task firstprivate(taskQueue/*, taskNumber*/)  UNTIED()   \
  shared(treeTable, forestTable, treeBackTable, forestBackTable, \
         treeChildDone, treeNodeDone) if(isTree1)
        {
#endif
        ftm::FTMTree_MT *treeT = (isTree1) ? tree1 : tree2;
        while(!taskQueue.empty()) {
          nodeT = taskQueue.front();
          //MA_mditz
          /*
          if (isTree1){
            taskOfNode1[nodeT] = taskNumber;
          }else{
            taskOfNode2[nodeT] = taskNumber;
          }
          */
          taskQueue.pop();
          int const t = nodeT + 1;
          ftm::idNode const nodeI = i - 1;
          if(isTree1) {
            parallelTreeDistance_v2(
              tree1, tree2, false, t, tree1Leaves, tree1NodeChildSize,
              tree2Leaves, tree2NodeChildSize, treeTable, forestTable,
              treeBackTable, forestBackTable, false);
            //}else{
          } else if(checkEntry(nodeI,nodeT,tree1,tree2)){    
             
            nodesDone++;

            int const j = nodeT + 1;
            std::vector<ftm::idNode> children1;
            tree1->getChildren(nodeI, children1);
            std::vector<ftm::idNode> children2;
            tree2->getChildren(nodeT, children2);
            Timer computeTimer;
            // --- Forests distance
            
            computeForestsDistance(tree1, tree2, i, j, treeTable, forestTable,
                                    forestBackTable, children1, children2);

            // --- Subtrees distance
            
            computeSubtreesDistance(tree1, tree2, i, j, nodeI, nodeT,
                                    treeTable, forestTable, treeBackTable,
                                    children1, children2);
            double s = computeTimer.getElapsedTime();
            
            if(statsTest and omp_get_num_threads()==1)
              computeTime += s;
          }

          
          if(not isTree1 and not keepSubtree_
              and not MA_mditz and checkEntry(nodeI,nodeT, tree1,tree2))
            continue;
          
          std::queue<ftm::idNode> parentQueue;
          Timer parentTimer;
          if(MA_mditz){
            std::vector<ftm::idNode> DAGparents;
            treeT->getParents_DAG(nodeT,DAGparents);
            for(ftm::idNode p : DAGparents){
              parentQueue.emplace(p);
            }
          }
          else  
            parentQueue.emplace(treeT->getParentSafe(nodeT));
          //MA_mditz
          while(not parentQueue.empty()){ 
            
            ftm::idNode const nodeTParent = parentQueue.front();
            parentQueue.pop();
            int const childSize = (isTree1) ? tree1NodeChildSize[nodeTParent]
                                            : tree2NodeChildSize[nodeTParent];
            int oldTreeChildDone;
            
#ifdef TTK_ENABLE_OPENMP4
#pragma omp atomic capture
            {
#endif
            oldTreeChildDone = treeChildDone[nodeTParent];
            treeChildDone[nodeTParent]++;
                
                
#ifdef TTK_ENABLE_OPENMP4
            } // pragma omp atomic capture
#endif
            
            if(not treeNodeDone[nodeTParent]
              and oldTreeChildDone + 1 == childSize) {
              
              taskQueue.emplace(nodeTParent);
              treeNodeDone[nodeTParent] = true;
            }
          } // while parentQueue loop (MA_mditz)
          double s = parentTimer.getElapsedTime();
          //#pragma omp atomic
          
          if(statsTest and omp_get_thread_num() == 1)
            itParentTime +=s;
          
        }//while nodeI loop
#ifdef TTK_ENABLE_OPENMP4
        } // pragma omp task
#endif
      
      } // while treeQueue loop
#ifdef TTK_ENABLE_OPENMP4
#pragma omp taskwait
#endif
    }

    // Subtree/Forest with empty tree distances
    template <class dataType>
    void parallelEmptyTreeDistance_v2(
      ftm::FTMTree_MT *tree,
      bool isTree1,
      std::vector<ftm::idNode> &treeLeaves,
      std::vector<int> &treeNodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable) {
      ftm::idNode const nodeT = -1;
      std::vector<int> treeChildDone(tree->getNumberOfNodes(), 0);
      std::vector<bool> treeNodeDone(tree->getNumberOfNodes(), false);
      std::queue<ftm::idNode> treeQueue;
      for(ftm::idNode const leaf : treeLeaves)
        treeQueue.emplace(leaf);
      if(not isCalled_)
        parallelEmptyTreeDistancePara(tree, isTree1, treeLeaves,
                                      treeNodeChildSize, treeTable, forestTable,
                                      treeBackTable, forestBackTable, nodeT,
                                      treeChildDone, treeNodeDone, treeQueue);
      else
        parallelEmptyTreeDistanceTask(tree, isTree1, treeLeaves,
                                      treeNodeChildSize, treeTable, forestTable,
                                      treeBackTable, forestBackTable, nodeT,
                                      treeChildDone, treeNodeDone, treeQueue);
    }

    template <class dataType>
    void parallelEmptyTreeDistancePara(
      ftm::FTMTree_MT *tree,
      bool isTree1,
      std::vector<ftm::idNode> &treeLeaves,
      std::vector<int> &treeNodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      ftm::idNode nodeT,
      std::vector<int> &treeChildDone,
      std::vector<bool> &treeNodeDone,
      std::queue<ftm::idNode> &treeQueue) {
#ifdef TTK_ENABLE_OPENMP4
#pragma omp parallel num_threads(this->threadNumber_)
      {
#pragma omp single nowait
#endif
        parallelEmptyTreeDistanceTask(tree, isTree1, treeLeaves,
                                      treeNodeChildSize, treeTable, forestTable,
                                      treeBackTable, forestBackTable, nodeT,
                                      treeChildDone, treeNodeDone, treeQueue);
#ifdef TTK_ENABLE_OPENMP4
      } // pragma omp parallel
#endif
    }

    template <class dataType>
    void parallelEmptyTreeDistanceTask(
      ftm::FTMTree_MT *tree,
      bool isTree1,
      std::vector<ftm::idNode> &ttkNotUsed(treeLeaves),
      std::vector<int> &treeNodeChildSize,
      std::vector<std::vector<dataType>> &treeTable,
      std::vector<std::vector<dataType>> &forestTable,
      std::vector<std::vector<std::tuple<int, int>>> &treeBackTable,
      std::vector<std::vector<std::vector<std::tuple<int, int>>>>
        &forestBackTable,
      ftm::idNode nodeT,
      std::vector<int> &treeChildDone,
      std::vector<bool> &treeNodeDone,
      std::queue<ftm::idNode> &treeQueue) {
      int nodePerTask = nodePerTask_;
      while(!treeQueue.empty()) {
        std::queue<ftm::idNode> taskQueue;
        nodePerTask = nodePerTask > (int)treeQueue.size() ? treeQueue.size()
                                                          : nodePerTask;
        for(int j = 0; j < nodePerTask; ++j) {
          nodeT = treeQueue.front();
          treeQueue.pop();
          taskQueue.emplace(nodeT);
        }
#ifdef TTK_ENABLE_OPENMP4
#pragma omp task firstprivate(nodeT) UNTIED()                    \
  shared(treeTable, forestTable, treeBackTable, forestBackTable, \
         treeChildDone, treeNodeDone)
        {
#endif
          //MA_mditz : Before that nodeT used
          // while(nodeT != -1){
          while(!taskQueue.empty()) {
            nodeT = taskQueue.front();
            taskQueue.pop();
            if(isTree1) {
              int const i = nodeT + 1;
              // --- Forest to empty tree distance
              computeForestToEmptyDistance(
                tree, nodeT, i, treeTable, forestTable, forestBackTable);

              // --- Subtree to empty tree distance
              computeSubtreeToEmptyDistance(
                tree, nodeT, i, treeTable, forestTable);
            } else {
              int const j = nodeT + 1;
              // --- Empty tree to forest distance
              computeEmptyToForestDistance(
                tree, nodeT, j, treeTable, forestTable, forestBackTable);

              // --- Empty tree to subtree distance
              computeEmptyToSubtreeDistance(
                tree, nodeT, j, treeTable, forestTable);
            }

            //MA_mditz
            std::queue<ftm::idNode> parentQueue;
            // Manage parent
            if(MA_mditz){
              std::vector<ftm::idNode> DAGparents;
              tree->getParents_DAG(nodeT,DAGparents);
              for(ftm::idNode p : DAGparents){
                parentQueue.emplace(p);
              }
            }
            else  
              parentQueue.emplace(tree->getParentSafe(nodeT));
            //MA_mditz
            while(not parentQueue.empty()){ 
              ftm::idNode const nodeTParent = parentQueue.front();
              parentQueue.pop();
              int oldTreeChildDone;
              
#ifdef TTK_ENABLE_OPENMP4
#pragma omp atomic capture
            {
#endif
              oldTreeChildDone = treeChildDone[nodeTParent];
              treeChildDone[nodeTParent]++;
#ifdef TTK_ENABLE_OPENMP4
            } // pragma omp atomic capture
#endif    
              if(not treeNodeDone[nodeTParent]
                and oldTreeChildDone + 1 == treeNodeChildSize[nodeTParent]) {
                taskQueue.emplace(nodeTParent);
                treeNodeDone[nodeTParent] = true;
#ifdef TTK_ENABLE_OPENMP4
#pragma omp taskyield
#endif
              } else 
                  nodeT = -1;
            } //while parentQueue
          } // while nodeI loop
#ifdef TTK_ENABLE_OPENMP4
        } // pragma omp task
#endif
      } // while treeQueue loop
#ifdef TTK_ENABLE_OPENMP4
#pragma omp taskwait
#endif

      TTK_FORCE_USE(treeBackTable);
      TTK_FORCE_USE(forestBackTable);
    }

    // ------------------------------------------------------------------------
    // Utils
    // ------------------------------------------------------------------------
    void printMapIntInt(std::map<int, int> theMap) {
      for(auto itr = theMap.begin(); itr != theMap.end(); ++itr) {
        std::stringstream ss;
        ss << '\t' << itr->first << '\t' << itr->second;
        printMsg(ss.str());
      }
      printMsg("");
    }

    template <class dataType>
    void verifyMergeTreeStructure(ftm::FTMTree_MT *tree) {
      bool problem = false;

      bool const isJT = tree->isJoinTree<dataType>();
      std::vector<std::tuple<ftm::idNode, ftm::idNode>> problemNodes;
      std::queue<ftm::idNode> queue;
      queue.emplace(tree->getRoot());
      while(!queue.empty()) {
        ftm::idNode const node = queue.front();
        queue.pop();

        if(!tree->isRoot(node)) {
          bool thisProblem;
          if(isJT)
            thisProblem = tree->getValue<dataType>(node)
                          > tree->getValue<dataType>(tree->getParentSafe(node));
          else
            thisProblem = tree->getValue<dataType>(node)
                          < tree->getValue<dataType>(tree->getParentSafe(node));

          if(thisProblem)
            problemNodes.emplace_back(node, tree->getParentSafe(node));

          problem |= thisProblem;
        }

        std::vector<ftm::idNode> children;
        tree->getChildren(node, children);
        for(auto c : children)
          queue.emplace(c);
      }

      if(problem) {
        printErr("merge tree in input is not valid");
        for(auto tup : problemNodes) {
          std::stringstream ss;
          ss << std::get<0>(tup) << " _ " << std::get<1>(tup);
          printMsg(ss.str());
        }
        printMsg(tree->printTree().str());
        printMsg(tree->printTreeScalars<dataType>().str());
      }
    }

    // ------------------------------------------------------------------------
    // Testing
    // ------------------------------------------------------------------------
    template <class dataType>
    void classicalPersistenceAssignmentProblem(ftm::FTMTree_MT *tree1,
                                               ftm::FTMTree_MT *tree2) {
      std::vector<std::tuple<ftm::idNode, ftm::idNode, dataType>> pairs1,
        pairs2;
      tree1->getPersistencePairsFromTree(pairs1);
      tree2->getPersistencePairsFromTree(pairs2);
      std::vector<std::vector<dataType>> costMatrix(
        pairs1.size() + 1, std::vector<dataType>(pairs2.size() + 1));
      std::stringstream const ss;
      ss << costMatrix.size() << " _ " << costMatrix[0].size();
      printMsg(ss.str());
      for(unsigned int i = 0; i < costMatrix.size() - 1; ++i) {
        dataType nodeIValue = tree1->getValue<dataType>(std::get<0>(pairs1[i]));
        dataType nodeIOriginValue
          = tree1->getValue<dataType>(std::get<1>(pairs1[i]));
        for(unsigned int j = 0; j < costMatrix[0].size() - 1; ++j) {
          dataType nodeJValue
            = tree2->getValue<dataType>(std::get<0>(pairs2[j]));
          dataType nodeJOriginValue
            = tree2->getValue<dataType>(std::get<1>(pairs2[j]));
          costMatrix[i][j] = std::pow(nodeIValue - nodeJValue, 2)
                             + std::pow(nodeIOriginValue - nodeJOriginValue, 2);
        }
        costMatrix[i][costMatrix[0].size() - 1]
          = 2 * std::pow(std::get<2>(pairs1[i]), 2) / (std::pow(2, 2));
      }
      for(unsigned int j = 0; j < costMatrix[0].size() - 1; ++j)
        costMatrix[costMatrix.size() - 1][j]
          = 2 * std::pow(std::get<2>(pairs2[j]), 2) / (std::pow(2, 2));
      std::vector<MatchingType> matchings;
      forestAssignmentProblemMunkres(costMatrix, matchings);
      dataType cost = 0;
      for(auto tuple : matchings)
        cost += std::get<2>(tuple);
      std::stringstream ss2;
      ss2 << "cost      = " << cost;
      printMsg(ss2.str());
      std::stringstream ss3;
      ss3 << "cost sqrt = " << std::sqrt(cost);
      printMsg(ss3.str());
    }

  }; // MergeTreeDistance class

} // namespace ttk
