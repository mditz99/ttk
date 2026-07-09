/// \ingroup vtk
/// \class ttkMergeTreeDistanceMatrix
/// \author Mathieu Pont <mathieu.pont@lip6.fr>
/// \author Florian Wetzels (wetzels@cs.uni-kl.de)
/// \date 2021.
///
/// \brief TTK VTK-filter that wraps the ttk::MergeTreeDistanceMatrix module.
///
/// This VTK filter uses the ttk::MergeTreeDistanceMatrix module to compute the
/// distance matrix of a group of merge trees.
///
/// \param Input vtkMultiBlockDataset
/// \param Output vtkTable
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutputDataObject()).
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
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
///
/// \b Related \b publication \n
/// "Branch Decomposition-Independent Edit Distances for Merge Trees." \n
/// Florian Wetzels, Heike Leitte, and Christoph Garth. \n
/// Computer Graphics Forum, 2022.
///
/// \b Related \b publication \n
/// "A Deformation-based Edit Distance for Merge Trees" \n
/// Florian Wetzels, Christoph Garth. \n
/// TopoInVis 2022.
///
/// \sa ttk::MergeTreeDistanceMatrix
/// \sa ttkAlgorithm
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/mergeTreeClustering/">Merge
///   Tree Clustering example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/mergeTreePGA/">Merge
///   Tree Principal Geodesic Analysis example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceDiagramPGA/">
///   Persistence Diagram Principal Geodesic Analysis example</a> \n

#pragma once

//MA_Mditz
#include <ttkMergeTreeUtils.h>

// VTK Module
#include <ttkMergeTreeDistanceMatrixModule.h>

// VTK Includes
#include <ttkAlgorithm.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

// TTK Base Includes
#include <MergeTreeDistanceMatrix.h>

class TTKMERGETREEDISTANCEMATRIX_EXPORT ttkMergeTreeDistanceMatrix
  : public ttkAlgorithm // we inherit from the generic ttkAlgorithm class
  ,
    protected ttk::MergeTreeDistanceMatrix // and we inherit from the base class
{
private:
  /**
   * Add all filter parameters only as private member variables and
   * initialize them here.
   */
  // Execution Options
  int Backend = 0;
  bool oldBD = branchDecomposition_;
  bool oldNW = normalizedWasserstein_;
  bool oldKS = keepSubtree_;

  bool UseFieldDataParameters = false;

public:
  /**
   * Automatically generate getters and setters of filter
   * parameters via vtkMacros.
   */
  // Input Options
  void SetEpsilon1UseFarthestSaddle(bool epsilon1UseFarthestSaddle) {
    epsilon1UseFarthestSaddle_ = epsilon1UseFarthestSaddle;
    Modified();
  }
  bool GetEpsilon1UseFarthestSaddle() {
    return epsilon1UseFarthestSaddle_;
  }

  void SetEpsilonTree1(double epsilonTree1) {
    epsilonTree1_ = epsilonTree1;
    Modified();
  }
  double SetEpsilonTree1() {
    return epsilonTree1_;
  }

  void SetEpsilon2Tree1(double epsilon2Tree1) {
    epsilon2Tree1_ = epsilon2Tree1;
    Modified();
  }
  double SetEpsilon2Tree1() {
    return epsilon2Tree1_;
  }

  void SetEpsilon3Tree1(double epsilon3Tree1) {
    epsilon3Tree1_ = epsilon3Tree1;
    Modified();
  }
  double SetEpsilon3Tree1() {
    return epsilon3Tree1_;
  }

  void SetPersistenceThreshold(double persistenceThreshold) {
    persistenceThreshold_ = persistenceThreshold;
    Modified();
  }
  double SetPersistenceThreshold() {
    return persistenceThreshold_;
  }

  void SetDeleteMultiPersPairs(bool doDelete) {
    deleteMultiPersPairs_ = doDelete;
    Modified();
  }
  bool SetDeleteMultiPersPairs() {
    return deleteMultiPersPairs_;
  }

  void SetBranchMetric(int m) {
    branchMetric_ = m;
    Modified();
  }

  void SetPathMetric(int m) {
    pathMetric_ = m;
    Modified();
  }

  // Execution Options
  void SetBackend(int newBackend) {
    if(Backend == 2) { // Custom
      oldBD = branchDecomposition_;
      oldNW = normalizedWasserstein_;
      oldKS = keepSubtree_;
    }
    if(newBackend == 2) { // Custom
      branchDecomposition_ = oldBD;
      normalizedWasserstein_ = oldNW;
      keepSubtree_ = oldKS;
    }
    Backend = newBackend;
    Modified();
  }
  vtkGetMacro(Backend, int);

  void SetAssignmentSolver(int assignmentSolver) {
    assignmentSolverID_ = assignmentSolver;
    Modified();
  }
  int GetAssignmentSolver() {
    return assignmentSolverID_;
  }

  void SetBranchDecomposition(bool branchDecomposition) {
    branchDecomposition_ = branchDecomposition;
    Modified();
  }
  int GetBranchDecomposition() {
    return branchDecomposition_;
  }

  void SetNormalizedWasserstein(bool normalizedWasserstein) {
    normalizedWasserstein_ = normalizedWasserstein;
    Modified();
  }
  int GetNormalizedWasserstein() {
    return normalizedWasserstein_;
  }

  void SetKeepSubtree(bool keepSubtree) {
    keepSubtree_ = keepSubtree;
    Modified();
  }
  int GetKeepSubtree() {
    return keepSubtree_;
  }

  //MA_mditz
  void SetAcceleration(bool acceleration) {
    acceleration_ = acceleration;
    Modified();
  }
  
  bool GetAcceleration() {
    return acceleration_;
  }

  //MA_mditz
  void SetStatsTest(bool b) {
    statsTest = b;
    Modified();
  }
  
  bool GetStatsTest() {
    return statsTest;
  }

  //MA_mditz
  void SetParallelFor(bool b) {
    parallelFor = b;
    Modified();
  }
  
  bool GetParallelFor() {
    return parallelFor;
  }

  //MA_mditz
  void SetUseThresholdCBD(bool b) {
    useThresholdCBD_ = b;
    Modified();
  }
  vtkGetMacro(useThresholdCBD_, int);
  
  bool GetUseThresholdCBD() {
    return useThresholdCBD_;
  }

  //MA_mditz
  void SetGlobalThreshold(bool b) {
    globalThreshold_ = b;
    Modified();
  }
  vtkGetMacro(globalThreshold_, int);
  
  bool GetGlobalThreshold() {
    return globalThreshold_;
  }

  //MA_mditz
  void SetThresholdOfCBD(double d) {
    thresholdOfCBD_ = d;
    Modified();
  }
  vtkGetMacro(thresholdOfCBD_, int);
  
  bool GetThresholdOfCBD() {
    return thresholdOfCBD_;
  }

  void SetDistanceSquaredRoot(bool distanceSquaredRoot) {
    distanceSquaredRoot_ = distanceSquaredRoot;
    Modified();
  }
  int GetDistanceSquaredRoot() {
    return distanceSquaredRoot_;
  }

  vtkSetMacro(UseFieldDataParameters, bool);
  vtkGetMacro(UseFieldDataParameters, bool);

  vtkSetMacro(mixtureCoefficient_, double);
  vtkGetMacro(mixtureCoefficient_, double);

  /**
   * This static method and the macro below are VTK conventions on how to
   * instantiate VTK objects. You don't have to modify this.
   */
  static ttkMergeTreeDistanceMatrix *New();
  vtkTypeMacro(ttkMergeTreeDistanceMatrix, ttkAlgorithm);

protected:
  /**
   * Implement the filter constructor and destructor
   *         (see cpp file)
   */
  ttkMergeTreeDistanceMatrix();
  ~ttkMergeTreeDistanceMatrix() override;

  /**
   * Specify the input data type of each input port
   *         (see cpp file)
   */
  int FillInputPortInformation(int port, vtkInformation *info) override;

  /**
   * Specify the data object type of each output port
   *         (see cpp file)
   */
  int FillOutputPortInformation(int port, vtkInformation *info) override;

  /**
   * Pass VTK data to the base code and convert base code output to VTK
   *          (see cpp file)
   */
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;

  template <class dataType>
  int run(vtkInformationVector *outputVector,
          std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees,
          std::vector<vtkSmartPointer<vtkMultiBlockDataSet>> &inputTrees2);
};



namespace ttk{
  namespace ftm{
    

    template <class dataType> 
    void testingStats(MergeTree<dataType>& completeBD, MergeTree<dataType>& inputMT){
      std::cout << "\n\n----------------------------------------------------------------------------\nStats of input Merge Tree\n";
      std::cout << "#Nodes: " << inputMT.tree.getNumberOfNodes()<<"; #Arcs: " << inputMT.tree.getNumberOfSuperArcs()
                <<"; Depth: " << inputMT.tree.getTreeDepth();

      std::cout << "\n\n----------------------------------------------------------------------------\nTesting stats to check correctness of CBD heuristcally\n";
      std::cout << "----------------------------------------------------------------------------\n";
      std::cout << "\n\nStats of complete Branch Decomposition\n";
      std::cout << "#Nodes: " << completeBD.tree.getNumberOfNodes()<<"; #Arcs: " << completeBD.tree.getNumberOfSuperArcs();

      
      
      unsigned int s = 0;
      unsigned int b = 0;
      for(unsigned int id = 0; id < completeBD.tree.getNumberOfNodes(); ++id){
        if(completeBD.tree.getNode(id)->getIsSubtree()){
          s++;
        }
        else{
          b++;
        }
      }
      std::cout << "\n#Subtree Nodes: " << s << "; #Branch Nodes: " << b << "\n\n";

      std::cout << "#Arcs in Merge Tree == #Subtree nodes?\n";
      if(s == inputMT.tree.getNumberOfSuperArcs()){
        std::cout << "YES! ️✅\n";
      }
      else{
        std::cout << "NO! ❌ #Arcs = " << inputMT.tree.getNumberOfSuperArcs()<< "; #Subtree Nodes = " << s <<"\n";
      }

      std::cout << "#Branches in Merge Tree == #Branch Nodes?\n";
      
      std::vector<idNode> leaves;
      inputMT.tree.getLeavesFromTree(leaves);
      std::queue<idNode> updraft;
      for(idNode leaf : leaves){
        updraft.push(leaf);
      }

      unsigned int srl = 0;
      while(!updraft.empty()){
        idNode front = updraft.front();
        updraft.pop();
        idNode parent = inputMT.tree.getParentSafe(front);
        if(parent != front){
          srl += 1;
          updraft.push(parent);
        }
      }

      if(b == srl){
        std::cout << "YES! ️✅\n";
      }
      else{
        std::cout << "NO! ❌ Sum = " << srl<< "; #Branch Nodes = " << b <<"\n";
      }

      std::cout << "----------------------------------------------------------------------------" << std::endl;
    }

    template <class dataType> 
    void testingStats(std::shared_ptr<MergeTree<dataType>> completeBDptr, MergeTree<dataType>* inputMTptr){
      std::cout << "\n\n----------------------------------------------------------------------------\nStats of input Merge Tree\n";
      std::cout << "#Nodes: " << inputMTptr->tree.getNumberOfNodes()<<"; #Arcs: " << inputMTptr->tree.getNumberOfSuperArcs()
                <<"; Depth: " << inputMTptr->tree.getTreeDepth();

      std::cout << "\n\n----------------------------------------------------------------------------\nTesting stats to check correctness of CBD heuristcally\n";
      std::cout << "----------------------------------------------------------------------------\n";
      std::cout << "\n\nStats of complete Branch Decomposition\n";
      std::cout << "#Nodes: " << completeBDptr->tree.getNumberOfNodes()<<"; #Arcs: " << completeBDptr->tree.getNumberOfSuperArcs();

      
      
      unsigned int s = 0;
      unsigned int b = 0;
      for(unsigned int id = 0; id < completeBDptr->tree.getNumberOfNodes(); ++id){
        if(completeBDptr->tree.getNode(id)->getIsSubtree()){
          s++;
        }
        else{
          b++;
        }
      }
      std::cout << "\n#Subtree Nodes: " << s << "; #Branch Nodes: " << b << "\n\n";

      std::cout << "#Arcs in Merge Tree == #Subtree nodes?\n";
      if(s == inputMTptr->tree.getNumberOfSuperArcs()){
        std::cout << "YES! ️✅\n";
      }
      else{
        std::cout << "NO! ❌ #Arcs = " << inputMTptr->tree.getNumberOfSuperArcs()<< "; #Subtree Nodes = " << s <<"\n";
      }

      std::cout << "#Branches in Merge Tree == #Branch Nodes?\n";
      
      std::vector<idNode> leaves;
      inputMTptr->tree.getLeavesFromTree(leaves);
      std::queue<idNode> updraft;
      for(idNode leaf : leaves){
        updraft.push(leaf);
      }

      unsigned int srl = 0;
      while(!updraft.empty()){
        idNode front = updraft.front();
        updraft.pop();
        idNode parent = inputMTptr->tree.getParentSafe(front);
        if(parent != front){
          srl += 1;
          updraft.push(parent);
        }
      }

      if(b == srl){
        std::cout << "YES! ️✅\n";
      }
      else{
        std::cout << "NO! ❌ Sum = " << srl<< "; #Branch Nodes = " << b <<"\n";
      }

      std::cout << "----------------------------------------------------------------------------" << std::endl;
    }


    
    template <class dataType> 
    std::shared_ptr<MergeTree<dataType>> computeCompleteBranchDecomposition(
      MergeTree<dataType>* inputMTptr, idNode pId, idNode cId) {
      

      //=====================================================
      //Bottom up  through Edge Recursion of Input MergeTree
      //=====================================================
      //Conventions for the result CBD for easier handling throughout the algorithm: 
      //- Subtree node resembling the entire tree are at entry 0
      //- idNode == Index of node in nodes vector
      //- Subtree nodes store the higher(closer to root) node's scalar, but just to have an origin for the branch nodes. 
      //  Theoretically, the not have any scalar value.
      //- Branch nodes store the lowest node's scalar

      
      std::vector<idNode> ccs; 
      inputMTptr->tree.getChildren(cId, ccs);
      unsigned int numCcs = ccs.size();
      //std::cout << "---------------------- Start Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;

      //std::cout << "children of " << cId <<": ";
      for(unsigned int i = 0; i < numCcs; ++i){
          //std::cout <<  ccs[i] << ", ";d
      }    
      //std::cout << std::endl;
      //  Base case
      //  O parent p
      //  |                      
      //  O child c
      if(numCcs ==  0) {
        MergeTree<dataType> in2 = createEmptyMergeTree<dataType>(2);
        std::shared_ptr<MergeTree<dataType>> completeBDptr = std::make_shared<MergeTree<dataType>>(in2);

        completeBDptr->tree.makeNode(0); 
        completeBDptr->tree.makeNode(1);

        completeBDptr->tree.makeSuperArc(1, 0);
        completeBDptr->tree.getNode(0)->setOrigin(1);  
        completeBDptr->tree.getNode(1)->setOrigin(0);
        
        completeBDptr->tree.getNode(0)->setIsSubtree(true);
        ////std::cout <<"(" <<pId << "," << cId <<") Root subtree node "<<completeBDptr->tree.getNode(0)->getIsSubtree() << std::endl;

        std::vector<dataType> scalarsCBD;
        scalarsCBD.push_back(inputMTptr->tree.template getValue<dataType>(pId));
        scalarsCBD.push_back(inputMTptr->tree.template getValue<dataType>(cId));

        completeBDptr->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBDptr->scalars->values = (void *)(completeBDptr->scalarsValues->data());
        completeBDptr->scalars->size = completeBDptr->scalarsValues->size();
        //ftm::setTreeScalars<dataType>(*completeBDptr, scalarsCBD);
        ////std::cout << "---------------------- End Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;
        return completeBDptr;
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
        std::vector<std::shared_ptr<MergeTree<dataType>>> recursion_results;
        for(unsigned int i = 0; i < numCcs; ++i){
          
          
          recursion_results.push_back(computeCompleteBranchDecomposition<dataType>(inputMTptr, cId, ccs[i]));
          
          ////std::cout <<"(" <<pId << "," << cId <<") Recursion result"<<"(" <<cId << "," << ccs[i] <<"): "<< recursion_results[i]->scalars->size << std::endl;
        }    
        
        //---------------------------------------------------------
        //Computing the number of nodes in the result
        
                        
        unsigned int numNodesResult = 1; //main subtree node
        for(unsigned int i = 0; i < numCcs; ++i){
          
          std::shared_ptr<MergeTree<dataType>> currCBD = recursion_results[i];
          unsigned int currNum = currCBD->scalars->size;
          numNodesResult += currNum; //add on of recursive result
          std::vector<idNode> cs;
          currCBD->tree.getChildren(0,cs);
          numNodesResult += cs.size(); //add extended main branches (c - l while l some leaf) from recursion result to p - l          
        }
        //---------------------------------------------------------
        //Initialize CBD result
        //std::cout <<"(" <<pId << "," << cId <<")"<< "numNodesResult: " << numNodesResult << std::endl;
        MergeTree<dataType> in2 = createEmptyMergeTree<dataType>(numNodesResult);
        std::shared_ptr<MergeTree<dataType>> completeBDptr = std::make_shared<MergeTree<dataType>>(in2);
        
        *completeBDptr = in2;

        //Remember scalars throughout merging and later steps to set at the end.
        std::vector<dataType>scalarsCBD = std::vector<dataType>(numNodesResult);

        //Make main subtree node
        completeBDptr->tree.makeNode(0); 
        scalarsCBD[0] = inputMTptr->tree.template getValue<dataType>(pId);
        completeBDptr->tree.getNode(0)->setIsSubtree(true);
        //std::cout <<"(" <<pId << "," << cId <<") Root subtree node "<<completeBDptr->tree.getNode(0)->getIsSubtree() << std::endl;

        //---------------------------------------------------------
        //Merge recursion results

        unsigned int numAddedNodes = 1;

        //References to main subtree nodes of the recursion results in merged CBDs
        std::vector<idNode> mainSTNodes = std::vector<idNode>(numCcs);

        for(unsigned int i = 0; i < numCcs; ++i){
          //cciIH := CBD of Merge Tree rooted at [c,cc_i] through induction hypothesis 
          std::shared_ptr<MergeTree<dataType>> cciIH = recursion_results[i];
          unsigned int currOffset = numAddedNodes;
          mainSTNodes[i] = currOffset;
        
          //Add all nodes, their origins, scalar values and subtree bool of cciIH
          for(unsigned int nId = 0; nId < cciIH->tree.getNumberOfNodes(); ++nId){//(By conventions fine traversal of arcs)
            unsigned int newId = currOffset + nId;
            completeBDptr->tree.makeNode(newId);
            //std::cout <<"(" <<pId << "," << cId << ")"<< "(Merge)Node made: " <<  newId ;
            Node* newNode = completeBDptr->tree.getNode(newId);
            Node* oldNode = cciIH->tree.getNode(nId);
            newNode->setOrigin(oldNode->getOrigin() + currOffset); //Get origin of node, set it with offset
            newNode->setIsSubtree(oldNode->getIsSubtree());
            //std::cout << "; is subtree: " << newNode->getIsSubtree() << std::endl;
            numAddedNodes += 1;
            dataType val = cciIH->tree.template getValue<dataType>(nId);
            scalarsCBD[newId] = val;
          }

          //Add all arcs of cciIH 
          for(unsigned int aId = 0; aId < cciIH->tree.getNumberOfSuperArcs(); ++aId){ //(By conventions fine traversal of arcs)
            SuperArc* currArc = cciIH->tree.getSuperArc(aId);
            completeBDptr->tree.makeSuperArc(currArc->getDownNodeId()+ currOffset , currArc->getUpNodeId()+ currOffset);
            //std::cout <<"(" <<pId << "," << cId << ")"<< "(Merge)Arc made: " <<  currArc->getUpNodeId() << ","<< currArc->getDownNodeId()<<std::endl;
          }
        }
        //---------------------------------------------------------
        //Fill up the CBD 
        // - Add Extended main branches (c-l) of the cciIHs to (p-l)
        // - Add arc [Main subtree node, p-l]
        // - Add arcs of (c-l) to (p-l) (all adjacent subtree nodes to (c-l) are also adjacent to (p-l))
        // - Add arcs [p-l, mainSTNodes[j]] with j not i

        for(unsigned int i = 0; i < numCcs; ++i){
          std::vector<idNode> mainbranches;
          completeBDptr->tree.getChildren(mainSTNodes[i],mainbranches);
          //std::cout<<"(" <<pId << "," << cId <<")" <<"(Fill) #Mainbranches of " << i << ": " << mainbranches.size() << std::endl;
          for(idNode currMainBranch : mainbranches){
            unsigned int currExtendedMainBranch = numAddedNodes;

            //Add (p-l) as a node
            completeBDptr->tree.makeNode(currExtendedMainBranch);
            scalarsCBD[currExtendedMainBranch] = scalarsCBD[currMainBranch];
            completeBDptr->tree.getNode(currExtendedMainBranch)->setOrigin(0); 
            numAddedNodes += 1;

            //std::cout<<"(" <<pId << "," << cId <<")" <<"(Fill)Node made: " <<  currExtendedMainBranch << std::endl;

            //Add arc [Main subtree node, p-l]
            completeBDptr->tree.makeSuperArc(currExtendedMainBranch, 0);
            
            //Add arcs of (c-l) to (p-l)
            std::vector<idNode> adjSubtreesCL;
            completeBDptr->tree.getChildren(currMainBranch, adjSubtreesCL);
            for(idNode adjTree : adjSubtreesCL){
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
        //std::cout <<"(" <<pId << "," << cId << ")"<<"NumAddedNodes" << numAddedNodes << std::endl;
        //---------------------------------------------------------
        completeBDptr->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBDptr->scalars->values = (void *)(completeBDptr->scalarsValues->data());
        completeBDptr->scalars->size = completeBDptr->scalarsValues->size();
        //setTreeScalars(completeBDptr, scalarsCBD);
        //std::cout << "---------------------- End Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;
        return completeBDptr;
      }
    }
    

    
    template <class dataType> 
    MergeTree<dataType>& computeCompleteBranchDecomposition(
      MergeTree<dataType>& inputMT, idNode pId, idNode cId) {
      

      //=====================================================
      //Bottom up  through Edge Recursion of Input MergeTree
      //=====================================================
      //Conventions for the result CBD for easier handling throughout the algorithm: 
      //- Subtree node resembling the entire tree are at entry 0
      //- idNode == Index of node in nodes vector
      //- Subtree nodes store the higher(closer to root) node's scalar, but just to have an origin for the branch nodes. 
      //  Theoretically, the not have any scalar value.
      //- Branch nodes store the lowest node's scalar

      
      std::vector<idNode> ccs; 
      inputMT.tree.getChildren(cId, ccs);
      unsigned int numCcs = ccs.size();
      std::cout << "---------------------- Start Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;

      //std::cout << "children of " << cId <<": ";
      for(unsigned int i = 0; i < numCcs; ++i){
          //std::cout <<  ccs[i] << ", ";d
      }    
      //std::cout << std::endl;
      //  Base case
      //  O parent p
      //  |                      
      //  O child c
      if(numCcs ==  0) {
        MergeTree<dataType> completeBD = createEmptyMergeTree<dataType>(2);
        //std::shared_ptr<MergeTree<dataType>> completeBDptr = std::make_shared<MergeTree<dataType>>(in2);

        completeBD->tree.makeNode(0); 
        completeBD->tree.makeNode(1);

        completeBD->tree.makeSuperArc(1, 0);
        completeBD->tree.getNode(0)->setOrigin(1);  
        completeBD->tree.getNode(1)->setOrigin(0);
        
        completeBD->tree.getNode(0)->setIsSubtree(true);
        std::cout <<"(" <<pId << "," << cId <<") Root subtree node "<<completeBD->tree.getNode(0)->getIsSubtree() << std::endl;

        std::vector<dataType> scalarsCBD;
        scalarsCBD.push_back(inputMT.tree.template getValue<dataType>(pId));
        scalarsCBD.push_back(inputMT.tree.template getValue<dataType>(cId));

        completeBD->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBD->scalars->values = (void *)(completeBD->scalarsValues->data());
        completeBD->scalars->size = completeBD->scalarsValues->size();
        //ftm::setTreeScalars<dataType>(*completeBD, scalarsCBD);
        std::cout << "---------------------- End Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;
        return completeBD;
      }
      else{ 
        //  Step case
        //      O parent p
        //      |
        //      O child c
        //   ( ... ) 
        //   O     O children 
        

        //---------------------------------------------------------
        //Recurse
        std::vector<MergeTree<dataType>> recursion_results;
        for(unsigned int i = 0; i < numCcs; ++i){
          MergeTree<dataType>& inres = computeCompleteBranchDecomposition<dataType>(inputMT, cId, ccs[i]);
          
          recursion_results.emplace_back(inres);
          
          //std::cout <<"(" <<pId << "," << cId <<") Recursion result"<<"(" <<cId << "," << ccs[i] <<"): "<< recursion_results[i].scalars->size << std::endl;
        }    
        
        //---------------------------------------------------------
        //Computing the number of nodes in the result
        
                        
        unsigned int numNodesResult = 1; //main subtree node
        for(unsigned int i = 0; i < numCcs; ++i){
          
          MergeTree<dataType>& currCBD = recursion_results[i];
          unsigned int currNum = currCBD.scalars->size;
          numNodesResult += currNum; //add on of recursive result
          std::vector<idNode> cs;
          currCBD.tree.getChildren(0,cs);
          numNodesResult += cs.size(); //add extended main branches (c - l while l some leaf) from recursion result to p - l          
        }
        //---------------------------------------------------------
        //Initialize CBD result
        //std::cout <<"(" <<pId << "," << cId <<")"<< "numNodesResult: " << numNodesResult << std::endl;
        MergeTree<dataType> completeBD = createEmptyMergeTree<dataType>(numNodesResult);
        //std::shared_ptr<MergeTree<dataType>> completeBDptr = std::make_shared<MergeTree<dataType>>(in2);
        
        //*completeBD = in2;

        //Remember scalars throughout merging and later steps to set at the end.
        std::vector<dataType>scalarsCBD = std::vector<dataType>(numNodesResult);

        //Make main subtree node
        completeBD->tree.makeNode(0); 
        scalarsCBD[0] = inputMT.tree.template getValue<dataType>(pId);
        completeBD->tree.getNode(0)->setIsSubtree(true);
        //std::cout <<"(" <<pId << "," << cId <<") Root subtree node "<<completeBD->tree.getNode(0)->getIsSubtree() << std::endl;

        //---------------------------------------------------------
        //Merge recursion results

        unsigned int numAddedNodes = 1;

        //References to main subtree nodes of the recursion results in merged CBDs
        std::vector<idNode> mainSTNodes = std::vector<idNode>(numCcs);

        for(unsigned int i = 0; i < numCcs; ++i){
          //cciIH := CBD of Merge Tree rooted at [c,cc_i] through induction hypothesis 
          MergeTree<dataType>& cciIH = recursion_results[i];
          unsigned int currOffset = numAddedNodes;
          mainSTNodes[i] = currOffset;
        
          //Add all nodes, their origins, scalar values and subtree bool of cciIH
          for(unsigned int nId = 0; nId < cciIH.tree.getNumberOfNodes(); ++nId){//(By conventions fine traversal of arcs)
            unsigned int newId = currOffset + nId;
            completeBD->tree.makeNode(newId);
            //std::cout <<"(" <<pId << "," << cId << ")"<< "(Merge)Node made: " <<  newId ;
            Node* newNode = completeBD->tree.getNode(newId);
            Node* oldNode = cciIH.tree.getNode(nId);
            newNode->setOrigin(oldNode->getOrigin() + currOffset); //Get origin of node, set it with offset
            newNode->setIsSubtree(oldNode->getIsSubtree());
            //std::cout << "; is subtree: " << newNode->getIsSubtree() << std::endl;
            numAddedNodes += 1;
            dataType val = cciIH.tree.template getValue<dataType>(nId);
            scalarsCBD[newId] = val;
          }

          //Add all arcs of cciIH 
          for(unsigned int aId = 0; aId < cciIH.tree.getNumberOfSuperArcs(); ++aId){ //(By conventions fine traversal of arcs)
            SuperArc* currArc = cciIH.tree.getSuperArc(aId);
            completeBD->tree.makeSuperArc(currArc->getDownNodeId()+ currOffset , currArc->getUpNodeId()+ currOffset);
            //std::cout <<"(" <<pId << "," << cId << ")"<< "(Merge)Arc made: " <<  currArc->getUpNodeId() << ","<< currArc->getDownNodeId()<<std::endl;
          }
        }
        //---------------------------------------------------------
        //Fill up the CBD 
        // - Add Extended main branches (c-l) of the cciIHs to (p-l)
        // - Add arc [Main subtree node, p-l]
        // - Add arcs of (c-l) to (p-l) (all adjacent subtree nodes to (c-l) are also adjacent to (p-l))
        // - Add arcs [p-l, mainSTNodes[j]] with j not i

        for(unsigned int i = 0; i < numCcs; ++i){
          std::vector<idNode> mainbranches;
          completeBD->tree.getChildren(mainSTNodes[i],mainbranches);
          //std::cout<<"(" <<pId << "," << cId <<")" <<"(Fill) #Mainbranches of " << i << ": " << mainbranches.size() << std::endl;
          for(idNode currMainBranch : mainbranches){
            unsigned int currExtendedMainBranch = numAddedNodes;

            //Add (p-l) as a node
            completeBD->tree.makeNode(currExtendedMainBranch);
            scalarsCBD[currExtendedMainBranch] = scalarsCBD[currMainBranch];
            completeBD->tree.getNode(currExtendedMainBranch)->setOrigin(0); 
            numAddedNodes += 1;

            //std::cout<<"(" <<pId << "," << cId <<")" <<"(Fill)Node made: " <<  currExtendedMainBranch << std::endl;

            //Add arc [Main subtree node, p-l]
            completeBD->tree.makeSuperArc(currExtendedMainBranch, 0);
            
            //Add arcs of (c-l) to (p-l)
            std::vector<idNode> adjSubtreesCL;
            completeBD->tree.getChildren(currMainBranch, adjSubtreesCL);
            for(idNode adjTree : adjSubtreesCL){
              completeBD->tree.makeSuperArc(adjTree,currExtendedMainBranch);
            }

            //Add arcs [p-l, mainSTNodes[j]] with j not i
            for(unsigned int j = 0; j < numCcs; ++j){
              if(i != j){
                completeBD->tree.makeSuperArc(mainSTNodes[j], currExtendedMainBranch);
              }
            }
            
          } 
        }
        //std::cout <<"(" <<pId << "," << cId << ")"<<"NumAddedNodes" << numAddedNodes << std::endl;
        //---------------------------------------------------------
        completeBD->scalarsValues = std::make_shared<std::vector<dataType>>(scalarsCBD);
        completeBD->scalars->values = (void *)(completeBD->scalarsValues->data());
        completeBD->scalars->size = completeBD->scalarsValues->size();
        //setTreeScalars(completeBD, scalarsCBD);
        std::cout << "---------------------- End Recursion call " << pId << " " << cId << "-----------------------------" << std::endl;
        return completeBD;
      }
    }


    template <class dataType>
    void executeAndPrint(MergeTree<dataType>& exp){
      printTree(exp, true);

      std::cout << "Scalars of Merge Tree: " << std::endl;
      for (unsigned int i = 0; i < exp.tree.getNumberOfNodes(); ++i){
        std::cout <<"   Node " << i << ":  " << exp.tree.template getValue<dataType>(i)<< "\n";
      }
      std::cout <<"\n\n";
      
      
      std::vector<idNode> children;
      exp.tree.getChildren(exp.tree.getRoot(),children);
      
      //Timer timer;
      std::shared_ptr<MergeTree<dataType>> resexpptr=computeCompleteBranchDecomposition<dataType>(&exp, exp.tree.getRoot(), children[0]);
      //std::cout << "Computation of CBD took " << timer.getElapsedTime() << "s\n";
      
      std::cout << "Shared pointer object:" << "\n";
      for (unsigned int i = 0; i < resexpptr->tree.getNumberOfNodes(); ++i){
        std::cout << "Node " << i ; 
        std::cout << "\n    Scalar: "  << resexpptr->tree.template getValue<dataType>(i);
        std::cout << "\n    Origin: " << resexpptr->tree.getNode(i)->getOrigin();
        std::cout << "\n    Is subtree: " << resexpptr->tree.getNode(i)->getIsSubtree() ;
        std::cout << "\n    Children: ";
        std::vector<idNode> ccs2;
        resexpptr->tree.getChildren(i,ccs2);
        for (idNode c: ccs2){
          std::cout << c << ", ";
        }
        std::cout << "\n";
      }
      std::cout << std::endl;
      

      testingStats(resexpptr, &exp);
    }
  }
}


