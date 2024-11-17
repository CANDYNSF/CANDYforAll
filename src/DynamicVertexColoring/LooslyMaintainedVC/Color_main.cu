#include <stdio.h>
#include "all_structure_undir.cuh"
#include "gpuFunctions_undir.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<vector>
#include <chrono>
#include <algorithm>
#include "compactor.cuh"
#include "supportingFunctions.cu"


#define THREADS_PER_BLOCK 1024 //we can change it

using namespace std;
using namespace std::chrono;



/*
1st arg: original graph file name (input graph should be without wt)
2nd arg: no. of nodes
3rd arg: no. of edges
4th arg: input SSSP file name
5th arg: change edges file name
6th arg: max_iteration (Not used currently) //uncomment max_iteration and use it to control number of iterations
****main commands to run****
nvcc -o op_main CudaSSSPmain.cu
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name
*/
int main(int argc, char* argv[]) {

	int nodes, edges, deviceId, numberOfSMs;
	cudaError_t cudaStatus;
	nodes = atoi(argv[2]);
	edges = atoi(argv[3]);
	char* inputColorfile = argv[4];
	int totalInsertion = 0;
	bool zeroDelFlag = false, zeroInsFlag = false;
	vector<ColList> AdjList; //stores input graph in 2D adjacency list
	vector<ColWt> AdjListFull; //Row-major implementation of adjacency list (1D)
	ColWt* AdjListFull_device; //1D array in GPU to store Row-major implementation of adjacency list 
	int* AdjListTracker_device; //1D array to track offset for each node's adjacency list
	vector<changeEdge> allChange_Ins, allChange_Del;
	changeEdge* allChange_Ins_device; //stores all change edges marked for insertion in GPU
	changeEdge* allChange_Del_device; //stores all change edges marked for deletion in GPU
	int* counter;
	int* affected_marked;
	int* affectedNodeList;
	int* vertexcolor;
	int* previosVertexcolor;
	float total_time = 0.0;
	int SCmaskArrayElement; //not in use. we can remove it from delEdge and insEdge function
	//int sc_mask_startingBlock; //sc_mask_startingBlock sends the 1st element to start from while checking available color. (optimization)
	int max_iteration = 0;

	//max_iteration = atoi(argv[6]); //uncomment this line and use it to control number of iterations

	////Get gpu device id and number of SMs
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	size_t  numberOfBlocks = 32 * numberOfSMs;


	////read input vertex color label
	int maxColor = -1;
	cudaStatus = cudaMallocManaged(&vertexcolor, nodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at vertexcolor structure");
	}
	read_Input_Color(vertexcolor, inputColorfile, &maxColor);

	//test
	//for (int index = 0; index<nodes;index++ )
	//{
	//	
	//		printf("@@@@@vertex color of: %d is: %d\n", index, vertexcolor[index]);
	//	
	//}


	//SCmaskArrayElement = maxColor / 32 + 2; //we take 1 element extra to manage the situation if number of color increases
	SCmaskArrayElement = 2;
	printf("Max color id in input graph: %d\n", maxColor);
	/*sc_mask_startingBlock = ((maxColor / 31) + 1) / 2;
	printf("sc_mask_startingBlock: %d\n", sc_mask_startingBlock);*/

	////Read Original input graph
	AdjList.resize(nodes);
	int* AdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
	read_graphEdges(AdjList, argv[1]);
	/*printf("printing adj list:\n");
	for (int i = 0; i < nodes; i++)
	{
		printf("\nAdj list for %d:\n", i);
		for (int j = 0; j < AdjList[i].size(); j++)
		{
			printf("%d:", AdjList[i][j].col);
		}
	}*/



	////Read change edges input
	readin_changes(argv[5], allChange_Ins, allChange_Del, AdjList, totalInsertion, vertexcolor);
	int totalChangeEdges_Ins = allChange_Ins.size();
	if (totalChangeEdges_Ins == 0) {
		zeroInsFlag = true;
	}
	int totalChangeEdges_Del = allChange_Del.size();
	if (totalChangeEdges_Del == 0) {
		zeroDelFlag = true;
	}
	/*for (int i = 0; i < totalChangeEdges_Ins; i++)
	{
		printf("\nEffective Change edges INS: %d %d:\n", allChange_Ins[i].node1, allChange_Ins[i].node2);
	}*/


	////Transfer input graph, changed edges to GPU and set memory advices
	transfer_data_to_GPU(AdjList, AdjListTracker, AdjListFull, AdjListFull_device,
		nodes, edges, totalInsertion, AdjListTracker_device, zeroInsFlag,
		allChange_Ins, allChange_Ins_device, totalChangeEdges_Ins,
		deviceId, totalChangeEdges_Del, zeroDelFlag, allChange_Del_device,
		counter, affected_marked, affectedNodeList, previosVertexcolor,/*updatedAffectedNodeList_del, updated_counter_del,*/ allChange_Del, numberOfBlocks);


	//test code
	//printf("printing AdjListFull_device:\n"); //after adding ins edges and deleting(or setting flag =-1) del edges
	//printAdj << <numberOfBlocks, THREADS_PER_BLOCK >> > (AdjListFull_device, AdjListTracker_device, nodes);
	//cudaDeviceSynchronize();

	//Saturation color mask
	//int* saturationColorMask;
	//cudaStatus = cudaMallocManaged(&saturationColorMask, nodes * sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed at saturationColor structure");
	//}
	////compute saturation mask
	//computeSCMask << <numberOfBlocks, THREADS_PER_BLOCK >> > (AdjListFull_device, AdjListTracker_device, nodes, saturationColorMask, vertexcolor);
	//cudaDeviceSynchronize();

	//printMask << <numberOfBlocks, THREADS_PER_BLOCK >> > (saturationColorMask, AdjListTracker_device, nodes);
	//cudaDeviceSynchronize();
	//test code ends


	//input color validation //it gives error as we add ins edges while reading
	/*validate << < numberOfBlocks, THREADS_PER_BLOCK >> > (AdjListFull_device, AdjListTracker_device, nodes, vertexcolor);
	cudaDeviceSynchronize();*/


	////Initialize supporting variables
	int* change = 0;
	cudaStatus = cudaMallocManaged(&change, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at change structure");
	}


	////process change edges////

	////Process del edges
	if (zeroDelFlag != true) {
		auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start

		deleteEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (/*states,*/ allChange_Del_device, vertexcolor, previosVertexcolor, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device, affected_marked, change, SCmaskArrayElement);
		cudaDeviceSynchronize(); //comment this if required

		auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
		auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
		cout << "**Time taken for processing del edges: "
			<< float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;
		total_time += float(durationDelEdge.count()) / 1000;
	}

	//Process ins edges
	if (zeroInsFlag != true) {
		auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start

		insEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_device, vertexcolor, previosVertexcolor, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device, affected_marked, change, SCmaskArrayElement);
		cudaDeviceSynchronize(); //comment this if required

		auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
		auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
		cout << "**Time taken for processing ins edges: "
			<< float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
		total_time += float(durationInsEdge.count()) / 1000;
	}

	if (zeroDelFlag != true) {
		cudaFree(allChange_Del_device);
	}
	if (zeroInsFlag != true) {
		cudaFree(allChange_Ins_device);
	}


	auto startTimeDelNeig = high_resolution_clock::now(); //Time calculation start

	//we use compactor in place of just adding directly using atomic fn to avoid duplication of affected vertices in list
	*counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, nodes, predicate(), THREADS_PER_BLOCK);

	/*printf("At first: affectedNodeList elements:\n");
	for (int i = 0; i < *counter; i++)
	{
		printf("%d:", affectedNodeList[i]);
	}*/

	int iteration = 0;

	//recolor affected neighbors
	while (*change > 0)
	{
		*change = 0;
		//reset affected_marked to 0
		cudaMemset(affected_marked, 0, nodes * sizeof(int));
		//printf("after memset 0: affected_del flag for %d = %d \n", 1, affected_del[1]);

		//find eligible neighbors which should be updated
		findEligibleNeighbors << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter, change, iteration, max_iteration);

		if (iteration < max_iteration) {
			recolorNeighbors << <numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter, change, iteration, max_iteration);
		}
		//cudaDeviceSynchronize();
		//find the next frontier: it collects the vertices to be recolored and store without duplicate in affectedNodeList
		*counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, nodes, predicate(), THREADS_PER_BLOCK);
		/*printf("After findEligibleNeighbors *counter: %d affectedNodeList_del elements:", *counter);
		for (int i = 0; i < *counter; i++)
		{
			printf("%d\n", affectedNodeList[i]);
		}*/
		//cudaMemset(affected_marked, 0, nodes * sizeof(int)); //new
		//recolor the eligible neighbors
		//recolorNeighbor << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, vertexcolor, previosVertexcolor, AdjListFull_device, AdjListTracker_device, affected_marked, counter, change, SCmaskArrayElement);
		cudaDeviceSynchronize();
		iteration++;
	}
	auto stopTimeDelNeig = high_resolution_clock::now();//Time calculation ends
	auto durationDelNeig = duration_cast<microseconds>(stopTimeDelNeig - startTimeDelNeig);// duration calculation
	cout << "**Time taken for processing affected neighbors: "
		<< float(durationDelNeig.count()) / 1000 << " milliseconds**" << endl;
	total_time += float(durationDelNeig.count()) / 1000;
	cout << "****Total Time for Vertex Color Update: "
		<< total_time << " milliseconds****" << endl;

	////print output vertex color
	printf("\nprinting output vertex colors:\n");
	for (int i = 0; i < nodes; i++)
	{
		printf("%d:%d\n", i, vertexcolor[i]);
	}

	//Print max color id used
	maxColor = -1;
	for (int i = 0; i < nodes; i++)
	{
		if (vertexcolor[i] > maxColor) {
			maxColor = vertexcolor[i];
		}
	}
	printf("highest color id used: %d\n", maxColor);

	validate << < numberOfBlocks, THREADS_PER_BLOCK >> > (AdjListFull_device, AdjListTracker_device, nodes, vertexcolor);
	cudaDeviceSynchronize();



	cudaFree(change);
	cudaFree(vertexcolor);
	cudaFree(affected_marked);
	cudaFree(affectedNodeList);
	cudaFree(counter);
	cudaFree(AdjListFull_device);
	cudaFree(AdjListTracker_device);
	cudaFree(previosVertexcolor);
	return 0;
}