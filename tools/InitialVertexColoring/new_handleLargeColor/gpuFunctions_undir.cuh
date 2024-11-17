#ifndef GPUFUNCTIONS_UNDIR_CUH
#define GPUFUNCTIONS_UNDIR_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "all_structure_undir.cuh"
#include <math.h>
using namespace std;

//for random int generation
#include <curand_kernel.h>
#include <curand.h>


#define THREADS_PER_BLOCK 1024 //we can change it


/*
* BitCount returns number of 1 in mask
* It indicates how many colors are used in SC mask
*/
__device__ int BitCount(unsigned int mask)
{
	unsigned int uCount;
	uCount = mask
		- ((mask >> 1) & 033333333333)
		- ((mask >> 2) & 011111111111);
	return ((uCount + (uCount >> 3))
		& 030707070707) % 63;
}

/*
* FirstZeroBit fn finds first 0 bit and returns the index
* It helps to find minimum available color in SC mask
*/
__device__ int FirstZeroBit(int* maskArray, int maskElement)
{
	int firstZeroBit = -1;
	for (int j = 0; j < maskElement; j++)
	{
		if (maskArray[j] < 4294967295) //2^32 -1 = 4294967295 //when all 32 bits are set to 1
		{
			int i = maskArray[j];
			i = ~i;
			firstZeroBit = BitCount((i & (-i)) - 1) + j * 32;
			//printf("\n maskArray: %d firstZeroBit: %d", maskArray[j], firstZeroBit);
			break;
		}

	}

	return firstZeroBit;
}


/*
* Test : use any available color
*/
__device__ int selectColor(int* maskArray, int maskElement, curandState* states, int id)
{
	int colorId = -1;
	//int flag = 0;
	//int y = 0, size_Array = 0;
	//int availableColorArray[5];
	//for (int j = 0; j < maskElement; j++)
	//{
	//	if (maskArray[j] < 4294967295) //2^32 -1 = 4294967295 //when all 32 bits are set to 1
	//	{
	//		y = 0;
	//		int n = maskArray[j];
	//		while (n != 0) {
	//			if ((n & 1) == 0 && size_Array < 5 ) {
	//				availableColorArray[size_Array] = y + j * 32;
	//				size_Array++;
	//			}
	//			n = n >> 1; //right shift 1 bit
	//			y++;
	//		}
	//	}
	//	if (size_Array == 5) {
	//		break;
	//	}
	//}
	//size_Array--;
	int seed = id; // different seed per thread //**change it to time
	int flag = 0;
	curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND //taking seed = 0, sequence = id
	while (flag != 1)
	{

		float myrandf = curand(&states[id]) % 32;
		if ((maskArray[0] & int(exp2(double(myrandf)))) == 1)
		{
			colorId = int(myrandf);
			flag = 1;
		}
	}
	//float myrandf = curand_uniform(&states[id]);
	//myrandf *= (size_Array - 0 + 0.999999);  //max is size_Array, min is 0 as we have array of size 5
	////myrandf += 0;  //min = 0, so adding min. we can ignore this line here as it is 0
	//int myrand = (int)truncf(myrandf);
	/*colorId = availableColorArray[myrand];
	printf("generated randon number is: %d, colorID:%d", myrand, colorId);*/

	return colorId;
}






/*
* computeSC function creates a 32 bit mask, where ith bit is set if one of the adjacent vertex has color i
*/
__device__ void computeSC(ColWt* AdjListFull_device, int* AdjListTracker_device, int node, int* vertexcolor, int* maskArray) {
	//int saturationColorMask = 0;
	int maskElement;
	//printf("\n node: %d ", node);
	for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
		if (AdjListFull_device[j].flag != -1)
		{
			//printf("ngbr: %d ", AdjListFull_device[j].col);
			maskElement = vertexcolor[AdjListFull_device[j].col] / 31; //it should be 31 (maskElement will be 0) as color id 32 will be in maskElement 1
			//printf(" maskElement: %d ", maskElement);
			//printf(" maskArray[maskElement](before): %d ", maskArray[maskElement]);
			maskArray[maskElement] = maskArray[maskElement] | int(exp2(double(vertexcolor[AdjListFull_device[j].col] % 32)));  //%32 is used so that in every element of the maskArray max value will be 2^32
			//printf(" maskArray[maskElement]: %d ", maskArray[maskElement]);

			//saturationColorMask = saturationColorMask | int(exp2(double(vertexcolor[AdjListFull_device[j].col])));
		}
	}
	//return maskArray;
}



__device__ int computeSC_32bit(ColWt* AdjListFull_device, int* AdjListTracker_device, int node, int* vertexcolor, int itr) {
	int saturationColorMask = 0;
	for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
		if (AdjListFull_device[j].flag != -1 && (vertexcolor[AdjListFull_device[j].col] < 31*itr) && (vertexcolor[AdjListFull_device[j].col] >= 31 * (itr - 1)))
		{
			//select colors x if 31 * (itr - 1) <= x < 31 * itr
			saturationColorMask = saturationColorMask | int(exp2(double(vertexcolor[AdjListFull_device[j].col]%31))); //as signed int can take upto 2^31 -1
			//printf("@@@@saturationColorMask: %d\n", saturationColorMask);
		}
	}
	return saturationColorMask;
}

__device__ int FirstZeroBit_32bit(int i)
{
	i = ~i;
	return BitCount((i & (-i)) - 1);
}


__global__ void printAdj(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			//printf("\nAdj list for %d:\n", index);
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				if (AdjListFull_device[j].flag != -1)
				{
					//printf("%d:", AdjListFull_device[j].col);
				}
			}
		}
	}
}

__global__ void printMask(int* saturationColorMask, int* AdjListTracker_device, int nodes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			int SC_size;
			unsigned int s = saturationColorMask[index];
			SC_size = BitCount(s);
			printf("node: %d  SCMask:%d  SC_size:%d\n", index, s, SC_size);
		}
	}
}

///*
//* computeSCMask is not in use currently
//*/
//__global__ void computeSCMask(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes, int* saturationColorMask, int* vertexcolor) {
//	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nodes; index += blockDim.x * gridDim.x)
//	{
//		saturationColorMask[index] = computeSC(AdjListFull_device, AdjListTracker_device, index, vertexcolor);
//
//	}
//}

/*
* deleteEdgeFromAdj fn marks the del edges in adjacency list
*/
__global__ void deleteEdgeFromAdj(changeEdge* allChange_Del_device, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
		//int edge_weight = allChange_Del_device[index].edge_wt;

		//mark the edge as deleted in Adjlist
		for (int j = AdjListTracker_device[node_2]; j < AdjListTracker_device[node_2 + 1]; j++) {
			if (AdjListFull_device[j].col == node_1) {
				AdjListFull_device[j].flag = -1; //flag set -1 to indicate deleted
				//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
			}

		}
		for (int j = AdjListTracker_device[node_1]; j < AdjListTracker_device[node_1 + 1]; j++) {
			if (AdjListFull_device[j].col == node_2) {
				AdjListFull_device[j].flag = -1;
				//printf("inside del outedge: %d %d %d \n", node_1, node_2, edge_weight);
			}
		}
	}
}



/// <summary>
/// deleteEdge fn processes del edges and color if required.
/// ****fn updated::  now recolors only when there is no other ngbr with vertexcolor[other_node].
/// Else does not recolor/does not mark affected. So, it decreases number of affected vertices.
/// The color quality depends on prev color as we dont select min available color, we only check if the other endpoints color can be used.
/// </summary>
/// <param name="allChange_Del_device"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="totalChangeEdges_Del"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_del"></param>
/// <param name="change"></param>
/// <param name="SCmaskArrayElement"></param>
/// <returns></returns>
__global__ void deleteEdge(/*curandState* states,*/ changeEdge* allChange_Del_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_del, int* change, int SCmaskArrayElement) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		int targeted_node = 0, other_node = 0;
		if (vertexcolor[allChange_Del_device[index].node1] > vertexcolor[allChange_Del_device[index].node2]) {
			targeted_node = allChange_Del_device[index].node1;
			other_node = allChange_Del_device[index].node2;
		}
		else {
			targeted_node = allChange_Del_device[index].node2;
			other_node = allChange_Del_device[index].node1;
		}

		for (int j = AdjListTracker_device[targeted_node]; j < AdjListTracker_device[targeted_node + 1]; j++) {
			if (AdjListFull_device[j].flag != -1 && vertexcolor[AdjListFull_device[j].col] == vertexcolor[other_node])
			{
				//if vertexcolor[other_node] is used for other ngbr of targeted_node then return without recoloring
				return;
			}
		}
		//if vertexcolor[other_node] is not used for other ngbr of targeted_node then recolor targeted_node with vertexcolor[other_node]
		previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
		vertexcolor[targeted_node] = vertexcolor[other_node];
		affected_del[targeted_node] = 1;
		*change = 1;
	}
}

/*
* insEdge fn processes ins edges and color if required
*/

/// <summary>
/// ****fn updated::  now target is selected by comparing vertex id only. 
/// mask is 32 bit, uses 2 new fns computeSC_32bit and FirstZeroBit_32bit
/// can handle any large color id
/// </summary>
/// <param name="allChange_Ins_device"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="totalChangeEdges_Ins"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="change"></param>
/// <param name="SCmaskArrayElement"></param>
/// <returns></returns>
__global__ void insEdge(changeEdge* allChange_Ins_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Ins, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* change, int SCmaskArrayElement) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Ins; index += blockDim.x * gridDim.x)
	{
		if (vertexcolor[allChange_Ins_device[index].node1] == vertexcolor[allChange_Ins_device[index].node2])
		{
			//simply selecting the endpoint with larger vertex id as target when both the endpoints have same color
			int targeted_node = allChange_Ins_device[index].node1 > allChange_Ins_device[index].node2 ? allChange_Ins_device[index].node1 : allChange_Ins_device[index].node2;

			int itr = 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
			while (1) {
				unsigned int SCMask = computeSC_32bit(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, itr);
				if (SCMask < 2147483647){ //2^31 -1 = 2147483647 //int value can take upto 2,147,483,647 only
					previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
					vertexcolor[targeted_node] = FirstZeroBit_32bit(SCMask) + 31*(itr - 1);
					//printf("FirstZeroBit_32bit(SCMask): %d \n", FirstZeroBit_32bit(SCMask));
					//printf("$$$$color of targeted_node: %d becomes %d\n", targeted_node, vertexcolor[targeted_node]);
					affected_marked[targeted_node] = 1;
					*change = 1;
					break;
				}
				itr = itr + 1;
			}
			//printf("Itr: %d \n", itr);
		}

	}
}

/*
* findEligibleNeighbors is the combination of check conflict and find affected neighbor
* it selects neighbor vertices to color
*/

/// <summary>
/// 
/// </summary>
/// <param name="affectedNodeList"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="vertexcolor"></param>
/// <param name="counter"></param>
/// <param name="change"></param>
/// <returns></returns>
__global__ void findEligibleNeighbors(int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* previosVertexcolor, int* vertexcolor, int* counter, int* change) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
	{
		int node = affectedNodeList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++)
		{
			int targeted_node = 0;
			//Conflict resolution step
			if (AdjListFull_device[j].flag > -1 && vertexcolor[node] == vertexcolor[AdjListFull_device[j].col] && node != AdjListFull_device[j].col) //node != AdjListFull_device[j].col removes self loop
			{
				//vertex with larger id is selected for recoloring
				targeted_node = AdjListFull_device[j].col > node ? AdjListFull_device[j].col : node;
				//printf("target: %d\n", targeted_node);
				int itr = 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
				while (1) {
					unsigned int SCMask = computeSC_32bit(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, itr);
					if (SCMask < 2147483647) { //2^31 -1 = 2147483647 //int value can take upto 2,147,483,647 only
						previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
						vertexcolor[targeted_node] = FirstZeroBit_32bit(SCMask) + 31 * (itr - 1);
						//printf("FirstZeroBit_32bit(SCMask): %d \n", FirstZeroBit_32bit(SCMask));
						//printf("$$$$color of targeted_node: %d becomes %d\n", targeted_node, vertexcolor[targeted_node]);
						affected_marked[targeted_node] = 1;
						//printf("marked vertex: %d color changed to: %d inside conflict chk\n", targeted_node, vertexcolor[targeted_node]);
						*change = 1;
						break;
					}
					itr = itr + 1;
				}
				//printf("Itr: %d \n", itr);




				//affected_marked[targeted_node] = 1;
				//printf("AdjListFull_device[j].flag of %d = %d\n", AdjListFull_device[j].col, AdjListFull_device[j].flag);
			}
			//mark neighbor: when previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col] that means  node has directed edge toward j, 
			//or the node can affect the neighbor(this checks the total order condition)
			else if (AdjListFull_device[j].flag > -1 && previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col])
			{

				targeted_node = AdjListFull_device[j].col;
				
				//**process 1: find min available color not equal to current color
				//int itr = 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
				//while (1) {
				//	unsigned int SCMask = computeSC_32bit(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, itr);
				//	if (SCMask < 2147483647) { //2^31 -1 = 2147483647 //int value can take upto 2,147,483,647 only
				//		int smallest_available_color = FirstZeroBit_32bit(SCMask) + 31 * (itr - 1);
				//		if (vertexcolor[targeted_node] > smallest_available_color)
				//		{
				//			previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
				//			vertexcolor[targeted_node] = smallest_available_color;
				//			//printf("FirstZeroBit_32bit(SCMask): %d \n", FirstZeroBit_32bit(SCMask));
				//			//printf("$$$$color of targeted_node: %d becomes %d\n", targeted_node, vertexcolor[targeted_node]);
				//			affected_marked[targeted_node] = 1;
				//			printf("marked vertex: %d  inside ngbr chk\n", targeted_node);
				//			*change = 1;
				//		}
				//		break;
				//	}
				//	itr = itr + 1;
				//}



				//**process 2: select other vertex color to recolor this vertex. if not available dont color
				//printf("&&&&targeted node: %d color: %d other nodes prev color: %d", targeted_node, vertexcolor[targeted_node], previosVertexcolor[node]);
				int color_flag = 0;
				for (int i = AdjListTracker_device[targeted_node]; i < AdjListTracker_device[targeted_node + 1]; i++) {
					if (AdjListFull_device[i].flag != -1 && vertexcolor[AdjListFull_device[i].col] == previosVertexcolor[node])
					{
						//if previosVertexcolor[node] is used for other ngbr of targeted_node then return without recoloring
						color_flag = 1;
						break;
					}
				}
				//if previosVertexcolor[node]is not used for other ngbr of targeted_node then recolor targeted_node with vertexcolor[other_node]
				if (color_flag == 0) {
					previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
					vertexcolor[targeted_node] = previosVertexcolor[node];
					affected_marked[targeted_node] = 1;
					*change = 1;
					//printf("@@@@targeted node: %d color changed from %d to %d", targeted_node, previosVertexcolor[targeted_node], vertexcolor[targeted_node]);
				}
			}

			//test for RMAT18 error check
			/*if (node == 3112 && AdjListFull_device[j].col == 108702)
			{
				printf("$$affected_marked[%d] = %d$$\n", AdjListFull_device[j].col, affected_marked[AdjListFull_device[j].col]);
				printf("$$vertexcolor[%d] = %d$$\n", node, vertexcolor[node]);
				printf("$$vertexcolor[%d] = %d$$\n", AdjListFull_device[j].col, vertexcolor[AdjListFull_device[j].col]);
			}*/


		}
	}
}




//NOT IN USE
/*
* recolors the selected vertices in findEligibleNeighbors fn
*/
__global__ void recolorNeighbor(int* affectedNodeList, int* vertexcolor, int* previosVertexcolor, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* counter_del, int* change, int SCmaskArrayElement) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter_del; index += blockDim.x * gridDim.x)
	{
		int targeted_node = affectedNodeList[index];
		/*int* maskArray = new int[SCmaskArrayElement];*/
		int maskArray[5];
		for (int i = 0; i < SCmaskArrayElement; i++)
		{
			maskArray[i] = 0;
		}
		computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, maskArray);
		int smallest_available_color = FirstZeroBit(maskArray, SCmaskArrayElement);
		previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
		vertexcolor[targeted_node] = smallest_available_color;
		affected_marked[targeted_node] = 1;
		*change = 1;
		delete[] maskArray;
		//printf("color of %d became:%d", targeted_node, vertexcolor[targeted_node]); //test
		//printf("affected_marked flag for %d = %d \n *change = %d\n", targeted_node, affected_marked[targeted_node], *change);
	}
}

/*
* validates the correctness of output color
*/
__global__ void validate(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes, int* vertexcolor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			//printf("\nAdj list for %d:\n", index);
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				if (AdjListFull_device[j].flag != -1 && AdjListFull_device[j].col != index && vertexcolor[AdjListFull_device[j].col] == vertexcolor[index])
				{
					printf("##Error found: %d %d", index, AdjListFull_device[j].col);
				}
			}
		}
	}
}


//used for affected_del
struct predicate
{
	__host__ __device__
		bool operator()(int x)
	{
		return x == 1;
	}
};

//used for affected_all
struct predicate2
{
	__host__ __device__
		bool operator()(int x)
	{
		return x > 0;
	}
};


#endif