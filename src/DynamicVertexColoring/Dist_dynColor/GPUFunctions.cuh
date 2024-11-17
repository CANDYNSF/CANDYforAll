#ifndef GPUFUNCTIONS_CUH
#define GPUFUNCTIONS_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "supportVC.h"
#include <math.h>
using namespace std;

#define THREADS_PER_BLOCK 1024 //we can change it

/// <summary>
/// This predicate is used with compactor function for finding vertices with priority 2. 
/// </summary>
struct predicate
{
	__host__ __device__
		bool operator()(int x)
	{
		return x == 2;
	}
};
/// <summary>
/// This predicate is used with compactor function for finding vertices with affected_marked status 1
/// </summary>
struct predicate_findAffected
{
	__host__ __device__
		bool operator()(int x)
	{
		return x == 1;
	}
};

struct predicate_findRegion3InsEdges
{
	__host__ __device__
		bool operator()(changeEdge x)
	{
		return x.inst == 3;
	}
};

/// <summary>
/// Marks the deleted edges in the adjacency list
/// </summary>
/// <param name="allChange_Del_device"></param>
/// <param name="totalChangeEdges_Del"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <returns></returns>
__global__ void deleteEdgeFromAdj(changeEdge* allChange_Del_device, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
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
/// Assign priority 2 to all this part border vertices
/// </summary>
/// <param name="borderVertexList"></param>
/// <param name="total_borderVertex"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="priority_d"></param>
/// <param name="affected_marked"></param>
/// <returns></returns>
__global__ void assignPriority2ThisPartBorder(int* borderVertexList, int total_borderVertex, ColWt* AdjListFull_device, int* AdjListTracker_device, int* priority_d, int* affected_marked) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_borderVertex; index += blockDim.x * gridDim.x)
	{
		int node = borderVertexList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
			int ngbr = AdjListFull_device[j].col;
			if (priority_d[ngbr] == 1) {
				priority_d[ngbr] = 2;
				affected_marked[ngbr] = 1;
			}
		}
	}
}


/// <summary>
/// Assigns priority 3 to neighbors of border and ghost vertices
/// </summary>
/// <param name="borderVertexList"></param>
/// <param name="total_borderVertex"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="totalThisPartVertices"></param>
/// <param name="priority_d"></param>
/// <returns></returns>
__global__ void assignPriority3(int* borderVertexList, int total_borderVertex, ColWt* AdjListFull_device, int* AdjListTracker_device, int* priority_d) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_borderVertex; index += blockDim.x * gridDim.x)
	{
		int node = borderVertexList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
			int ngbr = AdjListFull_device[j].col;
			if (priority_d[ngbr] == 1) {
				priority_d[ngbr] = 3;
			}
		}
	}
}

/// <summary>
/// This function creates a 32 bit mask, where ith bit is set if one of the adjacent vertex has color i
/// </summary>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="node"></param>
/// <param name="vertexcolor"></param>
/// <returns></returns>
__device__ int computeSC(ColWt* AdjListFull_device, int* AdjListTracker_device, int node, int* vertexcolor) {
	int saturationColorMask = 0;
	for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
		if (AdjListFull_device[j].flag != -1)
		{
			saturationColorMask = saturationColorMask | int(exp2(double(vertexcolor[AdjListFull_device[j].col])));
		}
	}
	return saturationColorMask;
}

/// <summary>
/// It returns number of 1 in mask
/// </summary>
/// <param name="mask"></param>
/// <returns></returns>
__device__ int BitCount(unsigned int mask)
{
	unsigned int uCount;
	uCount = mask
		- ((mask >> 1) & 033333333333)
		- ((mask >> 2) & 011111111111);
	return ((uCount + (uCount >> 3))
		& 030707070707) % 63;
}

/// <summary>
/// It returns the position of 1st zero (from right) in the mask => 0 in mask meaans the color is not assigned to any neighbor vertices
/// </summary>
/// <param name="i"></param>
/// <returns></returns>
__device__ int FirstZeroBit(int i)
{
	i = ~i;
	return BitCount((i & (-i)) - 1);
}


__device__ int computeSC_32bit(ColWt* AdjListFull_device, int* AdjListTracker_device, int node, int* vertexcolor, int itr) {
	int saturationColorMask = 0;
	for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
		if (AdjListFull_device[j].flag != -1 && (vertexcolor[AdjListFull_device[j].col] < 31 * itr) && (vertexcolor[AdjListFull_device[j].col] >= 31 * (itr - 1)))
		{
			//select colors x if 31 * (itr - 1) <= x < 31 * itr
			saturationColorMask = saturationColorMask | int(exp2(double(vertexcolor[AdjListFull_device[j].col] % 31))); //as signed int can take upto 2^31 -1
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




/// <summary>
/// processes delete edges in parallel and marks affected vertices
/// /// ****fn updated::  now recolors only when there is no other ngbr with vertexcolor[other_node].
/// Else does not recolor/does not mark affected. So, it decreases number of affected vertices.
/// The color quality depends on prev color as we dont select min available color, we only check if the other endpoints color can be used.
/// </summary>
/// <param name="allChange_Del_device"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="totalChangeEdges_Del"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="change"></param>
/// <param name="priority_d"></param>
/// <returns></returns>
__global__ void deleteEdge(changeEdge* allChange_Del_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* change, int* priority_d) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		int node1 = allChange_Del_device[index].node1;
		int node2 = allChange_Del_device[index].node2;
		int targeted_node = 0, other_node = 0;;
		if (priority_d[node1] == 3 && priority_d[node2] == 3) { //avoid processing edges with both endpoints having priority 3
			continue;
		}
		//if priority of n1 and n2 are different we select the lower priority vertex as target if it has higher vertex color than the other end
		if (priority_d[node1] < priority_d[node2] && vertexcolor[node1] > vertexcolor[node2]) {
			targeted_node = node1;
			other_node = node2;
		}
		else if (priority_d[node2] < priority_d[node1] && vertexcolor[node2] > vertexcolor[node1]) {
			targeted_node = node2;
			other_node = node1;
		}
		else if(priority_d[node1] == priority_d[node2]){
			//if priority of n1 and n2 are same we select the vertex with higher color id as the higher color id can be reduced only
			targeted_node = vertexcolor[node1] > vertexcolor[node2] ? node1 : node2;
			other_node = targeted_node == node1 ? node2 : node1;
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
		affected_marked[targeted_node] = 1;
		*change = 1;
	}
}

/// <summary>
/// ****fn updated::  now target is selected by comparing vertex id only. 
/// mask is 32 bit, uses 2 new fns computeSC_32bit and FirstZeroBit_32bit
/// can handle any large color id
/// Overview:
/// Avoids processing edges with both endpoints having priority 3
/// If priority of n1 and n2 are different we select the lower priority vertex as target
/// If priority of n1 and n2 are same but not 3 then targeted node is selected by comparing vertex id only. vertex with higher globalID is selected.
/// </summary>
/// <param name="allChange_Ins_device"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="totalChangeEdges_Ins"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="change"></param>
/// <param name="priority_d"></param>
/// <returns></returns>
__global__ void insEdge(changeEdge* allChange_Ins_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Ins, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* change, int* priority_d, int* globalID_d) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Ins; index += blockDim.x * gridDim.x)
	{
		int node1 = allChange_Ins_device[index].node1;
		int node2 = allChange_Ins_device[index].node2;
		if (priority_d[node1] == 3 && priority_d[node2] == 3) { //avoid processing edges with both endpoints having priority 3
			allChange_Ins_device[index].inst = 3; //making ins edges with both endpoint priority 3
			continue;
		}
		if (vertexcolor[node1] == vertexcolor[node2])
		{
			
			int targeted_node;
			if (priority_d[node1] != priority_d[node2]) {
				//if priority of n1 and n2 are different we select the lower priority vertex as target
				targeted_node = priority_d[node1] < priority_d[node2] ? node1 : node2;
			}
			else {
				//if priority of n1 and n2 are same but not 3 then targeted node is selected by comparing vertex id only. 
				//vertex with higher globalID is selected.
				targeted_node = globalID_d[node1] > globalID_d[node2] ? node1 : node2;
			}
			int itr = vertexcolor[targeted_node] / 31 + 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
			while (1) {
				unsigned int SCMask = computeSC_32bit(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, itr);
				if (SCMask < 2147483647) { //2^31 -1 = 2147483647 //int value can take upto 2,147,483,647 only
					previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
					vertexcolor[targeted_node] = FirstZeroBit_32bit(SCMask) + 31 * (itr - 1);
					//printf("FirstZeroBit_32bit(SCMask): %d \n", FirstZeroBit_32bit(SCMask));
					//printf("$$$$color of targeted_node: %d becomes %d\n", targeted_node, vertexcolor[targeted_node]);
					affected_marked[targeted_node] = 1;
					*change = 1;
					break;
				}
				itr = itr + 1;
			}
		}

	}
}


/// <summary>
/// check conflicts and select vertex with higher Global ID.
/// check neighbors of all affected vertices and select eligible neighbors for recoloring.
/// There is no chance of a conflict between vertex a and b if they are in different region as the changes are processed separately in different regions
/// E.g. When region 1 and 2 are processed in parallel region 3 is stable and it works as region 3 is in between of region 1 and 2
/// **fn updated:: for conflict resolution the target is colored using smalled available color. (supports large color)
/// A ngbr vertex v is recolored when previosVertexcolor[node]is not used for other ngbrs of v and previosVertexcolor[node] less than vertexcolor[v]
/// </summary>
/// <param name="affectedNodeList"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="vertexcolor"></param>
/// <param name="counter"></param>
/// <param name="priority_d"></param>
/// <param name="globalID_d"></param>
/// <param name="change"></param>
/// <returns></returns>
__global__ void findEligibleNeighbors(int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* previosVertexcolor, int* vertexcolor, int* counter, int* priority_d, int* globalID_d, int* change) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
	{
		int node = affectedNodeList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++)
		{
			int targeted_node = 0;
			int ngbr = AdjListFull_device[j].col;
			//Conflict resolution step
			if (AdjListFull_device[j].flag > -1 && vertexcolor[node] == vertexcolor[ngbr] && node != ngbr) //node != AdjListFull_device[j].col removes self loop
			{
				//There will be no conflict between two vertices with different priority as we don't change color for priority 3 vertices while processing priority 1 and 2 vertices
				//and dont change color for priority 1 and 2 while processing priority 3
				//vertex with larger id is selected for recoloring
				targeted_node = globalID_d[node] > globalID_d[ngbr] ? node : ngbr; //as globalID is unique, there will be no conflict.
				//printf("target: %d\n", targeted_node);
				int itr = vertexcolor[targeted_node] / 31 + 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
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
			}

			//mark neighbor: when previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col] that means  node has directed edge toward j, 
			//or the node can affect the neighbor(this checks the total order condition)
			//else if (AdjListFull_device[j].flag > -1 && previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col] && priority_d[node] == priority_d[AdjListFull_device[j].col]) 
			//{
			//	//**priority_d[node] == priority_d[AdjListFull_device[j].col ensures the color quality correction is done only when node and ngbr are in same region
			//	targeted_node = AdjListFull_device[j].col;
			//	//**process 2: select other vertex color to recolor this vertex. if not available dont color
			//	//printf("&&&&targeted node: %d color: %d other nodes prev color: %d", targeted_node, vertexcolor[targeted_node], previosVertexcolor[node]);
			//	int color_flag = 0;
			//	for (int i = AdjListTracker_device[targeted_node]; i < AdjListTracker_device[targeted_node + 1]; i++) {
			//		if (AdjListFull_device[i].flag != -1 && vertexcolor[AdjListFull_device[i].col] == previosVertexcolor[node])
			//		{
			//			//if previosVertexcolor[node] is used for other ngbr of targeted_node then return without recoloring
			//			color_flag = 1;
			//			break;
			//		}
			//	}
			//	//if previosVertexcolor[node]is not used for other ngbr of targeted_node then recolor targeted_node with vertexcolor[other_node]
			//	if (color_flag == 0) {
			//		previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
			//		vertexcolor[targeted_node] = previosVertexcolor[node];
			//		affected_marked[targeted_node] = 1;
			//		*change = 1;
			//		//printf("@@@@targeted node: %d color changed from %d to %d", targeted_node, previosVertexcolor[targeted_node], vertexcolor[targeted_node]);
			//	}
			//}
		}
	}
}




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
/// <param name="iteration"></param>
/// <param name="max_iteration"></param>
/// <returns></returns>
__global__ void recolorNeighbors(int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* previosVertexcolor, int* vertexcolor, int* counter, int* change, int iteration, int max_iteration) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
	{
		int node = affectedNodeList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++)
		{
			int targeted_node = 0;

			//mark neighbor: when previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col] that means  node has directed edge toward j, 
			//or the node can affect the neighbor(this checks the total order condition)
			//iteration < max_iteration is the user control over the number of iterations
			if (AdjListFull_device[j].flag > -1 && previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col])
			{

				targeted_node = AdjListFull_device[j].col;


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
		}
	}
}





//NOT IN USE
/// <summary>
/// //NOT IN USE
/// it recolors the eligible neighbors selected by findEligibleNeighbors method
/// </summary>
/// <param name="affectedNodeList"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="counter_del"></param>
/// <param name="change"></param>
/// <returns></returns>
//__global__ void recolorNeighbor(int* affectedNodeList, int* vertexcolor, int* previosVertexcolor, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* counter, int* change) {
//
//	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
//	{
//		//test
//		/*if (index == 0)
//		{
//			printf("## number of eligible vertices: %d", *counter);
//		}*/
//		int targeted_node = affectedNodeList[index];
//		int SCMask = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
//		int smallest_available_color = FirstZeroBit(SCMask);
//		previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
//		vertexcolor[targeted_node] = smallest_available_color;
//		affected_marked[targeted_node] = 1;
//		*change = 1;
//
//		//printf("color of %d became:%d", targeted_node, vertexcolor[targeted_node]);
//		//printf("affected_marked flag for %d = %d \n *change = %d\n", targeted_node, affected_marked[targeted_node], *change);
//	}
//}

/// <summary>
/// ****fn updated::  now target is selected by comparing vertex id only. 
/// process the inserted edges for which both the end points have priority 3
/// </summary>
/// <param name="Region3InsEdgeIDList"></param>
/// <param name="allChange_Ins_device"></param>
/// <param name="vertexcolor"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="counter"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="change"></param>
/// <param name="globalID_d"></param>
/// <returns></returns>
__global__ void insEdgeRegion3(int* Region3InsEdgeIDList, changeEdge* allChange_Ins_device, int* vertexcolor, int* previosVertexcolor, int* counter, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* change, int* globalID_d) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
	{
		int node1 = allChange_Ins_device[Region3InsEdgeIDList[index]].node1;
		int node2 = allChange_Ins_device[Region3InsEdgeIDList[index]].node2;
		int targeted_node;
		if (vertexcolor[node1] == vertexcolor[node2])
		{
			//vertex with higher globalID is selected.
			targeted_node = globalID_d[node1] > globalID_d[node2] ? node1 : node2;
			
			int itr = vertexcolor[targeted_node] / 31 + 1; //As in case of insertion color id always increases from the prev color id in general, we can take itr =  vertexcolor[targeted_node]/31 + 1; to decrease time
			while (1) {
				unsigned int SCMask = computeSC_32bit(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor, itr);
				if (SCMask < 2147483647) { //2^31 -1 = 2147483647 //int value can take upto 2,147,483,647 only
					previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
					vertexcolor[targeted_node] = FirstZeroBit_32bit(SCMask) + 31 * (itr - 1);
					//printf("FirstZeroBit_32bit(SCMask): %d \n", FirstZeroBit_32bit(SCMask));
					//printf("$$$$color of targeted_node: %d becomes %d\n", targeted_node, vertexcolor[targeted_node]);
					affected_marked[targeted_node] = 1;
					*change = 1;
					break;
				}
				itr = itr + 1;
			}
		}
	}
}

/// <summary>
/// NOT IN USE
/// Check conflict and eligible neighbors in Region 3
/// There is no chance of a conflict between vertex a and b if they are in different region as the changes are processed separately in different regions
/// E.g. when region 3 is processed, region 1 and 2 are stable. 
/// When region 1 and 2 are processed in parallel region 3 is stable and it works as region 3 is in between of region 1 and 2
/// </summary>
/// <param name="affectedNodeList"></param>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="affected_marked"></param>
/// <param name="previosVertexcolor"></param>
/// <param name="vertexcolor"></param>
/// <param name="counter"></param>
/// <param name="priority_d"></param>
/// <param name="globalID_d"></param>
/// <returns></returns>
//__global__ void findEligibleNeighborsInRegion3(int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* previosVertexcolor, int* vertexcolor, int* counter, int* priority_d, int* globalID_d) {
//	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
//	{
//		int node = affectedNodeList[index]; //here node is in region 3
//		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++)
//		{
//			int ngbr = AdjListFull_device[j].col;
//
//			if (AdjListFull_device[j].flag == -1 || priority_d[ngbr] != 3) {
//				//if the edge is marked as deleted, then skip 
//				//OR if ngbr is not in Region 3(priority 3) then skip
//				continue;
//			}
//
//			//we process further only when node and ngbr both in region 3
//			//Conflict resolution step:
//			if (vertexcolor[node] == vertexcolor[ngbr])
//			{
//				//if there is color conflict => node and ngbr should both be with priority 3 as vertices with priority < 3 are not parallely processed in insEdgeRegion3 method
//				//selecting vertex with lower global ID (we need to choose uniquely, random selection will not work)
//				int targeted_node = globalID_d[node] < globalID_d[ngbr] ? node : ngbr;
//				affected_marked[targeted_node] = 1;
//				//printf("AdjListFull_device[j].flag of %d = %d\n", AdjListFull_device[j].col, AdjListFull_device[j].flag);
//			}
//			//mark neighbor step: when previosVertexcolor[node] < vertexcolor[ngbr] that means  node has directed edge toward ngbr, 
//			//or the node can affect the neighbor(this checks the total order condition)
//			if (previosVertexcolor[node] < vertexcolor[ngbr])
//			{
//				affected_marked[ngbr] = 1;
//			}
//		}
//	}
//}

/// <summary>
/// validates vertex coloring
/// </summary>
/// <param name="AdjListFull_device"></param>
/// <param name="AdjListTracker_device"></param>
/// <param name="total_vertex"></param>
/// <param name="vertexcolor"></param>
/// <returns></returns>
__global__ void validate(ColWt* AdjListFull_device, int* AdjListTracker_device, int total_vertex, int* vertexcolor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < total_vertex; index++)
		{
			//printf("\nAdj list for %d:\n", index);
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				if (AdjListFull_device[j].flag != -1 && vertexcolor[AdjListFull_device[j].col] == vertexcolor[index] && index != AdjListFull_device[j].col)
				{
					printf("##Error found: %d %d", index, AdjListFull_device[j].col);
				}
			}
		}
	}
}
#endif