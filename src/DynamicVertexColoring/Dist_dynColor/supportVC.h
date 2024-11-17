#ifndef SUPPORTVC_H
#define SUPPORTVC_H
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <chrono>
#include <map>
#include <set>
#include <mpi.h>
//#include "GPUFunctions.cuh"

using namespace std;
using namespace std::chrono;

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }
/// <summary>
/// ColWt is the type for each element in the Adjacency List
/// </summary>
struct ColWt {
    int col;
    int flag = 0; //default 0, deleted -1
};
typedef vector<ColWt> ColList; //ColList is the type of each 1D list in the 2D Adjacency List 

/// <summary>
/// changeEdge is used for change edges. change edges should be in format "node1 node2 insertion_status"
/// </summary>
struct changeEdge {
    int node1;
    int node2;
    int inst;
};


/// <summary>
/// This function reads the partitionAll.txt and stores all partition IDs for all the vertices
/// It also stores the local vertices 
/// </summary>
/// <param name="myfile"></param>
/// <param name="PartitionID_all"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="totalLocalVertices"></param>
/// <param name="rank"></param>
void read_PartitionID_AllVertices(char* myfile, vector<int>& PartitionID_all, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* totalLocalVertices, int rank)
{
    FILE* graph_file;
    char line[128];
    int localID = 0, globalID = 0;
    graph_file = fopen(myfile, "r");
    while (fgets(line, 128, graph_file) != NULL)
    {
        int partID;
        sscanf(line, "%d", &partID);
        PartitionID_all.push_back(partID);
        
        if (partID == rank) {
            Global2LocalMap.insert(make_pair(globalID, localID));//It assigns a local vertex ID for all vertices in a partition
            Local2GlobalMap.insert(make_pair(localID, globalID));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
            localID++;
        }
        globalID++;
    }
    fclose(graph_file);
    *totalLocalVertices = localID;

    return;
}

/// <summary>
/// 
/// </summary>
/// <param name="n1"></param>
/// <param name="n2"></param>
/// <param name="AdjList"></param>
/// <param name="PartitionID_all"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="total_vertex"></param>
/// <param name="priority"></param>
/// <param name="rank"></param>
void read_graphEdges_helper(int n1, int n2, vector<ColList>& AdjList, vector<int>& PartitionID_all, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority, int rank)
{
    int first_endpoint, second_endpoint;
    if (PartitionID_all.at(n1) == rank) {//if n1 is in this part
        first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
        ColWt c1, c2;
        if (PartitionID_all.at(n2) == rank) {
            second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID
            //c1.col = first_endpoint;
            //c1.flag = 0;
            c2.col = second_endpoint;
            //c2.flag = 0;
            AdjList.at(first_endpoint).push_back(c2);
            //AdjList.at(second_endpoint).push_back(c1);
        }
        else if (Global2LocalMap.find(n2) == Global2LocalMap.end()) { //When n2 is not from this part and not in map
            Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
            Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
            priority.push_back(2); //1-hop Ghost vertex has priority 2
            //priority.at(first_endpoint) = 2; //as first_endpoint is bv
            
            //cout << "rank"<<rank<<"total vertex:" << *total_vertex<<endl;
            ColList clmL;
            c1.col = first_endpoint;
            //c1.flag = 0;
            clmL.push_back(c1);
            //adding *total_vertex-th element in AdjList
            AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
            c2.col = *total_vertex; //Getting local ID
            //c2.flag = 0;
            AdjList.at(first_endpoint).push_back(c2);
            *total_vertex = *total_vertex + 1;
        }
        else { //When n2 is not from this part but already in map. It means priority of n2 is already assigned to 2 
            second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID
            c1.col = first_endpoint;
            //c1.flag = 0;
            c2.col = second_endpoint;
            //c2.flag = 0;
            AdjList.at(first_endpoint).push_back(c2);
            AdjList.at(second_endpoint).push_back(c1);
            //if (PartitionID_all.at(n2) != rank) {//if n2 is in different part(n2  is ghost), but n2 is in map => n2 is ghost vertex and priority is already assigned to 2
            //    priority.at(first_endpoint) = 2; //first_endpoint is bv as n2 is in other part
            //}

        }
    }
}

/// <summary>
/// Reads graph edges and store in adjacency list. 
/// It assigns priority 2 to all 1-hop ghosts only
/// </summary>
/// <param name="AdjList"></param>
/// <param name="PartitionID_all"></param>
/// <param name="myfile"> is the edge file</param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="total_vertex"></param>
/// <param name="priority"></param>
/// <param name="rank">holds this part ID</param>
void read_graphEdges(vector<ColList>& AdjList, vector<int>& PartitionID_all, char* myfile, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority, int rank)
{
    //cout << "Reading input graph..." << endl;
    //cout << "inside read_graphEdges. total vertex till now:" << *total_vertex << endl;
    FILE* graph_file;
    char line[128];
    graph_file = fopen(myfile, "r");
    
    while (fgets(line, 128, graph_file) != NULL)
    {
        int n1, n2;
        sscanf(line, "%d %d", &n1, &n2); //our input graph has wt. But we don't need it. So we will ignore wt.
        read_graphEdges_helper(n1, n2, AdjList, PartitionID_all, Global2LocalMap, Local2GlobalMap, total_vertex, priority, rank);
        read_graphEdges_helper(n2, n1, AdjList, PartitionID_all, Global2LocalMap, Local2GlobalMap, total_vertex, priority, rank);
        
    }

    fclose(graph_file);
    //Test
    /*int t = 0;
    for (int i = 0; i < *total_vertex; i++) {
        t = t + AdjList.at(i).size();
    }
    cout << "rank: " << rank <<"AFTER 1st E read: total vertices:"<< *total_vertex << "total this part edges:" << (t/2) << endl;*/
    return;
}




/// <summary>
/// read_Ins checks all the ins edges crossing the partition borders (edges which has only one endpoint in this part)
/// If for an ins edge (u,v), u is in this part and v is in another part, read_Ins function assigns priority 2 for v. 
/// So, there will be no ins edge with one end point local and another 2-hop ngbr.
/// This function DOES NOT add the ins edge in any edge list, just assigns priority 2.
/// </summary>
/// <param name="priority"></param>
/// <param name="total_vertex"></param>
/// <param name="myfile"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="PartitionID_all"></param>
/// <param name="rank"></param>
void read_Ins(vector<ColList>& AdjList, vector<int>& priority, int* total_vertex, char* myfile, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, vector<int>& PartitionID_all, int rank)
{
    //cout << "Reading input changed edges data..." << endl;
    FILE* graph_file;
    char line[128];
    graph_file = fopen(myfile, "r");
    while (fgets(line, 128, graph_file) != NULL)
    {
        int n1, n2, inst_status;
        sscanf(line, "%d %d %d", &n1, &n2, &inst_status); //edge wt is there in input file. But we don't need it. So we will ignore it.
        if (inst_status == 1) //we handle only edge insertion of this part here
        {
            if ((PartitionID_all.at(n1) == rank) && (PartitionID_all.at(n2) != rank)) {
                if (Global2LocalMap.find(n2) == Global2LocalMap.end()) { //When n2 is not from this part and not in map
                    Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
                    Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
                    priority.push_back(2);
                    *total_vertex = *total_vertex + 1;
                    AdjList.resize(AdjList.size() + 1); //add one element in adj list for the new vertex
                }
                /*else {
                    priority.at(Global2LocalMap.find(n2)->second) = 2;
                }*/
            }
            else if ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) == rank)) {
                if (Global2LocalMap.find(n1) == Global2LocalMap.end()) { //When n2 is not from this part and not in map
                    Global2LocalMap.insert(make_pair(n1, *total_vertex));//It assigns a local vertex ID
                    Local2GlobalMap.insert(make_pair(*total_vertex, n1));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
                    priority.push_back(2);
                    *total_vertex = *total_vertex + 1;
                    AdjList.resize(AdjList.size() + 1); //add one element in adj list for the new vertex
                }
                /*else {
                    priority.at(Global2LocalMap.find(n1)->second) = 2;
                }*/
            }
        }
    }
    fclose(graph_file);
    //Test
    /*int t = 0;
    for (int i = 0; i < *total_vertex; i++) {
        t = t + AdjList.at(i).size();
    }
    cout << "rank: " << rank << "AFTER read_Ins: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
    for (int i = 0; i < *total_vertex; i++) {
        cout << "[ Priority of " << Local2GlobalMap.find(i)->second << " is: " << priority.at(i) << "]";
    }
    cout << endl;*/
    return;
}

/// <summary>
/// Helper function of read_2hopghosts
/// </summary>
/// <param name="n1"></param>
/// <param name="n2"></param>
/// <param name="AdjList"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="total_vertex"></param>
/// <param name="priority"></param>
void read_2hopghosts_helper(int n1, int n2, vector<ColList>& AdjList, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority)
{
    int first_endpoint, second_endpoint;
    if (Global2LocalMap.find(n1) != Global2LocalMap.end() && Global2LocalMap.find(n2) != Global2LocalMap.end()) {//if n1, n2 both in map already
        first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
        second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n1
        ColWt c1, c2;
        //test
        /*if (n2 == 6)
        {
            cout << "n1:" << n1 << "first_endpoint" << first_endpoint << "n2:" <<n2 << "second_endpoint" << second_endpoint << "priority_size: " << priority.size();
            cout << "priority.at(first_endpoint):" << priority.at(first_endpoint);
            cout << "priority.at(second_endpoint):" << priority.at(second_endpoint);
        }*/
        if (priority.at(first_endpoint) == 2 && priority.at(second_endpoint) >= 2) {
            c1.col = first_endpoint;
            c2.col = second_endpoint;
            AdjList.at(first_endpoint).push_back(c2);
            AdjList.at(second_endpoint).push_back(c1);
        }
        else if (priority.at(first_endpoint) > 2 && priority.at(second_endpoint) == 2) {
            c1.col = first_endpoint;
            c2.col = second_endpoint;
            AdjList.at(first_endpoint).push_back(c2);
            AdjList.at(second_endpoint).push_back(c1);
        }
    }
    else if (Global2LocalMap.find(n1) != Global2LocalMap.end() && Global2LocalMap.find(n2) == Global2LocalMap.end()) {//if n1 in map, but n2 not in map
        first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
        if (priority.at(first_endpoint) == 2) {
            Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
            Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
            priority.push_back(4); //2-hop Ghost vertex has priority 4
            ColWt c1, c2;
            ColList clmL;
            c1.col = first_endpoint;
            clmL.push_back(c1);
            AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
            c2.col = *total_vertex; //local ID of n2
            AdjList.at(first_endpoint).push_back(c2);
            *total_vertex = *total_vertex + 1;
        }
    }
    else if (Global2LocalMap.find(n1) == Global2LocalMap.end() && Global2LocalMap.find(n2) != Global2LocalMap.end()) {//if n2 in map, but n1 not in map
        second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n1
        if (priority.at(second_endpoint) == 2) {
            Global2LocalMap.insert(make_pair(n1, *total_vertex));//It assigns a local vertex ID
            Local2GlobalMap.insert(make_pair(*total_vertex, n1));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
            priority.push_back(4); //2-hop Ghost vertex has priority 4
            ColWt c1, c2;
            ColList clmL;
            c1.col = second_endpoint;
            clmL.push_back(c1);
            AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
            c2.col = *total_vertex; //local ID of n2
            AdjList.at(second_endpoint).push_back(c2);
            *total_vertex = *total_vertex + 1;
        }
    }
}
/// <summary>
/// Read all edges b/w two priority 2 vertices or edges b/w priority 2 and 2-hop ghosts
/// It assigns priority 4 to 2-hop ghosts
/// </summary>
/// <param name="AdjList"></param>
/// <param name="PartitionID_all"></param>
/// <param name="myfile"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="total_vertex"></param>
/// <param name="priority"></param>
/// <param name="rank"></param>
void read_2hopghosts(vector<ColList>& AdjList, vector<int>& PartitionID_all, char* myfile, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority, int rank)
{
    //cout << "Reading input graph..." << endl;
    //cout << "inside read_graphEdges. total vertex till now:" << *total_vertex << endl;
    FILE* graph_file;
    char line[128];
    graph_file = fopen(myfile, "r");

    while (fgets(line, 128, graph_file) != NULL)
    {
        int n1, n2;
        sscanf(line, "%d %d", &n1, &n2); //our input graph has wt. But we don't need it. So we will ignore wt.
        if (PartitionID_all.at(n1) != rank && PartitionID_all.at(n2) != rank) {
            read_2hopghosts_helper(n1, n2, AdjList, Global2LocalMap, Local2GlobalMap, total_vertex, priority);
        }
    }
    fclose(graph_file);
    ////Test
    //int t = 0;
    //for (int i = 0; i < *total_vertex; i++) {
    //    t = t + AdjList.at(i).size();
    //}
    //cout << "rank: " << rank << "AFTER 1st E read: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
    return;
}


/// <summary>
/// Read changed edges (u,v), s.t. both u and v are in map already.
/// Avoid changed edges (u,v) s.t. u and v both have priority 4, as these edges
///  will be taken care of at the partitions where the endpoints are considered as priority 3.
/// </summary>
/// <param name="priority"></param>
/// <param name="myfile"></param>
/// <param name="allChange_Ins"></param>
/// <param name="allChange_Del"></param>
/// <param name="AdjList"></param>
/// <param name="Global2LocalMap"></param>
/// <param name="Local2GlobalMap"></param>
/// <param name="PartitionID_all"></param>
/// <param name="rank"></param>
void read_changEdges(vector<int>& priority, char* myfile, vector<changeEdge>& allChange_Ins, vector<changeEdge>& allChange_Del, vector<ColList>& AdjList, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex)
{
    //cout << "Reading input changed edges data..." << endl;
    FILE* graph_file;
    char line[128];
    graph_file = fopen(myfile, "r");
    int n1, n2, inst_status, first_endpoint, second_endpoint;
    changeEdge cE;
    while (fgets(line, 128, graph_file) != NULL)
    {
        
        sscanf(line, "%d %d %d", &n1, &n2, &inst_status); //edge wt is there in input file. But we don't need it. So we will ignore it.
        
        if ((Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) != Global2LocalMap.end())) //When n1 and n2 both are in the map
        {
            first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
            second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
            if (priority.at(first_endpoint) != 4 && priority.at(second_endpoint) != 4) { 
                //**we avoid the edge change between any two priority 4 vertices 
                //as these edges will be taken care of at the partitions where the endpoints are considered as priority 3.**
                cE.node1 = first_endpoint;
                cE.node2 = second_endpoint;
                cE.inst = inst_status;
                if (inst_status == 1) {
                    allChange_Ins.push_back(cE);
                    ColWt c1, c2;
                    c1.col = first_endpoint;
                    c2.col = second_endpoint;
                    AdjList.at(first_endpoint).push_back(c2);
                    AdjList.at(second_endpoint).push_back(c1);
                }
                else {
                    allChange_Del.push_back(cE);
                }
            }
        }
        else if ((Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) == Global2LocalMap.end())) //When n1 in map but n2 is not in map
        {
            first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
            if (priority.at(first_endpoint) == 2) {
                //when n2 was not in map and priority of n1 is 2, that means it is definitly an edge ins between priority 2 vertex and priority 4 vertex
                Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
                Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
                priority.push_back(4); //2-hop Ghost vertex has priority 4
                ColWt c1, c2;
                ColList clmL;
                c1.col = first_endpoint;
                clmL.push_back(c1);
                AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
                c2.col = *total_vertex; //local ID of n2
                AdjList.at(first_endpoint).push_back(c2);
                cE.node1 = first_endpoint;
                cE.node2 = *total_vertex;
                cE.inst = inst_status;
                allChange_Ins.push_back(cE); //when n2 was not in map that means it is definitly an edge ins between priority 2 vertex and priority 4 vertex
                *total_vertex = *total_vertex + 1;
            }
        }
        else if ((Global2LocalMap.find(n2) != Global2LocalMap.end()) && (Global2LocalMap.find(n1) == Global2LocalMap.end())) //When n2 in map but n1 is not in map
        {
            first_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n1
            if (priority.at(first_endpoint) == 2) {
                //when n1 was not in map and priority of n2 is 2, that means it is definitly an edge ins between priority 4 vertex and priority 2 vertex
                Global2LocalMap.insert(make_pair(n1, *total_vertex));//It assigns a local vertex ID
                Local2GlobalMap.insert(make_pair(*total_vertex, n1));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
                priority.push_back(4); //2-hop Ghost vertex has priority 4
                ColWt c1, c2;
                ColList clmL;
                c1.col = first_endpoint;
                clmL.push_back(c1);
                AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
                c2.col = *total_vertex; //local ID of n2
                AdjList.at(first_endpoint).push_back(c2);
                cE.node1 = first_endpoint;
                cE.node2 = *total_vertex;
                cE.inst = inst_status;
                allChange_Ins.push_back(cE); //when n2 was not in map that means it is definitly an edge ins between priority 2 vertex and priority 4 vertex
                *total_vertex = *total_vertex + 1;
            }
        }
    }
    fclose(graph_file);
    //Test
    /*int t = 0;
    for (int i = 0; i < *total_vertex; i++) {
        t = t + AdjList.at(i).size();
    }
    cout << "rank: " << rank << "AFTER readin_changes1: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
    for (int i = 0; i < *total_vertex; i++) {
        cout << "[ Priority of " << Local2GlobalMap.find(i)->second << " is: " << priority.at(i) << "]";
    }
    cout << endl;*/
    return;
}
















//
//
///// <summary>
///// reads changed edges and assign priority 2 for ghost vertices (in case of insertion)
///// </summary>
///// <param name="priority"></param>
///// <param name="total_vertex"></param>
///// <param name="myfile"></param>
///// <param name="allChange_Ins"></param>
///// <param name="allChange_Del"></param>
///// <param name="AdjList"></param>
///// <param name="Global2LocalMap"></param>
///// <param name="Local2GlobalMap"></param>
///// <param name="PartitionID_all"></param>
///// <param name="rank"></param>
//void readin_changes1(vector<int>& priority, int* total_vertex, char* myfile, vector<changeEdge>& allChange_Ins, /*vector<changeEdge>& allChange_Del,*/ vector<ColList>& AdjList, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, vector<int>& PartitionID_all, int rank)
//{
//    //cout << "Reading input changed edges data..." << endl;
//    FILE* graph_file;
//    char line[128];
//    graph_file = fopen(myfile, "r");
//    while (fgets(line, 128, graph_file) != NULL)
//    {
//        int n1, n2, inst_status;
//        changeEdge cE;
//        sscanf(line, "%d %d %d", &n1, &n2, &inst_status); //edge wt is there in input file. But we don't need it. So we will ignore it.
//
//        cE.inst = inst_status;
//        if (inst_status == 1) //we handle only edge insertion of this part here
//        {
//            if (PartitionID_all.at(n1) == rank) {//if n1 is in this part
//                int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
//                ColWt c1, c2;
//                c1.col = first_endpoint;
//                c1.flag = 0;
//                cE.node1 = first_endpoint;
//                if (Global2LocalMap.find(n2) != Global2LocalMap.end()) //When n1 and n2 both are in the map=>n2 is either ghost vertex or internal
//                {
//                    int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//                    cE.node2 = second_endpoint;
//                    allChange_Ins.push_back(cE);
//                    c2.col = second_endpoint;
//                    c2.flag = 0;
//                    AdjList.at(first_endpoint).push_back(c2);
//                    AdjList.at(second_endpoint).push_back(c1);
//                    if (PartitionID_all.at(n2) != rank) {//if n2 is from different part=>n2 is bv and priority of n2 is already 2
//                        priority.at(first_endpoint) = 2; //as first_endpoint is bv
//                    }
//                }
//                else {//if n1 is in this part and n2 is in different part (as not in map)
//                    Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
//                    Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
//                    priority.push_back(2); //as local vertex with id *total_vertex is ghost vertex. priority had *total_vertex elements before
//                    priority.at(first_endpoint) = 2; //as first_endpoint is bv
//                    *total_vertex = *total_vertex + 1;
//                    //cout << "rank"<<rank<<"total vertex:" << *total_vertex<<endl;
//                    ColList clmL;
//                    clmL.push_back(c1);
//                    //adding *total_vertex-th element in AdjList
//                    AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
//                    int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//                    c2.col = second_endpoint;
//                    c2.flag = 0;
//                    AdjList.at(first_endpoint).push_back(c2);
//                    cE.node2 = second_endpoint;
//                    allChange_Ins.push_back(cE);
//                }
//            }
//            if ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) == rank)) {//if n2 is in this part but n1 is in different part
//                int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//                ColWt c1, c2;
//                c1.col = second_endpoint;
//                c1.flag = 0;
//                if (Global2LocalMap.find(n1) != Global2LocalMap.end()) //When both n1 and n2 are in the map=>n1 is ghost vertex(priority 2 already) as n1 is not in this part
//                {
//                    int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID
//                    c2.col = first_endpoint;
//                    c2.flag = 0;
//                    AdjList.at(first_endpoint).push_back(c1);
//                    AdjList.at(second_endpoint).push_back(c2);
//                    cE.node1 = first_endpoint;
//                    cE.node2 = second_endpoint;
//                    allChange_Ins.push_back(cE);
//                    priority.at(second_endpoint) = 2; //second_endpoint is bv as n1 is in other part but in the map already
//                }
//                if (Global2LocalMap.find(n1) == Global2LocalMap.end()) //When n2 is in the map(as it is in this part), but n1 is not in map
//                {
//                    Global2LocalMap.insert(make_pair(n1, *total_vertex));//It assigns a local vertex ID
//                    Local2GlobalMap.insert(make_pair(*total_vertex, n1));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
//                    priority.push_back(2); //as local vertex with id *total_vertex is ghost vertex. priority had *total_vertex elements before
//                    priority.at(second_endpoint) = 2; //as second_endpoint is bv
//                    *total_vertex = *total_vertex + 1;
//                    //cout << "rank"<<rank<<"total vertex:" << *total_vertex<<endl;
//                    ColList clmL;
//                    clmL.push_back(c1);
//                    //adding *total_vertex-th element in AdjList
//                    AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
//                    int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID
//                    c2.col = first_endpoint;
//                    c2.flag = 0;
//                    AdjList.at(second_endpoint).push_back(c2);
//                    cE.node1 = first_endpoint;
//                    cE.node2 = second_endpoint;
//                    allChange_Ins.push_back(cE);
//                }
//            }
//        }
//    }
//    fclose(graph_file);
//    //Test
//    int t = 0;
//    for (int i = 0; i < *total_vertex; i++) {
//        t = t + AdjList.at(i).size();
//    }
//    cout << "rank: " << rank << "AFTER readin_changes1: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
//    for (int i = 0; i < *total_vertex; i++) {
//        cout << "[ Priority of " << Local2GlobalMap.find(i)->second << " is: " << priority.at(i) << "]";
//    }
//    cout << endl;
//    return;
//}
//
//
//
///// <summary>
///// Reads graph edges to find 2hop ghost and store in adjacency list. 
///// Also assigns priority 3 for 2hop ghosts
///// </summary>
///// <param name="AdjList"></param>
///// <param name="PartitionID_all"></param>
///// <param name="myfile"> is the edge file</param>
///// <param name="Global2LocalMap"></param>
///// <param name="Local2GlobalMap"></param>
///// <param name="total_vertex"></param>
///// <param name="priority"></param>
///// <param name="rank">holds this part ID</param>
//void read_graphEdges2(vector<ColList>& AdjList, vector<int>& PartitionID_all, char* myfile, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority, int rank)
//{
//    //cout << "Reading input graph 2..." << endl;
//    //cout << "inside read_graphEdges. total vertex till now:" << *total_vertex << endl;
//    FILE* graph_file;
//    char line[128];
//    graph_file = fopen(myfile, "r");
//    while (fgets(line, 128, graph_file) != NULL)
//    {
//        int n1, n2;
//        sscanf(line, "%d %d", &n1, &n2); //our input graph has no wt.
//        /*ColWt c1, c2;*/
//        add2hopghost(n1, n2, AdjList, PartitionID_all, Global2LocalMap, Local2GlobalMap, total_vertex, priority, rank);
//    }
//    fclose(graph_file);
//    //Test
//    int t = 0;
//    for (int i = 0; i < *total_vertex; i++) {
//        t = t + AdjList.at(i).size();
//    }
//    cout << "rank: " << rank << "AFTER read_graphEdges2: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
//    for (int i = 0; i < *total_vertex; i++) {
//        for (int j = 0; j < AdjList.at(i).size(); j++) {
//            cout << "(" << Local2GlobalMap.find(i)->second << "," << Local2GlobalMap.find(AdjList.at(i).at(j).col)->second << ")";
//        }
//    }
//    cout << endl;
//    for (int i = 0; i < *total_vertex; i++) {
//        cout << "[ Priority of " << Local2GlobalMap.find(i)->second << " is: " << priority.at(i) << "]";
//    }
//    cout << endl;
//    return;
//}
//
//void readin_changes2(vector<int>& priority, int* total_vertex, char* myfile, vector<changeEdge>& allChange_Ins, vector<changeEdge>& allChange_Del, vector<ColList>& AdjList, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, vector<int>& PartitionID_all, int rank)
//{
//    //cout << "Reading input changed edges data(2)..." << endl;
//    FILE* graph_file;
//    char line[128];
//    graph_file = fopen(myfile, "r");
//    while (fgets(line, 128, graph_file) != NULL)
//    {
//        int n1, n2, inst_status;
//        changeEdge cE;
//        sscanf(line, "%d %d %d", &n1, &n2, &inst_status); //edge wt is not there in input file. 
//
//        cE.inst = inst_status;
//        if (inst_status == 1 && ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) != rank))) //we handle only edge insertion of GV and 2hop GV here
//        {
//            add2hopghost(n1, n2, AdjList, PartitionID_all, Global2LocalMap, Local2GlobalMap, total_vertex, priority, rank);
//            if ((Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) != Global2LocalMap.end())) {
//                //if n1 and n2 both are in map after add2hopghost but both of them are from different part 
//                //=> priority of them is 2 or 3 => this edge should be inserted in this part
//                int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
//                int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//                cE.node1 = first_endpoint;
//                cE.node2 = second_endpoint;
//                allChange_Ins.push_back(cE);
//            }
//        }
//        if (inst_status == 0 && (Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) != Global2LocalMap.end())) {
//            //if both n1 and n2 are in map
//            int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
//            int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//            cE.node1 = first_endpoint;
//            cE.node2 = second_endpoint;
//            allChange_Del.push_back(cE);
//        }
//    }
//    fclose(graph_file);
//    //Test
//    int t = 0;
//    for (int i = 0; i < *total_vertex; i++) {
//        t = t + AdjList.at(i).size();
//    }
//    cout << "rank: " << rank << "AFTER readin_changes2: total vertices:" << *total_vertex << "total this part edges:" << (t / 2) << endl;
//    for (int i = 0; i < *total_vertex; i++) {
//        for (int j = 0; j < AdjList.at(i).size(); j++) {
//            cout << "(" << Local2GlobalMap.find(i)->second << "," << Local2GlobalMap.find(AdjList.at(i).at(j).col)->second << ")";
//        }
//    }
//    cout << endl;
//    for (int i = 0; i < *total_vertex; i++) {
//        cout << "[ Priority of " << Local2GlobalMap.find(i)->second << " is: " << priority.at(i) << "]";
//    }
//    cout << endl;
//    return;
//}
//
//
//
///// <summary>
///// This function finds 2-hop ghost and add related edges to adjacency list. 
///// Also assigns priority 3 for 2-hop ghost 
///// </summary>
///// <param name="n1"></param>
///// <param name="n2"></param>
///// <param name="AdjList"></param>
///// <param name="PartitionID_all"></param>
///// <param name="Global2LocalMap"></param>
///// <param name="Local2GlobalMap"></param>
///// <param name="total_vertex"></param>
///// <param name="priority"></param>
///// <param name="rank"></param>
//void add2hopghost(int n1, int n2, vector<ColList>& AdjList, vector<int>& PartitionID_all, map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, int* total_vertex, vector<int>& priority, int rank) {
//    ColWt c1, c2;
//    if (((Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) != Global2LocalMap.end())) && ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) != rank))) {
//        //if n1 and n2 both are in map but both of them are from different part => priority of them is 2 or 3
//        int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
//        int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//        c1.col = first_endpoint;
//        c1.flag = 0;
//        c2.col = second_endpoint;
//        c2.flag = 0;
//        AdjList.at(first_endpoint).push_back(c2);
//        AdjList.at(second_endpoint).push_back(c1);
//    }
//    if (((Global2LocalMap.find(n1) != Global2LocalMap.end()) && (Global2LocalMap.find(n2) == Global2LocalMap.end())) && ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) != rank))) {
//        //if n1 is in map, n2 is not in map and both of them are from different part => priority of n1 is 2 or 3
//        int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID of n1
//        if (priority.at(first_endpoint) == 2) { //if n1 is ghost
//            Global2LocalMap.insert(make_pair(n2, *total_vertex));//It assigns a local vertex ID
//            Local2GlobalMap.insert(make_pair(*total_vertex, n2));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
//            priority.push_back(3); //as n2 is 1hop ngbr of ghost vertex
//            *total_vertex = *total_vertex + 1;
//            //cout << "rank"<<rank<<"total vertex:" << *total_vertex<<endl;
//            ColList clmL;
//            c1.col = first_endpoint;
//            c1.flag = 0;
//            clmL.push_back(c1);
//            //adding *total_vertex-th element in AdjList
//            AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
//            c2.col = Global2LocalMap.find(n2)->second; //Getting local ID
//            c2.flag = 0;
//            AdjList.at(first_endpoint).push_back(c2);
//        }
//    }
//    if (((Global2LocalMap.find(n1) == Global2LocalMap.end()) && (Global2LocalMap.find(n2) != Global2LocalMap.end())) && ((PartitionID_all.at(n1) != rank) && (PartitionID_all.at(n2) != rank))) {
//        //if n2 is in map, n1 is not in map and both of them are from different part => priority of n2 is 2 or 3
//        int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID of n2
//        if (priority.at(second_endpoint) == 2) { //if n1 is ghost
//            Global2LocalMap.insert(make_pair(n1, *total_vertex));//It assigns a local vertex ID
//            Local2GlobalMap.insert(make_pair(*total_vertex, n1));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
//            priority.push_back(3); //as n2 is 1hop ngbr of ghost vertex
//            *total_vertex = *total_vertex + 1;
//            //cout << "rank"<<rank<<"total vertex:" << *total_vertex<<endl;
//            ColList clmL;
//            c1.col = second_endpoint;
//            c1.flag = 0;
//            clmL.push_back(c1);
//            //adding *total_vertex-th element in AdjList
//            AdjList.push_back(clmL);//AdjList is vector<vector<colWt>>. So for ghost vertex we add the 1st element as direct push_back
//            c2.col = Global2LocalMap.find(n1)->second; //Getting local ID
//            c2.flag = 0;
//            AdjList.at(second_endpoint).push_back(c2);
//        }
//    }
//}

/// <summary>
/// Reads initial vertex color file and stores the initial colors for IV, GV, 2-hop GV
/// Important:: In input vertex color file color starts from 1. Here we start from 0
/// </summary>
/// <param name="vertexcolor"></param>
/// <param name="myfile"></param>
/// <param name="Global2LocalMap"></param>
void read_Input_Color(int* vertexcolor, char* myfile, map<int, int>& Global2LocalMap, int* maxColor)
{
    FILE* graph_file;
    char line[128];

    graph_file = fopen(myfile, "r");
    while (fgets(line, 128, graph_file) != NULL)
    {
        int node, color;
        sscanf(line, "%d %d", &node, &color);
        if ((color - 1) > * maxColor)
        {
            *maxColor = color - 1;
        }
        if (Global2LocalMap.find(node) != Global2LocalMap.end())
        {
            vertexcolor[Global2LocalMap.find(node)->second] = color - 1; //in input vertex color file color starts from 1. But we need to start from 0
        }
    }
    fclose(graph_file);
    return;
}


void transfer_data_to_GPU(vector<ColList>& AdjList, int*& AdjListTracker, vector<ColWt>& AdjListFull, ColWt*& AdjListFull_device,
    int total_vertex, int totalLocalEdges, int*& AdjListTracker_device, bool zeroInsFlag,
    vector<changeEdge>& allChange_Ins, changeEdge*& allChange_Ins_device, int totalChangeEdges_Ins,
    int deviceId, int totalChangeEdges_Del, bool zeroDelFlag, changeEdge*& allChange_Del_device,
    int*& counter, int*& affected_marked, int*& affectedNodeList, int*& previosVertexcolor, /*int*& updatedAffectedNodeList_del, int*& updated_counter_del,*/ vector<changeEdge>& allChange_Del, size_t  numberOfBlocks, int rank)
{
    //create 1D array from 2D to fit it in GPU
    //cout << "creating 1D array from 2D to fit it in GPU" << endl;
    AdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
    int max_deg = 0, min_deg = 99999;
    for (int i = 0; i < total_vertex; i++) {
        if (AdjList.at(i).size() > max_deg) {
            max_deg = AdjList.at(i).size();
        }
        if (AdjList.at(i).size() < min_deg) {
            min_deg = AdjList.at(i).size();
        }
        AdjListTracker[i + 1] = AdjListTracker[i] + AdjList.at(i).size();
        AdjListFull.insert(std::end(AdjListFull), std::begin(AdjList.at(i)), std::end(AdjList.at(i)));
    }
    totalLocalEdges = AdjListFull.size();
    int actualEdges = totalLocalEdges / 2;
    int avgDeg = totalLocalEdges / total_vertex; //totalLocalEdges = (2 * (edges + totalInsertion))
    cout << "rank: " << rank << " total this part edges after insertion:" << actualEdges << " Max degree:" << max_deg << " Min degree:" << min_deg << " Avg degree:" << avgDeg << endl;
    //cout << "Total Edges in this part: " << totalLocalEdges << endl;
    //cout << "creating 1D array from 2D completed" << endl;


    //Transferring input graph and change edges data to GPU
    //cout << "Transferring graph data from CPU to GPU" << endl;
    //auto startTime_transfer = high_resolution_clock::now();

    CUDA_RT_CALL(cudaMallocManaged(&AdjListFull_device, totalLocalEdges * sizeof(ColWt))); //totalLocalEdges = (2 * (edges + totalInsertion))
    std::copy(AdjListFull.begin(), AdjListFull.end(), AdjListFull_device);


    CUDA_RT_CALL(cudaMalloc((void**)&AdjListTracker_device, (total_vertex + 1) * sizeof(int)));
    CUDA_RT_CALL(cudaMemcpy(AdjListTracker_device, AdjListTracker, (total_vertex + 1) * sizeof(int), cudaMemcpyHostToDevice));

    ////Asynchronous prefetching of data
    CUDA_RT_CALL(cudaMemPrefetchAsync(AdjListFull_device, totalLocalEdges * sizeof(ColWt), deviceId));

    if (zeroInsFlag != true) {
        CUDA_RT_CALL(cudaMallocManaged(&allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge)));
        std::copy(allChange_Ins.begin(), allChange_Ins.end(), allChange_Ins_device);
        //set cudaMemAdviseSetReadMostly by the GPU for change edge data
        CUDA_RT_CALL(cudaMemAdvise(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
        //Asynchronous prefetching of data
        CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), deviceId));
    }

    if (zeroDelFlag != true) {
        CUDA_RT_CALL(cudaMallocManaged(&allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge)));
        std::copy(allChange_Del.begin(), allChange_Del.end(), allChange_Del_device);
        //set cudaMemAdviseSetReadMostly by the GPU for change edge data
        CUDA_RT_CALL(cudaMemAdvise(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
        //Asynchronous prefetching of data
        CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), deviceId));
    }
    counter = 0;
    CUDA_RT_CALL(cudaMallocManaged(&counter, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&affected_marked, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&affectedNodeList, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(affectedNodeList, 0, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&previosVertexcolor, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(previosVertexcolor, -1, total_vertex * sizeof(int)));



    //auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
    //auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
    //cout << "**Time taken to transfer graph data from CPU to GPU: "
    //	<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;
}

#endif
