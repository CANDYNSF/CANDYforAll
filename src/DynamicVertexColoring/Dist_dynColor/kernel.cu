#include <mpi.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "supportVC.h"
#include <stdio.h>
#include <iostream>
#include<vector>
#include <chrono>
#include <map>
#include <set> 
#include <string>
#include "compactor.cuh"
#include "GPUFunctions.cuh"
using namespace std;


#define THREADS_PER_BLOCK 1024 //we can change it

//How to run on LongHorn:
//idev - N1
//cd distVC2 /
//module load cuda / 10.2
//module load gcc / 7.3.0
//nvcc - I / opt / ibm / spectrum_mpi / include - L / opt / ibm / spectrum_mpi / lib - lmpi_ibm kernel.cu - o op_testNew
//ibrun -gpu -np 4 ./op_testNew roadNet-CA_wp_sor color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50 partFileName
//IMPORTANT:: in our vertex color input file lowest color is 1. So while storing input color store as vc -1 => color id should starts from 0 in our code


/// <summary>
/// arg1:graph arg2:initialcolor arg3:changededges arg4:part file
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>

int main(int argc, char** argv) {

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    // Initialize the MPI environment
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    // Get the rank of this process
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size; //stores number of processes/parts
    // Get the number of processes
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    // Get the name of the processor
    /*char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);*/

    if (rank == 0) {
        cout << " number of processors: " << size << endl;
    }


    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    //Setting a GPU for a process rank
    CUDA_RT_CALL(cudaSetDevice(rank % 4));
    CUDA_RT_CALL(cudaFree(0));

    //Below variables are local to each process
    int totalLocalEdges = 0; //totalLocalEdges : every edges counted twice a->b b->a
    int deviceId, numberOfSMs;
    vector<int> PartitionID_all; //stores partition ID for all vertices
    map<int, int> Global2LocalMap; //Used for Mapping GlobalID -> LocalID for local vertices
    map<int, int> Local2GlobalMap; //Used for Mapping LocalID -> GlobalID for local vertices
    int total_vertex; //total of internal, bv, ghost, 2-hop ghost
    vector<ColList> AdjList; //stores input graph in 2D adjacency list
    vector<ColWt> AdjListFull; //Row-major implementation of adjacency list (1D)
    ColWt* AdjListFull_device; //1D array in GPU to store Row-major implementation of adjacency list 
    vector<changeEdge> allChange_Ins, allChange_Del;
    changeEdge* allChange_Ins_device; //stores all change edges marked for insertion in GPU
    changeEdge* allChange_Del_device; //stores all change edges marked for deletion in GPU
    bool zeroDelFlag = false, zeroInsFlag = false;
    int* borderVertexList;
    int total_borderVertex = 0; //holds the total number of border vertices
    int* vertexcolor;
    char* inputColorfile = argv[2];
    int* AdjListTracker;
    int* AdjListTracker_device; //1D array to track offset for each node's adjacency list
    int* counter;
    int* affected_marked;
    int* affectedNodeList;
    int* previosVertexcolor;
    float total_time = 0;
    int* globalID_d; //stores global ID for this part vertices in device
    int max_iteration = 0;
    int iteration = 0;

    //Get gpu device id and number of SMs
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    size_t  numberOfBlocks = 32 * numberOfSMs;

    //Read part ID for all vertices and store
    read_PartitionID_AllVertices(argv[4], PartitionID_all, Global2LocalMap, Local2GlobalMap, &total_vertex, rank); //stores partIDs for GlobalIDs of vertices
    cout << "rank: " << rank << " Initially:: total this part vertices" << total_vertex << endl;

    //initializing priority of all this part vertices as 1
    vector<int> priority(total_vertex, 1);//internal vertex 1, bv 2, 1-hop ngbr of bv 3
    AdjList.resize(total_vertex); //2D adjacency list for storing the graph
    //We need to read files twice to store upto 2 hop distance from this part border vertices
    //Read edges and assign priority 2 to 1-hop ngbrs.
    read_graphEdges(AdjList, PartitionID_all, argv[1], Global2LocalMap, Local2GlobalMap, &total_vertex, priority, rank);
    //Read ins edges and for an edge (u,v) crossing the border, assign priority 2 to the other part end point. So other part endpoint becomes 1-hop ngbr.
    read_Ins(AdjList, priority, &total_vertex, argv[3], Global2LocalMap, Local2GlobalMap, PartitionID_all, rank);
    //Read all edges b/w two priority 2 vertices or edges b/w priority 2 and 2-hop ghosts. It assigns priority 4 to 2-hop ghosts.
    read_2hopghosts(AdjList, PartitionID_all, argv[1], Global2LocalMap, Local2GlobalMap, &total_vertex, priority, rank);
    //Read changed edges.
    read_changEdges(priority, argv[3], allChange_Ins, allChange_Del, AdjList, Global2LocalMap, Local2GlobalMap, &total_vertex);
    int totalChangeEdges_Ins = allChange_Ins.size();
    if (totalChangeEdges_Ins == 0) {
        zeroInsFlag = true;
    }
    int totalChangeEdges_Del = allChange_Del.size();
    if (totalChangeEdges_Del == 0) {
        zeroDelFlag = true;
    }
    cout << "rank: " << rank << " Total Ins edges:" << totalChangeEdges_Ins << "Total Del edges:" << totalChangeEdges_Del << endl;
    
    //Test
    int t = 0;
    //cout << "rank" << rank << "total vertex:" << total_vertex << "size_priority:" << priority.size();
    for (int i = 0; i < total_vertex; i++) {
        t = t + AdjList.at(i).size();
        /*cout << "ngbr of vertex: " << Local2GlobalMap.find(i)->second << " are: ";
        for (int j = 0; j < AdjList.at(i).size(); j++)
        {
            cout << Local2GlobalMap.find(AdjList.at(i).at(j).col)->second << ", ";
        }
        cout << endl;
        cout << "rank: " << rank << "vertex: " << Local2GlobalMap.find(i)->second << "priority: " << priority.at(i) << endl;*/
    }
    cout << "rank: " << rank << " AFTER all read: total vertices:" << total_vertex << "total this part edges:" << (t / 2) << endl;
    
    
    


   ////convert priority vector to array
    int* priority_a = &priority[0];
    int* priority_d;
    CUDA_RT_CALL(cudaMallocManaged(&priority_d, total_vertex * sizeof(int)));
    cudaMemcpy(priority_d, priority_a, total_vertex * sizeof(int), cudaMemcpyDefault);

    //Compute total border vertices and store them in borderVertexList
    CUDA_RT_CALL(cudaMallocManaged(&borderVertexList, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(borderVertexList, 0, total_vertex * sizeof(int)));
    total_borderVertex = cuCompactor::compact<int, int>(priority_d, borderVertexList, total_vertex, predicate(), THREADS_PER_BLOCK);
    //test
    //cout << "rank: " << rank << "total BV" << total_borderVertex << endl;

    CUDA_RT_CALL(cudaMallocManaged(&vertexcolor, total_vertex * sizeof(int)));
    //read initial vertex colors
    int maxColor = -1;
    read_Input_Color(vertexcolor, inputColorfile, Global2LocalMap, &maxColor);
    if (rank == 0) {
        printf("Max color id in input graph: %d\n", maxColor);
    }
    ////test
    /*if (rank == 0) {
        for (int i = 1; i <= 10; i++)
        {
            cout << i << " : " << vertexcolor[i] << endl;
        }
    }*/

    AdjListTracker = (int*)malloc((total_vertex + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
    ////Transfer input graph, changed edges to GPU and set memory advices
    transfer_data_to_GPU(AdjList, AdjListTracker, AdjListFull, AdjListFull_device,
        total_vertex, totalLocalEdges, AdjListTracker_device, zeroInsFlag,
        allChange_Ins, allChange_Ins_device, totalChangeEdges_Ins,
        deviceId, totalChangeEdges_Del, zeroDelFlag, allChange_Del_device,
        counter, affected_marked, affectedNodeList, previosVertexcolor, allChange_Del, numberOfBlocks, rank);

    //Assign priority 2 to all this part border vertices
    assignPriority2ThisPartBorder << < numberOfBlocks, THREADS_PER_BLOCK >> > (borderVertexList, total_borderVertex, AdjListFull_device, AdjListTracker_device, priority_d, affected_marked);
    total_borderVertex = cuCompactor::compact<int, int>(affected_marked, borderVertexList, total_vertex, predicate_findAffected(), THREADS_PER_BLOCK);
    //Assign priority 3
    assignPriority3 << < numberOfBlocks, THREADS_PER_BLOCK >> > (borderVertexList, total_borderVertex, AdjListFull_device, AdjListTracker_device, priority_d);
    CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int)));
    CUDA_RT_CALL(cudaFree(borderVertexList)); //borderVertexList is not required after this

    ////TESTED OK upto this point
    //Test
    /*for (int i = 0; i < total_vertex; i++) {
        cout << "rank: " << rank << "vertex: " << Local2GlobalMap.find(i)->second << "priority: " << priority_d[i] << endl;
    }*/


    if (zeroDelFlag != true) {
        //modify adjacency list to adapt the deleted edges
        deleteEdgeFromAdj << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    ////store global ID of this part vertices
    int* globalID = (int*)malloc((total_vertex) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
    for (int i = 0; i < total_vertex; i++)
    {
        globalID[i] = Local2GlobalMap.find(i)->second;
    }
    //transfer globalID data to GPU (comes under preprocessing step)
    CUDA_RT_CALL(cudaMallocManaged(&globalID_d, total_vertex * sizeof(int)));
    cudaMemcpy(globalID_d, globalID, total_vertex * sizeof(int), cudaMemcpyDefault);
    //test
    /*for (int i = 0; i < 5; i++)
    {
        cout << "global id of" << i << " is: " << globalID[i] << endl;
    }*/



    ////process change edges////
    int* change = 0;
    CUDA_RT_CALL(cudaMallocManaged(&change, sizeof(int)));
    //Process del edges
    if (zeroDelFlag != true) {
        auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start
        deleteEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, vertexcolor, previosVertexcolor, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device, affected_marked, change, priority_d);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required
        CUDA_RT_CALL(cudaFree(allChange_Del_device)); //**releasing allChange_Del_device memory as we don't need it later
        auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
        auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
        cout << "rank: " << rank << "**Time taken for processing del edges: "
            << float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationDelEdge.count()) / 1000;
    }

    //Process ins edges
    if (zeroInsFlag != true) {
        auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start
        insEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_device, vertexcolor, previosVertexcolor, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device, affected_marked, change, priority_d, globalID_d);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required
        auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
        auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
        cout << "rank: " << rank << "**Time taken for processing ins edges(Region 1 and 2): "
            << float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationInsEdge.count()) / 1000;
    }

    //

    ////check conflict, find neighbors to update color
    auto startTimeUpdateNeig = high_resolution_clock::now(); //Time calculation start
    //we use compactor in place of just adding directly using atomic fn to avoid duplication of affected vertices in list
    *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, total_vertex, predicate_findAffected(), THREADS_PER_BLOCK);
    while (*change > 0)
    {
        *change = 0;
        CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int))); //reset affected_marked status for all vertices to 0
        //find eligible neighbors which should be updated
        findEligibleNeighbors << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter, priority_d, globalID_d, change);
        //CUDA_RT_CALL(cudaGetLastError());
        if (iteration < max_iteration) {
            recolorNeighbors << <numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter, change, iteration, max_iteration);
        }
        //find the next frontier: it collects the vertices to be recolored and store without duplicate in affectedNodeList
        *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, total_vertex, predicate_findAffected(), THREADS_PER_BLOCK);
        //CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
    auto stopTimeUpdateNeig = high_resolution_clock::now();//Time calculation ends
    auto durationUpdateNeig = duration_cast<microseconds>(stopTimeUpdateNeig - startTimeUpdateNeig);// duration calculation
    cout << "rank: " << rank << "**Time taken for processing affected neighbors: "
        << float(durationUpdateNeig.count()) / 1000 << " milliseconds**" << endl;
    total_time += float(durationUpdateNeig.count()) / 1000;




    //process insertion inside priority 3 region
    if (zeroInsFlag != true) {
        int* Region3InsEdgeIDList;

        auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start
        CUDA_RT_CALL(cudaMallocManaged(&Region3InsEdgeIDList, totalChangeEdges_Ins * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(Region3InsEdgeIDList, 0, totalChangeEdges_Ins * sizeof(int)));
        //find the Ins Edge ID for which both the endpoints are having priority 3 (we already marked them in insEdge method with inst status = 3)
        *counter = cuCompactor::compact<changeEdge, int, predicate_findRegion3InsEdges>(allChange_Ins_device, Region3InsEdgeIDList, totalChangeEdges_Ins, predicate_findRegion3InsEdges(), THREADS_PER_BLOCK);
        ////test
        //cout << "No. of region 3 change edges: " << *counter << endl;
        //for (int i = 0; i < 5; i++)
        //{
        //    cout << Region3InsEdgeIDList[i] << "::CE are: " << allChange_Ins_device[Region3InsEdgeIDList[i]].node1 << ", " << allChange_Ins_device[Region3InsEdgeIDList[i]].node2 << "priority: "<< priority_d[allChange_Ins_device[Region3InsEdgeIDList[i]].node1] <<" and " << priority_d[allChange_Ins_device[Region3InsEdgeIDList[i]].node2] << endl;
        //}

        //reset affected_marked status for all vertices to 0
        CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int)));
        insEdgeRegion3 << < numberOfBlocks, THREADS_PER_BLOCK >> > (Region3InsEdgeIDList, allChange_Ins_device, vertexcolor, previosVertexcolor, counter, AdjListFull_device, AdjListTracker_device, affected_marked, change, globalID_d);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

        auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
        auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
        cout << "rank: " << rank << "**Time taken for processing ins edges(Region 3): "
            << float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationInsEdge.count()) / 1000;

        //checked upto this

        ////check conflict, find neighbors to update color
        auto startTimeUpdateNeigR3 = high_resolution_clock::now(); //Time calculation start
        //we use compactor in place of just adding directly using atomic fn to avoid duplication of affected vertices in list
        *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, total_vertex, predicate_findAffected(), THREADS_PER_BLOCK);
        //recolor affected neighbors in region 3
        while (*change > 0)
        {
            *change = 0;
            //reset affected_marked status for all vertices to 0
            CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int)));
            //printf("after memset 0: affected_del flag for %d = %d \n", 1, affected_del[1]);

            //find eligible neighbors which should be updated
            findEligibleNeighbors << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter, priority_d, globalID_d, change);
            //CUDA_RT_CALL(cudaGetLastError());
            //find the next frontier: it collects the vertices to be recolored and store without duplicate in affectedNodeList
            *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, total_vertex, predicate_findAffected(), THREADS_PER_BLOCK);
            /*printf("After findEligibleNeighbors: affectedNodeList_del elements:\n");
            for (int i = 0; i < *counter_del; i++)
            {
                printf("%d:", affectedNodeList_del[i]);
            }*/
            //CUDA_RT_CALL(cudaMemset(affected_marked, 0, total_vertex * sizeof(int))); //new
            //recolor the eligible neighbors
           /* recolorNeighbor << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, vertexcolor, previosVertexcolor, AdjListFull_device, AdjListTracker_device, affected_marked, counter, change);
            CUDA_RT_CALL(cudaGetLastError());*/
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }
        auto stopTimeUpdateNeigR3 = high_resolution_clock::now();//Time calculation ends
        auto durationUpdateNeigR3 = duration_cast<microseconds>(stopTimeUpdateNeigR3 - startTimeUpdateNeigR3);// duration calculation
        cout << "rank: " << rank << "**Time taken for processing affected neighbors(Region 3): "
            << float(durationUpdateNeigR3.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationUpdateNeigR3.count()) / 1000;

    }
    cout << "rank: " << rank << "****Total Time for Vertex Color Update: "
        << total_time << " milliseconds****" << endl;

    //Print max color id used
    maxColor = -1;
    for (int i = 0; i < total_vertex; i++)
    {
        if (vertexcolor[i] > maxColor) {
            maxColor = vertexcolor[i];
        }
    }
    printf("COLOR_USED:: rank: %d highest color id used: %d\n", rank, maxColor);

    //validate ::print only if invalid coloring found
    validate << < numberOfBlocks, THREADS_PER_BLOCK >> > (AdjListFull_device, AdjListTracker_device, total_vertex, vertexcolor);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());


    //cleaning
    if (zeroInsFlag != true) {
        CUDA_RT_CALL(cudaFree(allChange_Ins_device));
    }

    CUDA_RT_CALL(cudaFree(vertexcolor));
    CUDA_RT_CALL(cudaFree(previosVertexcolor));
    CUDA_RT_CALL(cudaFree(affected_marked));
    CUDA_RT_CALL(cudaFree(affectedNodeList));
    CUDA_RT_CALL(cudaFree(counter));
    CUDA_RT_CALL(cudaFree(AdjListFull_device));
    CUDA_RT_CALL(cudaFree(AdjListTracker_device));
    delete AdjListTracker;
    CUDA_RT_CALL(cudaFree(priority_d));

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CALL(MPI_Finalize());
}