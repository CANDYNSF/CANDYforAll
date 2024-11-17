// changeEdgeForPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;


struct ColWt {
	int col;
	//int wt;
};
typedef vector<ColWt> ColWtList;
/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
we consider only undirected graph here. for edge e(a,b) with weight W represented as : a b W
*/
void read_graphEdges(vector<ColWtList>& AdjList, char* myfile, int* nodes, int outDegree[], int inDegree[])
{
	//cout << "Reading input graph..." << endl;
	
	auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt;
		sscanf(line, "%d %d", &n1, &n2);
		ColWt colwt2;
		colwt2.col = n2;
		//colwt2.wt = wt;
		AdjList.at(n1).push_back(colwt2);
		/*cout << n1 << " " << n2 << " " << wt << endl;*/
		outDegree[n1] = outDegree[n1] + 1;
		inDegree[n2] = inDegree[n2] + 1;
	}
	fclose(graph_file);
	auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	//cout << "Reading input graph completed" << endl;
	//cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;
	return;
}



/*
* arg 1: graph file
* arg 2: nodes
* arg 3: number of changed edges
* arg 4: % of insertion
* arg 5: % of lower outdegree to consider
* arg 6: 1 or 0 (1 for indegree, 0 for out degree)
*/

int main(int argc, char* argv[])
{
	int nodes, edges, lowestDegree = 999999, highestDegree = 0, p_lowDeg, cond;
	if (argc != 7) {
		cout << "Input error:: required 6 inputs. arg1: graphFile, arg2: # of nodes, arg3: # of CE, arg4: % insertion, arg5: % of lower outdegree to consider, arg6: 1 or 0 (1 for indegree, 0 for out degree)";
		exit(0);
	}
	int no_of_movement = 0;
	char* graphFile = argv[1];
	nodes = atoi(argv[2]);
	p_lowDeg = atoi(argv[5]); //percentage of low degree to consider
	cond = atoi(argv[6]); //1 or 0 (1 for indegree, 0 for out degree)
	//edges = atoi(argv[3]);
	int* outDegree = (int*)calloc(nodes, sizeof(int));
	int* inDegree = (int*)calloc(nodes, sizeof(int));
	int* Degree = (int*)calloc(nodes, sizeof(int));


	vector<ColWtList> AdjList; //stores input graph in 2D adjacency list
	AdjList.resize(nodes);
	read_graphEdges(AdjList, graphFile, &nodes, outDegree, inDegree);

	if (cond == 1)
	{
		Degree = inDegree;
	}
	else if(cond == 0)
	{
		Degree = outDegree;
	}
	else
	{
		cout << "wrong input::arg 6: 1 or 0 (1 for indegree, 0 for out degree)";
		exit(0);
	}


	for (int i = 0; i < nodes; i++)
	{
		if (Degree[i] > highestDegree)
		{
			highestDegree = Degree[i];
		}
		if (Degree[i] < lowestDegree)
		{
			lowestDegree = Degree[i];
		}
		//cout << i << " outDegree: " << outDegree[i] << " inDegree: " << inDegree[i]<< " highest out degree: "<< highestOutDegree<<" lowest out degree: "<< lowestOutDegree << endl;
	}

	int range = lowestDegree + (highestDegree - lowestDegree) * p_lowDeg / 100;
	/**** Create Set of Edges to Modify ****/

	int numE = atoi(argv[3]);
	int ins_per = atoi(argv[4]);


	double numF = (double)numE * ((double)ins_per / (double)100);
	int numI = (int)numF;
	int numD = numE - numI;

	int iI = 0;//number of inserts
	int iD = 0;//number of deletes
	int k;

	srand(time(NULL));
	while (1)
	{
		
		//k = rand() % 2;

		//Edges to Insert
		if (iI < numI)
		{
			
			int nx = rand() % nodes;
			
			int ny = rand() % nodes;
			
			int wt = rand() % 100; //highest wt 99
			if (nx == ny) { continue; }

			if (Degree[nx] <= range && Degree[ny] <= range) 
			{ 
				printf("%d  %d 1 \n", nx, ny);
				iI++;
			}
		}//end of if


			// Edge to Delete
		if (iD < numD)
		{
			int nz = rand() % nodes;
			if (Degree[nz] <= range && AdjList.at(nz).size() > 0)
			{
				
				int random1 = rand()% AdjList.at(nz).size();
				int nz1 = AdjList.at(nz)[random1].col;
				//cout << "nz:" << nz << "nz1" << nz1 << endl;
				if (Degree[nz1] <= range)
				{
					printf("%d  %d 0 \n", nz, nz1);
					iD++;
				}
			}
			
		} //end of if

		if (iI == numI && iD == numD) { break; }
	}//end of while



}

