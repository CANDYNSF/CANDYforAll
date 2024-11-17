// CovertToMetisFormat.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <vector>
#include <fstream> 
#include <sstream>
//#include "all_structure.h"

using namespace std;

struct ColWt {
	int col;
	//int flag; //default 0, deleted -1
};

//Structure for Edge
struct Edge
{
	int node1;
	int node2;
	double edge_wt;
};



struct changeEdge {
	int node1;
	int node2;
	int inst;
};

typedef vector<ColWt> ColList;

/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
we consider only undirected graph here. for edge e(a,b) with weight W represented as : a b W
*/
int read_graphEdges(vector<ColList>& AdjList, char* myfile)
{
	/*cout << "Reading input graph..." << endl;*/
	//auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	int edges = 0;
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		sscanf(line, "%d %d", &n1, &n2);
		ColWt c1, c2;
		c1.col = n1;
		//c1.flag = 0;
		c2.col = n2;
		//c2.flag = 0;
		AdjList.at(n2).push_back(c1);
		AdjList.at(n1).push_back(c2);
		edges++;
	}
	fclose(graph_file);
	//auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	//auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	/*cout << "Reading input graph completed" << endl;*/
	/*cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;*/
	return edges;
}

/*
Our graph format starts from vertex 0
It accepts a weighted graph format. But does not use weight. Can be modified to accept unweighted format easily
Accepted format: a b
E.g.:
1 7
0 5
8 6
Output format starts from vertex 1. Output is in mtx format. First line is <nodes> <nodes> <edges>.
E.g.
9 9 3
1 6
2 8
7 9
Input::
arg 1: input graph
arg 2: total vertices
*/


int main(int argc, char* argv[])
{
	int nodes = atoi(argv[2]);
	vector<ColList> AdjList; //stores input graph in 2D adjacency list
	AdjList.resize(nodes);
	int* AdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
	int edges = read_graphEdges(AdjList, argv[1]);
	//int x = 0;

	cout << nodes << " " << nodes << " " << edges << endl;
	for (int i = 0; i < nodes; i++)
	{
		int list_size = AdjList.at(i).size();
		for (int j = 0; j < list_size; j++)
		{
			if (i < AdjList.at(i).at(j).col) //to avoid writing a b when already wrote b a
			{
				cout << (i + 1) << " " << (AdjList.at(i).at(j).col + 1) << endl; //we add 1 as our format starts from 0
			}
				
		}
	}

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
