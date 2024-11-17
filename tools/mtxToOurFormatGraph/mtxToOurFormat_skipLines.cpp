// CovertToMetisFormat.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <vector>

using namespace std;
/******* Network Structures *********/
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
int changeFromMTXtoOur(char* myfile)
{
	/*cout << "Reading input graph..." << endl;*/
	//auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	int edges = 0;
	graph_file = fopen(myfile, "r");
	int max = 0, totalVertex = 0;
	int zeroIndexed = false;
	int n1, n2;
	int i = 0;
	while (fgets(line, 128, graph_file) != NULL)
	{
		if (line[0] == '%') continue;
		if (line[0] == '#') continue;
		if (i == 0)
		{
			sscanf(line, "%d %d", &n1, &n2);//get 1st line. We ignore 1st line of mtx format graph, as it has n n m as first line
			i = 1;
			continue;
		}
		sscanf(line, "%d %d", &n1, &n2); //our input graph has wt. But we don't need it. So we will ignore wt.
		cout << (n1 - 1) << " " << (n2 -1) << endl; //mtx is 1 indexed
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
It accepts a unweighted graph format.
E.g.:
0 1
1 3
1 2
Output: 4
*/
/*
* Arg 1: graph file
*/

int main(int argc, char* argv[])
{
	int edges = changeFromMTXtoOur(argv[1]);

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
