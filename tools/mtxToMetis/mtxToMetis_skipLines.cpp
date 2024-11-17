#include <iostream>
#include <vector>

using namespace std;

struct ColWt {
	int col;
	//int wt;
};

typedef vector<ColWt> ColList;

/*
* Input: an undirected graph in .mtx format.
* Output: an undirected graph in .metis format.
* 
* Commands to run:
* ---------------------
* compile:
* g++ -O3 -o op_mtx2metis mtxToMetis_skipLines.cpp
* run:
* ./op_mtx2metis <graph file name>
* 
It considers the input graph as an unweighted graph. Although it can be easily changed by uncommenting wt in ColWt and reading and storing wt. 
E.g.:
================
Input:
---------
%%MatrixMarket matrix coordinate real symmetric
 4 4 3
 1 2
 3 1
 4 2
Output:
---------
4 3
2 3
1 4
1
2
*/
/*
* Arg 1: graph file
*/

int main(int argc, char* argv[])
{
	FILE* graph_file;
	char line[128];
	int edges = 0, nodes = 0;
	vector<ColList> AdjList; //stores input graph in 2D adjacency list
	graph_file = fopen(argv[1], "r");
	int flag = 0;
	while (fgets(line, 128, graph_file) != NULL)
	{
		if (line[0] == '%') continue;
		if (line[0] == '#') continue;
		int n1, n2, wt;
		if (flag == 0)
		{
			sscanf(line, "%d %d %d", &nodes, &n2, &edges);//get 1st line. We ignore 1st line of mtx format graph, as it has n n m as first line
			AdjList.resize(nodes);
			flag = 1;
			//continue;
		}
		else {
			sscanf(line, "%d %d", &n1, &n2); //Consider edges without wt. If wt is there also, we ignore it here.
			ColWt c1, c2;
			c1.col = n1;
			c2.col = n2;
			AdjList.at(n2 - 1).push_back(c1); //n2 -1 as .mtx graph starts from 1
			AdjList.at(n1 - 1).push_back(c2);
		}
		
	}
	fclose(graph_file);

	int x = 0;

	cout << nodes << " " << edges << endl; //1st line of metis format is "nodes edges"
	for (int i = 0; i < nodes; i++)
	{
		int list_size = AdjList.at(i).size();
		for (int j = 0; j < list_size; j++)
		{
			if (x == 0)
			{
				cout << (AdjList.at(i).at(j).col);
				x++;
			}
			else {
				cout << " " << (AdjList.at(i).at(j).col);
			}
		}
		cout << endl;
		x = 0;
	}
	return 0;
}
