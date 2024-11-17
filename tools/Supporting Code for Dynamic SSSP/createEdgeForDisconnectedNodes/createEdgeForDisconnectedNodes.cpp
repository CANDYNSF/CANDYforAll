#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <queue>
using namespace std;



void create_edges_to_connect_nodes(char* myfile)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	int prev_node = -1;

	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		int wt;
		//Read line
		sscanf(line, "%d %d %d", &n1, &n2, &wt);
		/*cout << n1 << "prv:" << prev_node << endl;*/
		if (n1 == prev_node + 1)
		{
			prev_node = n1;
			continue;
		}
		if (n1 > prev_node + 1)
		{
			for (int i = prev_node + 1; i < n1; i++)
			{
				int random = rand() % 50;
				cout << i <<" "<< prev_node <<" "<< random <<endl;
				cout << prev_node << " " << i << " " << random << endl;
			}
			prev_node = n1;
		}
		

	}
}

/*
arg1: <SSSPGraph.txt>
output will be synthetic edges which will connect the disconnected nodes. So use the output nodes in fullgraph and again find the seqSSSP
Then use the seqSSSP, updated fullgraph and changeEdges for dynamic SSSP algorithm
*/

int main(int argc, char* argv[])
{



	if (argc < 1) { cout << "INPUT ERROR:: First: filename required \n"; return 0; }
	//Check to see if file opening succeeded
	ifstream the_file(argv[1]); if (!the_file.is_open()) { cout << "INPUT ERROR:: Could not open main file\n"; }
	/*** Create DataStructure Sparsifictaion Tree **/


		/******* Read Graph to EdgeList****************/

	string file1 = argv[1];
	char* cstr1 = &file1[0];
	create_edges_to_connect_nodes(cstr1);


	return 0;
}//end of main

	//==========================================//



