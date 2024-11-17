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
void read_graphEdges(char* myfile, int edges, int total_files)
{
	/*cout << "Reading input graph..." << endl;*/
	//auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	
	int x = 1, y =0;
	
	ofstream opfile;
	opfile.open (std::string(myfile + std::to_string(x)));
	
	
	
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		sscanf(line, "%d %d", &n1, &n2);
		
		y++;
		if((y % (edges/(total_files-1))) == 0)
		{
			opfile.close();
			string buf1("edge_files_");
			buf1.append(myfile);
			ofstream opfile1;
			opfile1.open (buf1, std::ios_base::app);
			string op = "./" + std::string(myfile) + std::to_string(x);
			opfile1 << op << "\n" ;
			opfile1.close();
			x++;
			opfile.open (std::string(myfile + std::to_string(x)));
		}
		opfile << n1 <<" "<<n2<<"\n";
	}
	opfile.close();
	fclose(graph_file);
	
	string buf1("edge_files_");
	buf1.append(myfile);
	ofstream opfile1;
	opfile1.open (buf1, std::ios_base::app);
	string op = "./" + std::string(myfile) + std::to_string(x);
	opfile1 << op << "\n" ;
	opfile1.close();
	return;
}

/*
Input::
arg 1: input graph
arg 2: total edges
arg 3: total_files to be created

Output:
part files of #total_files
a file with paths of all part files.
*/


int main(int argc, char* argv[])
{
	int edges = atoi(argv[2]);
	int total_files = atoi(argv[3]);
	read_graphEdges(argv[1], edges, total_files);
	
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
