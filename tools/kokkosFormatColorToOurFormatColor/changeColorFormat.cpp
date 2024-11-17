// CovertToMetisFormat.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <vector>

using namespace std;

/*
read_graphEdges reads the kokkos color file: <color>
writes color as : <id> <color>
*/
void read_writeColor(char* myfile)
{
	/*cout << "Reading input graph..." << endl;*/
	//auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	int i = 0;
	while (fgets(line, 128, graph_file) != NULL)
	{
		int c;
		sscanf(line, "%d", &c);
		if(c != null)
		cout << i << " " << c << endl;
		i++;
	}
	fclose(graph_file);
}


/*
*/
/*
* Arg 1: kokkos color file
*/

int main(int argc, char* argv[])
{
	read_writeColor(argv[1]);

	return 0;
}
