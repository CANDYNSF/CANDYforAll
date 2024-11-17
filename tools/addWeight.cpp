// Program to print BFS traversal from a given 
// source vertex. BFS(int s) traverses vertices 
// reachable from s. 
#include<iostream> 
#include <list> 

using namespace std;


void addWeight(char* myfile)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");


	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		sscanf(line, "%d %d", &n1, &n2);
		int wt = rand() % 100; //max weight 99
		printf("%d %d %d\n", n1, n2, wt);
	}
	fclose(graph_file);
	return;
}

int main(int argc, char* argv[])
{

	string file1 = argv[1];
	char* cstr1 = &file1[0];

	addWeight(cstr1);

	return 0;
}
