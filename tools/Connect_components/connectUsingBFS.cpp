// Program to print BFS traversal from a given 
// source vertex. BFS(int s) traverses vertices 
// reachable from s. 
#include<iostream> 
#include <list> 

using namespace std;

// This class represents a directed graph using 
// adjacency list representation 
class Graph
{
	int V; // No. of vertices 

	// Pointer to an array containing adjacency 
	// lists 
	list<int>* adj;
public:
	Graph(int V); // Constructor 

	// function to add an edge to graph 
	void addEdge(int v, int w);

	// prints BFS traversal from a given source s 
	void BFS(int s);
};

Graph::Graph(int V)
{
	this->V = V;
	adj = new list<int>[V];
}

void Graph::addEdge(int v, int w)
{
	adj[v].push_back(w); // Add w to v’s list. 
	adj[w].push_back(v); //****adds twice to convert to undirected graph.****
}

void Graph::BFS(int s)
{
	// Mark all the vertices as not visited 
	bool* visited = new bool[V];
	for (int i = 0; i < V; i++)
		visited[i] = false;

	// Create a queue for BFS 
	list<int> queue;

	// Mark the current node as visited and enqueue it 
	visited[s] = true;
	queue.push_back(s);
	int last_node = 0;

	for (int j = 0; j < V; j++) {
		if (visited[j] == true && j != s) {
			last_node = j;
			continue;
		}
		else if (visited[j] == false) {
			cout << last_node << " " << j <<endl;
			queue.push_back(j);
			last_node = j;
		}
		

		// 'i' will be used to get all adjacent 
		// vertices of a vertex 
		list<int>::iterator i;

		while (!queue.empty())
		{
			// Dequeue a vertex from queue and print it 
			s = queue.front();
			//cout << s << " ";
			queue.pop_front();

			// Get all adjacent vertices of the dequeued 
			// vertex s. If a adjacent has not been visited, 
			// then mark it visited and enqueue it 
			for (i = adj[s].begin(); i != adj[s].end(); ++i)
			{
				if (!visited[*i])
				{
					visited[*i] = true;
					queue.push_back(*i);
				}
			}
		}
	}
}


void readin_graphU2(char* myfile, int nodes)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	Graph g(nodes);
	

	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		int wt;
		sscanf(line, "%d %d", &n1, &n2);
		g.addEdge(n1, n2); //acceps only a->b. don't need b->a to add from graph file. it automatically converts to undirected graph.
	}
	fclose(graph_file);
	g.BFS(0);
	return;
}

// Driver program to test methods of graph class 
int main(int argc, char* argv[])
{
	// Create a graph given in the above diagram 
	//Graph g(8);
	//g.addEdge(0, 1);
	//g.addEdge(3, 1);
	///*g.addEdge(0, 2);
	//g.addEdge(1, 2);
	//g.addEdge(2, 0);*/
	//g.addEdge(5, 3);
	//g.addEdge(4, 7);

	////cout << "Following is Breadth First Traversal "<< "(starting from vertex 2) \n";
	//g.BFS(0);

	string file1 = argv[1];
	char* cstr1 = &file1[0];

	int nodes = 0;
	nodes = atoi(argv[2]); //we pass the number of total nodes as argument

	readin_graphU2(cstr1, nodes);

	return 0;
}
