"# DynColor" 

1st arg: original graph file name
2nd arg: no. of nodes
3rd arg: no. of edges
4th arg: input colored graph file at time t_(n-1)(lowest color id should be 1)(NOTE: Gunrock generates colors with lowest id 1)
5th arg: change edges file name

Compile:
nvcc -o op Color_main.cu


Run:
Example:
C:\Users\Arindam\source\repos\DynamicColoring\DynamicColoring>op.exe graph.txt 10 19 prevColor.txt cE1.txt
Max color id in input graph: 61
Reading input graph...
Reading input graph completed
Time taken to read input graph: 112 microseconds
Reading input changed edges data...
Reading input changed edges data completed. totalInsertion:1
Time taken to read input changed edges: 111 microseconds
creating 1D array from 2D to fit it in GPU
creating 1D array from 2D completed
Transferring graph data from CPU to GPU
**Time taken to transfer graph data from CPU to GPU: 0.97 milliseconds**
**Time taken for processing ins edges: 0.097 milliseconds**
**Time taken for processing affected neighbors: 1.218 milliseconds**
****Total Time for Vertex Color Update: 1.315 milliseconds****

printing output vertex colors:
0:0
1:1
2:61
3:2
4:32
5:0
6:2
7:34
8:4
9:2
highest color id used: 61
