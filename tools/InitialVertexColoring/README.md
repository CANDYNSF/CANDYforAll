"# InitialColor" 
Input:
1st arg: original graph file name
2nd arg: no. of nodes

Output: initial colors

Compile:
nvcc -o op_initialColor Color_main.cu


Run:
Example:
C:\Users\Arindam\source\repos\DynColorHeuristicIndependent\DynColorHeuristicIndependent\withoutWt\test_initialColor>op_initialColor.exe graph.txt 10
Reading input changed edges data...
Reading input changed edges data completed. totalInsertion:19
Time taken to read input changed edges: 409 microseconds
creating 1D array from 2D to fit it in GPU
creating 1D array from 2D completed
Transferring graph data from CPU to GPU
**Time taken to transfer graph data from CPU to GPU: 0.885 milliseconds**
**Time taken for processing ins edges: 0.648 milliseconds**
**Time taken for processing affected neighbors: 9.954 milliseconds**
****Total Time for Vertex Color Update: 10.602 milliseconds****
highest color id used: 4
0 0
1 1
2 1
3 0
4 0
5 0
6 1
7 2
8 3
9 4
