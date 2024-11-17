* Input: an undirected graph in .mtx format.
* Output: an undirected graph in .metis format.
* 
* Commands to run:
* ---------------------
* compile:
g++ -O3 -o op_mtx2metis mtxToMetis_skipLines.cpp
* run:
./op_mtx2metis mtx_graph_withComment.txt
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