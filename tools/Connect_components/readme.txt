connectUsingBFS.cpp takes graph as below format
<a b>


nvcc -o op_connect connectUsingBFS.cpp
./op_connect <graphFile> <number of nodes>  > newEdges.txt


it gives only the new edges required to connect the graph components
So add these new edges to previous edges.

cat newEdges.txt >> graphFile


Full example:
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ g++ -O connectUsingBFS.cpp -o op
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ ls
connectUsingBFS.cpp  cudaTest.vcxproj  cudaTest.vcxproj.user  graph_unweighted.txt  kernel.cu  op  op_connect.exe  x64
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ ./op graph_unweighted.txt 10
2 3
3 4
4 5
7 9
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ ./op graph_unweighted.txt 10 > newEdges
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ cp graph_unweighted.txt  graph2
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ cat newEdges >> graph2
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ cat graph2
0 1
2 1
7 2
5 6
8 6
2 3
3 4
4 5
7 9
akhanda@DESKTOP-BNF9MCK:/mnt/c/Users/Arindam/source/repos/cudaTest/cudaTest$ cat graph_unweighted.txt
0 1
2 1
7 2
5 6
8 6