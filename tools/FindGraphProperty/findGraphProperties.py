# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:28:20 2022

@author: Arindam
"""

import os
import networkit as nk
import argparse
import sys
from csv import writer

# path = r'C:\Phd\CUDA test\Test\test 1\Adaptive Pagerank\networkkit_related code'
# os.chdir(path)


def validateArgs(args):
    if os.path.exists(args.file) != True:
        print("ERROR: file", args.file, "doesn't exist.")
        parser.print_help(sys.stderr)
        sys.exit(1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Mining Frequent Itemsets")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path of input file")
    args = parser.parse_args()
    validateArgs(args)

    tlPath = args.file
    fileName = tlPath.split("/")[-1]
    outputFile = fileName + ".csv"
    colNames=['vertexID', 'degree', 'estBetweenness', 'approxCloseness', 'pageRank', 'KatzCentrality']
    # os.makedirs("output", exist_ok=True)
    with open(outputFile, 'w', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(colNames)
    
    # print(f"{step=}, {reset=}, {updateTime=}, {cardF=}, {cardC=}")
    
    
    #property computation
    edgeListReader = nk.graphio.EdgeListReader(' ', 0, directed=True)
    G = edgeListReader.read(tlPath)
    
    #Test
    print(G.numberOfNodes(), G.numberOfEdges())
    
    
    
    #Degree Centrality (considers out degree in directed graph)
    deg = nk.centrality.DegreeCentrality(G)
    deg.run()
    # 10 most central nodes according to degree centrality are
    deg.ranking()[:10]
    deg_scores = deg.scores()
    del deg
    
    
    
    ##Estimate Betweenness Centrality
    nk.setNumberOfThreads(64) #set number of threads
    
    est = nk.centrality.EstimateBetweenness(G, 50, False, True)
    est.run()
    # est.ranking()[:10]
    estBetwn_scores = est.scores()
    #use est.score(v) to get score of vertex v
    #est.scores() returns Returns the scores of all nodes for the centrality algorithm.
    del est
    
    #ApproxCloseness Centrality
    ac = nk.centrality.ApproxCloseness(G, 100)
    ac.run()
    # 10 most central nodes according to closeness are
    ac.ranking()[:10]
    appCls_scores = ac.scores()
    del ac
    
    
    # PageRank
    pr = nk.centrality.PageRank(G, 1e-6)
    pr.run()
    pr.ranking()[:10] # the 10 most central nodes
    pr_scores = pr.scores()
    del pr
    
    
    #Katz Centrality
    #Katz centrality computes the relative influence of a node within a network by measuring the number 
    #of the immediate neighbors, and also all other nodes in the network that connect to the node through 
    #these immediate neighbors. Connections made with distant neighbors are, however, penalized by an 
    #attenuation factor . Each path or connection between a pair of nodes is assigned a weight determined by  
    #and the distance between nodes as alpha.
    #Each iteration of the algorithm requires O(m) time. The number of iterations depends on how long it takes to reach the convergence (and therefore on the desired tolerance tol). 
    
    katz = nk.centrality.KatzCentrality(G, 1e-3)
    katz.run()
    # 10 most central nodes
    katz.ranking()[:10]
    katz_scores = katz.scores()
    del katz
    
    
    
    
    
    
    
    
    for i in range(G.numberOfNodes()):
        col = [i, deg_scores[i], estBetwn_scores[i], appCls_scores[i], pr_scores[i], katz_scores[i]]
        with open(outputFile, 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(col)