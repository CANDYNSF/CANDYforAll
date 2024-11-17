# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:32:25 2022

@author: Arindam
"""

import pandas as pd
# import numpy as np
import os
# from collections import defaultdict
import argparse
# import itertools
# import pickle
import random
import time
import sys
import os.path
# from csv import writer



def validateArgs(args):
    """
    validates input arguments
    Parameters
    ----------
    args : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    if args.noOfCE < 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.prop < 0 or args.prop > 4:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.prpercentage <= 0 or args.prpercentage > 100:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if os.path.exists(args.input) != True:
        print("ERROR: file", args.input, "doesn't exist.")
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    default_seed = int(time.time())
    default_noOfCE = 10
    default_prop = 0
    default_lowPrPercentage = 5
    
    parser = argparse.ArgumentParser(description="Change edge creation")
    parser.add_argument("-r", "--seed", type=int, default=default_seed, help="Seed for randomness")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path of input file")
    parser.add_argument("-n", "--noOfCE", type=int, default=default_noOfCE, help="No. of Change Edges")
    parser.add_argument("-p", "--prop", type=int, default=default_prop, help="0:lowPr_lowOutDeg, 1:lowPr_lowInDeg, 2:lowPr_lowEstBetwns, 3:lowPr_lowClsns, 4:lowPr_lowKatz")
    parser.add_argument("-l", "--prpercentage", type=int, default=default_lowPrPercentage, help="percentage of low PageRank values to consider")
    
    args = parser.parse_args()
    validateArgs(args) #validate args

    tlPath = args.input
    noOfCE = int(args.noOfCE)
    prop = int(args.prop)
    prp = .01*args.prpercentage
    
    df1 = pd.read_csv(tlPath, header = 0) #read input
    random.seed(args.seed)

    df_lowPr = df1.nsmallest(int(prp*df1.shape[0]), 'pageRank')
    
    # df_lowPr_lowEstBetwns = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), 'estBetweenness')
    # df_lowPr_lowOutDeg = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), 'outDegree')
    # df_lowPr_lowInDeg = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), 'inDegree')
    # df_lowPr_lowClsns = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), 'approxCloseness')
    # df_lowPr_lowKatz = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), 'KatzCentrality')
    
    # propName = [df_lowPr_lowOutDeg, df_lowPr_lowInDeg, df_lowPr_lowEstBetwns, df_lowPr_lowClsns, df_lowPr_lowKatz]
    
    # for i in range(noOfCE):
    #     s = propName[prop].shape[0]
    #     r1 = random.randint(0,s)
    #     r2 = random.randint(0,s)
    #     u = propName[prop]["vertexID"].iloc[r1]
    #     v = propName[prop]["vertexID"].iloc[r2]
    #     print(u," ",v)
    
    propName = ['outDegree','inDegree','estBetweenness','approxCloseness','KatzCentrality']
    df2 = df_lowPr.nsmallest(int(.5*df_lowPr.shape[0]), propName)
    for i in range(noOfCE):
        s = df2.shape[0]
        r1 = random.randint(0,s)
        r2 = random.randint(0,s)
        u = df2["vertexID"].iloc[r1]
        v = df2["vertexID"].iloc[r2]
        print(u," ",v) 
        
        
        