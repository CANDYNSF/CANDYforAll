1. Create graph property file(excel file) using code from folder "FindGraphProperty"
2. Use the graph property file to create changed edges
	python3 changeEdgeGenerator.py -i <graph property file name>
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -r SEED, --seed SEED  Seed for randomness
	  -i INPUT, --input INPUT
	                        Path of input file
	  -n NOOFCE, --noOfCE NOOFCE
	                        No. of Change Edges
	  -p PROP, --prop PROP  0:lowPr_lowOutDeg, 1:lowPr_lowInDeg,
	                        2:lowPr_lowEstBetwns, 3:lowPr_lowClsns,
	                        4:lowPr_lowKatz
	  -l PRPERCENTAGE, --prpercentage PRPERCENTAGE
                        percentage of low PageRank values to consider