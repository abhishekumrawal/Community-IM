Welcome to use the source code of the CoFIM algorithm proposed by Shang et al.


- Compile

	Makefile is included, just type type "make" to compile all the source codes.
	"gcc 4.7.2" preferred


- Execute

	Example:
		./CoFIM -data NetHEPT.txt -com NetHEPT_com.txt -gamma 3 -k 50

	Arguments:
		-data:
			the graph file
		-com:
			the community file
		-k:
			number of seed nodes
		-gamma:
			algorithm parameter


- Evalution

	To evaluate the influence spread of any algorithm using Monte-Carlo simulation, run:

		./CoFIM -data NetHEPT.txt -seeds NetHEPT_seeds.txt -k 50
	
	Arguments:
		-data:
			the graph file
		-seeds:
			the seed file consisting of k lines, where each line contains the seed node id.


- Other algorithms

	To run the CELF algorithm, run:
		./CELF -data NetHEPT.txt -k 50

	To run the SingleDiscount algorithm, run:
		./SingleDiscount -data NetHEPT.txt -k 50

	To run the Degree Heuristic algorithm, run:
		./Degree -data NetHEPT.txt -k 50


- Graph file format

	The first line indicates the number of nodes and edges.
	The following lines includes the source and destination node id of an edge

	line 1 : #nodes	#edges
	line 2 to 1+#edge : source id	destination id

	All inputs are separated by tab(\t).

	The graph is undirected, so please just remain one edge.

	Example:
	4	2
	0	1
	2	3

	This graph contains four nodes and two edges. 

	Do not use:
	4	2
	0	1
	1	0
	2	3
	3	2

	Sample graph file "NetHEPT.txt" is included.


- Community file format

	The file contains |C| lines, where |C| is the number of communities. 
	Each line represents a community, consisting of ids (separated by \t) of the nodes in that community

	Example:
	0	1
	2	3

	Sample community file "NetHEPT_com.txt" is included.

	The network consists of two communities {0,1} and {2,3}.

To run the program correctly, please make sure that all node ids are ranged from 0 to #nodes-1.
Also make sure that the ids in the community file is consist with the ids in the graph file.