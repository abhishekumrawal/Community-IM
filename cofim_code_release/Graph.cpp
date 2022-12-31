#include "Graph.h"
#include<set>
#include<fstream>
#include<string>
#include<iostream>
#include<math.h>
#include<stdlib.h>

using namespace std;

Graph::Graph(string data){
	cout << "building graph from: " << data << " ..." << endl;
	ifstream ifs(data.c_str());

	if(!ifs)
		cout << "graph file: " << data << " not openned!" << endl;
	else
		cout << "graph file: " << data << " successfully opened!" << endl;

	string buffer1, buffer2;

	ifs >> buffer1 >> buffer2;
	num_nodes = atoi(buffer1.c_str());
	num_edges = atoi(buffer2.c_str());
	cout << "# of nodes: " << num_nodes << endl;
	cout << "# of edges: " << num_edges << endl;

	cout << "reading edges ..." << endl;
	node_array = new Node[num_nodes];

	for(int i = 0; i < num_edges; i++){
		ifs >> buffer1 >> buffer2;
		int src = atoi(buffer1.c_str());
		int dest = atoi(buffer2.c_str());

		node_array[src].k_out++;
		node_array[dest].k_in++;
		node_array[src].out_edges->insert(dest);

		node_array[dest].k_out++;
		node_array[src].k_in++;
		node_array[dest].out_edges->insert(src);
	}

	set<int>::iterator it;
	set<int> *tmp_set;
	for(int i = 0; i < num_nodes; i++){
		int k_out = node_array[i].out_edges->size();
		node_array[i].k_out = k_out;
		node_array[i].k_in = k_out;
		node_array[i].id_array = new int[k_out];
		tmp_set = node_array[i].out_edges;
		int j = 0;
		for(it = tmp_set->begin(); it != tmp_set->end(); it++){
			node_array[i].id_array[j++] = *it;
		}
		delete node_array[i].out_edges;
	}
	ifs.close();
}

Graph::~Graph(){
	cout << "graph removed!" << endl;
}

void Graph::print(){
	cout << "print graph with " << num_nodes << " nodes and " << num_edges << " edges" << endl;
	for(int i = 0; i < num_nodes; i++){
		for(int j = 0; j < node_array[i].k_out; j++){	
			cout << i << " "  << node_array[i].id_array[j] << endl;
		}
	}
}
