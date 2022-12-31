#include "Node.h"

Node::Node(){
	k_out = 0;
	k_in = 0;
	id_array = NULL;
	out_edges = new set<int>();
}

Node::~Node(){
	if(out_edges != NULL){
		out_edges->clear();
		delete out_edges;
	}
	if(id_array != NULL)
		delete id_array;
}
