#pragma once
#include<set>
#include<stdio.h>

using namespace std;

class Node{
public:
	int k_out;  //# of out edges
	int k_in;  //# of in edges

	int* id_array;  // array of out edges

	set<int> *out_edges;  // set of out edges for temporal use

	Node();
	~Node();
};
