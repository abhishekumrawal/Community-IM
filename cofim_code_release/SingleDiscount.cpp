#include "Utility.h"
#include "Graph.h"
#include "MemoryUsage.h"

#include<stdlib.h>
#include<string.h>
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<queue>
#include<functional>


using namespace std;

void parseArg(int argn, char ** argv);
void run(Graph *g, int k);

int main(int argn, char ** argv)
{
	cout << "Program Start at: " << currentTimestampStr() << endl;
	cout << "Arguments: ";
	for(int i = 0; i < argn; i++){
		cout << argv[i]<<" ";
	}
	cout << endl;
	cout << "--------------------------------------------------------------------------------" << endl;
    parseArg( argn, argv );
    cout << "--------------------------------------------------------------------------------" << endl;
    cout<<"Program Ends Successfully at: " << currentTimestampStr() << endl;
    return 0;
}

void parseArg(int argn, char ** argv)
{
	// the parameters
    string data="";  // the path of the dataset
    int k=0;  //the # of seeds to be found

    for(int i=0; i<argn; i++)
    {
        if(argv[i] == string("-data"))
        	data=string(argv[i+1]);
        if(argv[i] == string("-k"))
        	k=atoi(argv[i+1]);
    }
    if (data=="")
        ExitMessage("argument data missing");
	if(k == 0)
		ExitMessage("argument k missing");
    Graph *g = new Graph(data);
	cout << "graph " << data << " was built successfully!" << endl;
	run(g, k);
}


void run(Graph *g, int k){
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Start Single Discount (SD) algorithm" << endl;
	time_t time_start = time(NULL);
	cout << "No.\tnode_id\ttime(s)" << endl;

	int *degree_arr = new int[g->num_nodes];
	for(int i = 0; i < g->num_nodes; i++){
		degree_arr[i] = g->node_array[i].k_out;
	}
	
	int *seed_arr = new int[k];
	for(int i = 0; i < k; i++){
		int max_degree = -1;
		int max_node = -1;
		for(int j = 0; j < g->num_nodes; j++){
			if(degree_arr[j] > max_degree){
				max_degree = degree_arr[j];
				max_node = j;
			}
		}
		seed_arr[i] = max_node;
		degree_arr[max_node] = -1; // mark the selected node as removed
		int *p = g->node_array[max_node].id_array;
		int k_out = g->node_array[max_node].k_out;
		for(int j = 0; j < k_out; j++){
			int neigh = p[j];
			if(degree_arr[neigh] != -1)  // if the neighbor has not been selected yet
				degree_arr[neigh]--;  // discount the degree of the neighbor
		}
		cout << i+1 << "\t" << max_node << "\t" << time(NULL) - time_start << endl;
	}
	delete[] seed_arr;
	delete[] degree_arr;

	disp_mem_usage("");
	cout << "Time used: " << time(NULL) - time_start << " s" << endl;
}

