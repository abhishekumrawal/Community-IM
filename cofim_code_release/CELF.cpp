#include "Utility.h"
#include "Graph.h"
#include "MemoryUsage.h"

#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<queue>
#include<functional>

#define S_STATE 0
#define I_STATE 1
#define SI_STATE 2
#define R_STATE 3
#define REDUND 10

struct Pair {
	int key;
	float value;
	Pair(int key, float value) :key(key), value(value) {};
};
typedef struct Pair Pair;

bool operator > (Pair a, Pair b) {
	return a.value < b.value;
}

struct MinPair {
	int key;
	float value;
	MinPair(int key, float value) :key(key), value(value) {};
};
typedef struct MinPair MinPair;

bool operator > (MinPair a, MinPair b) {
	return a.value > b.value;
}

using namespace std;

void parseArg(int argn, char ** argv);
void run(Graph *g, string data, int k);

float mc_influence(Graph *g, int *seed_arr, int k);

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
	run(g, data, k);
}

// The CELF algorithm
void run(Graph *g, string data, int k){
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Start CoFIM algorithm" << endl;
    cout << "data:" << data << " k:" << k << endl;
	time_t time_start = time(NULL);
	cout << "Finding top " << k << " nodes with CELF algorithm" << endl;
	cout << "No.\tnode_id\tinfluence\ttime(s)" << endl;

	// find the top k nodes
	int *seed_arr = new int[k];
	priority_queue<Pair, vector<Pair>, greater<Pair> > pqueue;
	for(int i = 0; i < g->num_nodes; i++){
		seed_arr[0] = i;
		float inf = mc_influence(g, seed_arr, 1);
		pqueue.push(Pair(i, inf));
	}
	int *updated = new int[g->num_nodes];  //the flag array indicates whehter marginal gain of a node is updated
	float total_inf = 0;
	for (int i = 0; i < g->num_nodes; i++)
		updated[i] = 1;
	for (int i = 0; i < k; i++) {
		Pair best_pair = pqueue.top();
		pqueue.pop();
		while (!updated[best_pair.key]) {
			seed_arr[i] = best_pair.key;
			float m_gain = mc_influence(g, seed_arr, i+1) - total_inf;
			best_pair.value = m_gain;
			updated[best_pair.key] = 1;
			pqueue.push(best_pair);
			best_pair = pqueue.top();
			pqueue.pop();
		}
		seed_arr[i] = best_pair.key;
		total_inf += best_pair.value;
		cout << i + 1 << "\t" << best_pair.key << "\t" << total_inf << "\t" << time(NULL) - time_start << endl;
		memset(updated, 0, g->num_nodes * sizeof(int)); // reset the flag array
	}
	delete[] updated;
	disp_mem_usage("");
	cout << "Time used: " << time(NULL) - time_start << " s" << endl;
}

//Compute the influence spread using Mento-Carlo simulation
float mc_influence(Graph *g, int *seed_arr, int k){
	srand((unsigned)time(NULL));
	double inf = 0;
	int *i_arr = new int[g->num_nodes]; //the array of current active nodes
	int i_size = 0; // the # of newly active nodes 
	int *r_arr = new int[g->num_nodes]; // the array of previous active nodes
	int r_size = 0; // the # of previously active nodes
	int *si_arr = new int[g->num_nodes];  // the array of nodes to be active in t+1
	int si_size = 0; // the # of nodes to be active in t+1
	int *state_arr = new int[g->num_nodes]; // the state of nodes
	memset(state_arr, S_STATE, g->num_nodes * sizeof(int)); // initialize the state array	
	int *rand_arr = new int[g->num_nodes]; //the 0 ~ n-1 numbers sorted by random order
	for(int r = 0; r < NUM_SIMUS; r++){
		double active_size = 0;
		//reset the state of all nodes		
		for(int i = 0; i < r_size; i++){
			state_arr[r_arr[i]] = S_STATE;
		}		
		r_size = 0;		
		// initialize the seed nodes
		for(int i = 0; i < k; i++){
			i_arr[i_size++] = seed_arr[i];
			state_arr[i_arr[i]] = I_STATE;
		}
		while(i_size > 0){
			active_size += i_size;
			si_size = 0;
			randomOrder(rand_arr, i_size);
			for(int i = 0; i < i_size; i++){
				int i_node = i_arr[i];
				int k_out = g->node_array[i_node].k_out;
				for(int j = 0; j < k_out; j++){
					int neigh = g->node_array[i_node].id_array[j];
					if (state_arr[neigh] == S_STATE) {
						int k_in = g->node_array[neigh].k_in;
						double pp = 1.0 / k_in;
						double rand_float = ((double)rand()) / RAND_MAX;
						if(rand_float < pp) {
							state_arr[neigh] = SI_STATE;
							si_arr[si_size++] = neigh;
						}
					}					
				}
			}
			for(int i = 0; i < i_size; i++){
				state_arr[i_arr[i]] = R_STATE;
				r_arr[r_size++] = i_arr[i];
			}
			i_size = 0;
			for(int i = 0; i < si_size; i++){
				state_arr[si_arr[i]] = I_STATE;
				i_arr[i_size++] = si_arr[i];
			}
		}
		inf += active_size;
	}
	delete[] i_arr;
	delete[] r_arr;
	delete[] si_arr;
	delete[] state_arr;
	delete[] rand_arr;
	return inf / NUM_SIMUS;
}
