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
void run_cofim(Graph *g, string data, string com, int k, float gamma);
void evaluate(Graph *g, string data, string seeds, int k);
void evaluate_total(Graph *g, string data, string seeds, int k, int simus);
void analysis(Graph *g, string data, string com);

float mc_influence(Graph *g, int *seed_arr, int k);
float mc_influence(Graph *g, int *seed_arr, int k, int simus);
float marginal_gain(Graph *g, int *n2c, set<int> node_set, set<int> com_set, int node, float gamma);
void add_seed(Graph *g, int *n2c, set <int> *seed_set, set<int> *node_set, set<int> *com_set, int node);
float get_score(Graph *g, int *n2c, set<int> seed_set, float gamma);

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
    string com = "";
    float gamma = -1;  //the algorithm parameter
    string seeds = "";  // the path of the seed nodes for MC simulation
	bool is_total = false, is_stat = false;
	int simus = 10000;

    for(int i=0; i<argn; i++)
    {
        if(argv[i] == string("-data"))
        	data=string(argv[i+1]);
        if(argv[i] == string("-k"))
        	k=atoi(argv[i+1]);
        if(argv[i] == string("-com"))
        	com = argv[i+1];
        if(argv[i] == string("-gamma"))
        	gamma = atof(argv[i+1]);
        if(argv[i] == string("-seeds"))
        	seeds = argv[i+1];
		if(argv[i] == string("-total"))
			is_total = true;
		if(argv[i] == string("-simus"))
			simus = atoi(argv[i+1]);
		if(argv[i] == string("-stat"))
			is_stat = true;
    }
    if (data=="")
        ExitMessage("argument data missing");
	if(k == 0 && !is_stat)
		ExitMessage("argument k missing");
    if(seeds == "" && !is_stat){
    	if(com == "")
    		ExitMessage("argument com is missing");
    	if(gamma == -1)
    		ExitMessage("argument gamma is missing");
    }	
    Graph *g = new Graph(data);
	cout << "graph " << data << " was built successfully!" << endl;
	if (seeds == ""){
		if(is_stat)
			analysis(g, data, com);
		else
			run_cofim(g, data, com, k, gamma);
	}
    else{
		if(is_total)
			evaluate_total(g, data, seeds, k, simus);
		else
			evaluate(g, data, seeds, k);
	}
}

void evaluate(Graph *g, string data, string seeds, int k){
	cout << "evaluating influence... data:" << data << " seeds:" << seeds << endl;
	int *seed_arr = new int[k];
	ifstream ifs(seeds.c_str());
	if(!ifs)
		cout << "seeds file: " << seeds << " cannot be openned!" << endl;
	else
		cout << "seeds file: " << seeds << " successfully openned!" << endl;
	cout << "id\tseed\tinfluence\ttimestamp" << endl;
	string buffer;
	int point_arr[11] = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
	for(int i = 0; i < k; i++){
		ifs >> buffer;
		seed_arr[i] = atoi(buffer.c_str());
		int match = 0;
		for(int j = 0; j < 11; j++){
			if(point_arr[j] == i+1){
				match = 1;
				break;
			}
		}
		if(match){
			float inf = mc_influence(g, seed_arr, i + 1);
			cout << i + 1 << "\t" << seed_arr[i] << "\t";
			cout << inf << '\t' << currentTimestampStr() << endl;
		}
	}
}

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

void run_cofim(Graph *g, string data, string com, int k, float gamma){
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Start CoFIM algorithm" << endl;
    cout << "data:" << data << " com:" << com << " k:" << k << " gamma:"<< gamma << endl;
	ifstream ifs(com.c_str());
	if (!ifs){
		cout << "community file: " << com << " not openned!" << endl;
		return;
	}
	else
		cout << "community file: " << com << " successfully opened!" << endl;
	int *n2c = new int[g->num_nodes];
	string str;
	int com_id = 0;
	while (getline(ifs, str)){
		istringstream iss(str);
		string buffer;
		while (iss >> buffer) {
			int node = atoi(buffer.c_str());
			n2c[node] = com_id;
		}
		com_id++;
	}
	cout << "# of communities: " << com_id << endl;
	time_t time_start = time(NULL);
	cout << "Finding top " << k << " nodes with CoFIM algorithm" << endl;
	cout << "No.\tnode_id\ttime(s)" << endl;
	priority_queue<MinPair, vector<MinPair>, greater<MinPair> > tmp_pqueue;
	float avg_k_out = 2.0 * g->num_edges / g->num_nodes;
	set<int> tmp_set;
	for (int i = 0; i < g->num_nodes; i++){
		int k_out = g->node_array[i].k_out;
		if (k_out < avg_k_out)
			continue;
		tmp_set.insert(i);
		float score = get_score(g, n2c, tmp_set, gamma);
		tmp_set.erase(tmp_set.begin());
		MinPair m_pair(i, score);
		if ((int)tmp_pqueue.size() >= REDUND * k && score <= tmp_pqueue.top().value)
			continue;
		tmp_pqueue.push(m_pair);
		if ((int)tmp_pqueue.size() > REDUND * k) {  // Kepp the top 10*k nodes with the maximum score
			tmp_pqueue.pop();
		}
	}

	priority_queue<Pair, vector<Pair>, greater<Pair> > pqueue;
	while (!tmp_pqueue.empty()) {
		MinPair min_pair = tmp_pqueue.top();
		tmp_pqueue.pop();
		Pair pair(min_pair.key, min_pair.value);
		pqueue.push(pair);
	}

	// find the top k nodes
	set<int> seed_set;  //seed set to be found
	
	int *updated = new int[g->num_nodes];  //the flag array indicates whehter marginal gain of a node is updated
	for (int i = 0; i < g->num_nodes; i++)
		updated[i] = 1;
	set<int> neigh_node_set;
	set<int> neigh_com_set;
	float total_score = 0;
	for (int i = 0; i < k; i++) {
		Pair best_pair = pqueue.top();
		pqueue.pop();
		while (!updated[best_pair.key]) {
			float m_gain = marginal_gain(g, n2c, neigh_node_set, neigh_com_set, best_pair.key, gamma);
			best_pair.value = m_gain;
			updated[best_pair.key] = 1;
			pqueue.push(best_pair);
			best_pair = pqueue.top();
			pqueue.pop();
		}
		add_seed(g, n2c, &seed_set, &neigh_node_set, &neigh_com_set, best_pair.key);
		total_score += best_pair.value;
		cout << i + 1 << "\t" << best_pair.key << "\t" << time(NULL) - time_start << endl;
		memset(updated, 0, g->num_nodes * sizeof(int)); // reset the flag array
	}
	
	delete[] n2c;
	delete[] updated;
	disp_mem_usage("");
	cout << "Time used: " << time(NULL) - time_start << " s" << endl;
}

float marginal_gain(Graph *g, int *n2c, set<int> node_set, set<int> com_set, int node, float gamma) {
	int k_out = g->node_array[node].k_out;
	int *p = g->node_array[node].id_array;
	set<int> tmp_node_set;
	set<int> tmp_com_set;
	for (int i = 0; i < k_out; i++) {
		int neigh_node = p[i];
		int neigh_com = n2c[neigh_node];
		if (node_set.find(neigh_node) == node_set.end()) {
			tmp_node_set.insert(neigh_node);
		}			
		if (com_set.find(neigh_com) == com_set.end()){
			tmp_com_set.insert(neigh_com);
		}			
	}
	float gain = tmp_node_set.size() + tmp_com_set.size() * gamma;
	return gain;
}

void add_seed(Graph *g, int *n2c, set<int> *seed_set, set<int> *node_set, set<int> *com_set, int node) {
	int k_out = g->node_array[node].k_out;
	int *p = g->node_array[node].id_array;
	for (int i = 0; i < k_out; i++) {
		int neigh_node = p[i];
		int neigh_com = n2c[neigh_node];
		node_set->insert(neigh_node);
		com_set->insert(neigh_com);
	}
	seed_set->insert(node);
}

float get_score(Graph *g, int *n2c, set<int> seed_set, float gamma) {
	set<int> neigh_node_set;
	set<int> neigh_com_set;
	set<int>::iterator it;
	for (it = seed_set.begin(); it != seed_set.end(); it++) {
		int seed = *it;
		int *p = g->node_array[seed].id_array;
		int k_out = g->node_array[seed].k_out;
		for (int i = 0; i < k_out; i++) {
			int neigh = p[i];
			int neigh_com = n2c[neigh];
			if (neigh_node_set.find(neigh) == neigh_node_set.end()) {
				neigh_node_set.insert(neigh);
				neigh_com_set.insert(neigh_com);
			}			
		}
	}
	int num_neigh = neigh_node_set.size();
	int num_com = neigh_com_set.size();
	float score = num_neigh + num_com * gamma;
	return score;
}

void evaluate_total(Graph *g, string data, string seeds, int k, int R){
	cout << "evaluating overall influence... data:" << data << " seeds:" << seeds << endl;
	int *seed_arr = new int[k];
	ifstream ifs(seeds.c_str());
	if(!ifs)
		cout << "seeds file: " << seeds << " cannot be openned!" << endl;
	else
		cout << "seeds file: " << seeds << " successfully openned!" << endl;
	string buffer;
	cout << "Seeds:";
	for(int i = 0; i < k; i++){
		ifs >> buffer;
		seed_arr[i] = atoi(buffer.c_str());
		cout << " " << seed_arr[i];
	}
	cout << endl;
	float total_inf = mc_influence(g, seed_arr, k, R);
	cout << "Total influence: " << total_inf << endl;
}

float mc_influence(Graph *g, int *seed_arr, int k, int simus){
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
	for(int r = 0; r < simus; r++){
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
	return inf / simus;
}

void analysis(Graph *g, string data, string com){
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Analysing graph and community ..." << endl;
    cout << "data:" << data << " com:" << com << endl;
	int max_k = 0;
	float avg_k = 0;
	for(int i = 0; i < g->num_nodes; i++){
		int k_out = g->node_array[i].k_out;
		avg_k += k_out;
		if(k_out > max_k)
			max_k = k_out;
	}
	avg_k /= g->num_nodes;
	int max_com = 0, min_com = 10000000;
	float avg_com = 0;
	ifstream ifs(com.c_str());
	if (!ifs){
		cout << "community file: " << com << " not openned!" << endl;
		return;
	}
	else
		cout << "community file: " << com << " successfully opened!" << endl;
	int *n2c = new int[g->num_nodes];
	string str;
	int com_id = 0;
	while (getline(ifs, str)){
		istringstream iss(str);
		string buffer;
		int com_size = 0;
		while (iss >> buffer) {
			int node = atoi(buffer.c_str());
			n2c[node] = com_id;
			com_size++;
		}
		com_id++;
		if(com_size < min_com)
			min_com = com_size;
		if(com_size > max_com)
			max_com = com_size;
		avg_com += com_size;
	}
	avg_com /= com_id;
	//compute modularity
	cout << "computing modularity ..." << endl;
	double *in = new double[com_id];
	double *tot = new double[com_id];
	memset(in, 0, com_id * sizeof(double));
	memset(tot, 0, com_id * sizeof(double));
	double mod = 0, m2 = 0;
	for(int i = 0; i < g->num_nodes; i++){
		int *p = g->node_array[i].id_array;
		int k_out = g->node_array[i].k_out;
		int src = i;
		int src_com = n2c[src];
		for(int j = 0; j < k_out; j++){
			int dest = p[j];
			int dest_com = n2c[dest];
			tot[src_com]++;
			m2++;
			if(src_com == dest_com)
				in[src_com]++;
		}
	}
	for(int i = 0; i < com_id; i++){
		if(tot[i] != 0){
			mod += in[i] / m2 + (tot[i] * tot[i]) / (m2 * m2);
		}
	}
	cout << "graph statistics: " << endl;
	cout << "# of nodes: " << g->num_nodes << endl;
	cout << "# of edges: " << g->num_edges << endl;
	cout << "max degree: " << max_k << endl;
	cout << "avg degree: " << avg_k << endl;
	cout << "community statistics: " << endl;
	cout << "# of communities: " << com_id << endl;
	cout << "max com size: " << max_com << endl;
	cout << "min com size: " << min_com << endl;
	cout << "avg com size: " << avg_com << endl;
	cout << "modularity: " << mod << endl;
	delete[] n2c;
	delete[] in;
	delete[] tot;
}
