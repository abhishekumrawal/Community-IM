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
	cout << "Start Degree Heursitic algorithm" << endl;
	time_t time_start = time(NULL);
	cout << "No.\tnode_id\ttime(s)\tdegree" << endl;
	priority_queue<MinPair, vector<MinPair>, greater<MinPair> > tmp_pqueue;
	for (int i = 0; i < g->num_nodes; i++){
		int k_out = g->node_array[i].k_out;
		MinPair m_pair(i, k_out);
		tmp_pqueue.push(m_pair);
		if ((int)tmp_pqueue.size() > k) {  // Kepp the top 10*k nodes with the maximum score
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

	for(int i = 0; i < k; i++){
		Pair pair = pqueue.top();
		pqueue.pop();
		cout << i+1 << "\t" << pair.key << "\t" << time(NULL) - time_start << "\t" << pair.value << endl;
	}
	disp_mem_usage("");
	cout << "Time used: " << time(NULL) - time_start << " s" << endl;
}

