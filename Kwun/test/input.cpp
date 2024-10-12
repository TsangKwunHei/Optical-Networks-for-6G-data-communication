#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
bool debug = false;
int N;
vector<int> Pi;
vector<vector<bool>> adjacency_list; // Adjacency list representation of the graph
vector<int> path_net;
struct service {
    int s, d, S, L, R, V;
    vector<int> path;
    vector<bool> hold_resource_path;
};
vector<service> init_Services; 


void debug_log(const string &message) {
    if (debug) {
        cout << message << endl;
    }
}

void Initialize_Environment() {
    ifstream infile("input.txt");
    if (!infile) {
        cout << "Error opening input file." << endl;
        return;
    }
    int M; 
    infile >> N >> M;
    
    debug_log("Number of nodes (N): " + to_string(N));
    debug_log("Number of edges (M): " + to_string(M));

    // Read Pi (maximum channel conversion opportunities per node)
    for (int i = 0; i < N; i++) {
        int pi;
        infile >> pi;
        Pi.push_back(pi);
        debug_log("Pi[" + to_string(i) + "] = " + to_string(pi));
    }

    // Initialize adjacency_list
    adjacency_list.resize(N, vector<bool>(N, false));
    debug_log("Initialized adjacency list of size " + to_string(N) + " x " + to_string(N));

    // Read edges and fill adjacency_list
    vector<pair<int, int>> edges(M + 1); // Edge indices start from 1
    for (int i = 1; i <= M; i++) {
        int u, v;
        infile >> u >> v;
        u--; v--; // Adjust to 0-based indexing
        adjacency_list[u][v] = true;
        adjacency_list[v][u] = true; // Assuming undirected graph
        edges[i] = make_pair(u, v);

        debug_log("Edge " + to_string(i) + ": (" + to_string(u + 1) + ", " + to_string(v + 1) + ")");
    }

    // Create path_net
    for (int i = 1; i <= M; i++) {
        int u = edges[i].first;
        int v = edges[i].second;
        path_net.push_back(u);
        path_net.push_back(v);

        debug_log("Path net edge: (" + to_string(u + 1) + ", " + to_string(v + 1) + ")");
    }

    // Read the number of initial services
    int K;
    infile >> K;
    debug_log("Number of services (K): " + to_string(K));

    for (int i = 0; i < K; i++) {
        service temp_service;
        infile >> temp_service.s >> temp_service.d >> temp_service.S >> temp_service.L >> temp_service.R >> temp_service.V;
        temp_service.s--; // Adjust to 0-based indexing
        temp_service.d--;

        debug_log("Service " + to_string(i + 1) + ": Source = " + to_string(temp_service.s + 1) + 
                  ", Destination = " + to_string(temp_service.d + 1) +
                  ", Number of edges = " + to_string(temp_service.S) + 
                  ", L = " + to_string(temp_service.L) + 
                  ", R = " + to_string(temp_service.R) + 
                  ", V = " + to_string(temp_service.V));

        // Read the sequence of edge numbers for the service path
        for (int j = 0; j < temp_service.S; j++) {
            int edge_num;
            infile >> edge_num;
            temp_service.path.push_back(edge_num);
            temp_service.hold_resource_path.push_back(true);
            
            debug_log("Service " + to_string(i + 1) + " edge number in path: " + to_string(edge_num));
        }
        init_Services.push_back(temp_service);
    }
    infile.close();

    debug_log("Environment initialization completed.");
}

int main() {

    debug = true; 
    
    Initialize_Environment(); 

    return 0;
}