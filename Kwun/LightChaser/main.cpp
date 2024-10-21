#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <cstdio>
#include <set>
#include <map>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <functional>
#include <random>
#include <bitset>
#include <utility>
#include <unordered_set>
#include <stack>

using namespace std;

// Constants
const int MAX_K = 40;
const int MAX_N = 200;
const int MAX_M = 1000;

// Service structure representing an optical service in the network
struct Service {
    int id;  // Service ID
    int s;   // Source node
    int d;   // Destination node
    int S;   // Number of edges traversed (path length)
    int L;   // Occupied wavelength range start
    int R;   // Occupied wavelength range end
    int V;   // Service value
    vector<int> path;        // Sequence of edge indices the service traverses
    bool alive = true;       // Whether the service is active
    bool need_to_replan = false;  // Whether the service needs to be replanned
    vector<int> wavelengths; // Start wavelength on each edge

    // Initial paths and wavelengths
    vector<int> initial_path;
    vector<int> initial_wavelengths;
};

// Structure representing a failure scenario with a sequence of edge failures
struct FailureScenario {
    int ci;                 // Number of edge failures in the scenario
    vector<int> edge_ids;   // Edge IDs of the failed edges
};

// Global variables representing the network environment
int N, M;  // Number of nodes and edges
vector<int> Pi;  // Channel conversion opportunities at nodes
vector<int> initialPi; // Initial channel conversion opportunities
vector<vector<pair<int, int>>> adjacency_list;  // Adjacency list of the network graph (neighbor node, edge ID)
vector<pair<int, int>> edges;  // List of edges (edge index to node pairs)
vector<bitset<MAX_K + 1>> edge_wavelengths;  // Edge ID to occupied wavelengths
vector<bitset<MAX_K + 1>> initial_edge_wavelengths;  // Initial wavelengths on edges
int K;  // Number of initial services
vector<Service> services;  // List of services
int T1 = 0;  // Number of edge failure scenarios we provide (will be set later)
vector<FailureScenario> failure_scenarios;  // Edge failure scenarios we generate

// GA parameters (Adjusted for performance)
const int POPULATION_SIZE = 10;      // Reduced population size
const int MAX_GENERATIONS = 10;      // Reduced number of generations
const double CROSSOVER_RATE = 0.7;   // Adjusted crossover rate
const double MUTATION_RATE = 0.2;    // Adjusted mutation rate
const int N_ELITE = 2;               // Number of elite individuals

// Function declarations
void InitializeEnvironment();
void OutputEdgeFailureScenarios();
void GenerateFailureScenarios();
void HandleTestScenarios();
void ResetEnvironment();
void HandleEdgeFailure(int e);
vector<int> DetectAffectedServices(int e);
vector<int> AssignWavelengthsUsingGA(vector<int>& affected_services);
bool RunGAWithTemp(Service& srv, vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, vector<int>& temp_Pi);
double FitnessFunction(const vector<int>& chromosome, const Service& srv, const vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, const vector<int>& temp_Pi);
vector<int> GenerateInitialChromosome(const Service& srv);
vector<int> MutateChromosome(const vector<int>& chromosome, const Service& srv);
pair<vector<int>, vector<int>> CrossoverChromosomes(const vector<int>& parent1, const vector<int>& parent2, const Service& srv);
bool DecodeChromosomeWithTemp(const vector<int>& chromosome, const Service& srv, vector<int>& path, vector<int>& wavelengths, map<int, int>& converters_needed, const vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, const vector<int>& temp_Pi);
double JaccardSimilarity(const set<int>& a, const set<int>& b);
vector<int> ComputeMinCut(int s, int d);
vector<int> GenerateRandomPath(int s, int d);
vector<int> GenerateInitialChromosomeBetweenNodes(int s, int d);
bool IsValidPath(const vector<int>& chromosome, const Service& srv);
bool HasDuplicateEdgesOrNodes(const vector<int>& chromosome, const Service& srv);

// Fast Max-Flow using Dinic's Algorithm
struct EdgeFlow {
    int to, rev;
    int cap;
};

class MaxFlow {
public:
    int N;
    vector<vector<EdgeFlow>> graph;
    vector<int> level;
    vector<int> ptr;

    MaxFlow(int N_) : N(N_), graph(N + 1), level(N + 1, -1), ptr(N + 1, 0) {}

    void add_edge(int from, int to, int cap) {
        EdgeFlow a = {to, (int)graph[to].size(), cap};
        EdgeFlow b = {from, (int)(graph[from].size()), 0};
        graph[from].push_back(a);
        graph[to].push_back(b);
    }

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        q.push(s);
        level[s] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : graph[u]) {
                if (e.cap > 0 && level[e.to] == -1) {
                    level[e.to] = level[u] + 1;
                    q.push(e.to);
                    if (e.to == t) return true;
                }
            }
        }
        return false;
    }

    int dfs(int u, int t, int pushed) {
        if (u == t) return pushed;
        for (; ptr[u] < graph[u].size(); ++ptr[u]) {
            EdgeFlow& e = graph[u][ptr[u]];
            if (e.cap > 0 && level[e.to] == level[u] + 1) {
                int tr = dfs(e.to, t, min(pushed, e.cap));
                if (tr > 0) {
                    e.cap -= tr;
                    graph[e.to][e.rev].cap += tr;
                    return tr;
                }
            }
        }
        return 0;
    }

    int max_flow(int s, int t) {
        int flow = 0;
        while (bfs(s, t)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(s, t, INT32_MAX)) {
                flow += pushed;
            }
        }
        return flow;
    }
};

// Main function
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    srand(time(NULL));  // Initialize random seed

    // Initialize the network environment by reading input data
    InitializeEnvironment();

    // Generate edge failure scenarios designed to be bottleneck cases
    GenerateFailureScenarios();

    // Output the edge failure scenarios we provide
    OutputEdgeFailureScenarios();

    // Handle the test scenarios provided by the system
    HandleTestScenarios();

    return 0;
}

// Function to initialize the environment by reading input data
void InitializeEnvironment() {
    // Read N (number of nodes) and M (number of edges)
    cin >> N >> M;

    // Read Pi (channel conversion opportunities at each node)
    Pi.resize(N + 1);
    initialPi.resize(N + 1);
    for (int i = 1; i <= N; ++i) {
        cin >> Pi[i];
        initialPi[i] = Pi[i];
    }

    // Initialize adjacency list and read edges
    adjacency_list.resize(N + 1);
    edges.resize(M + 1);
    for (int i = 1; i <= M; ++i) {
        int u, v;
        cin >> u >> v;
        adjacency_list[u].emplace_back(v, i);
        adjacency_list[v].emplace_back(u, i);
        edges[i] = {u, v};
    }

    // Read K (number of initial services)
    cin >> K;
    services.resize(K + 1); // 1-based indexing

    // Initialize edge wavelengths
    edge_wavelengths.resize(M + 1);
    initial_edge_wavelengths.resize(M + 1);

    // Read details for each service
    for (int i = 1; i <= K; ++i) {
        Service& srv = services[i];
        srv.id = i;
        cin >> srv.s >> srv.d >> srv.S >> srv.L >> srv.R >> srv.V;
        srv.path.resize(srv.S);
        srv.wavelengths.resize(srv.S, srv.L); // Initialize wavelengths
        // Read the sequence of edge indices
        for (int j = 0; j < srv.S; ++j) {
            cin >> srv.path[j];
        }
        // Store initial paths and wavelengths
        srv.initial_path = srv.path;
        srv.initial_wavelengths = srv.wavelengths;
        // Update edge_wavelengths to mark wavelengths occupied
        for (int j = 0; j < srv.S; ++j) {
            int edge_id = srv.path[j];
            for (int w = srv.L; w <= srv.R; ++w) {
                edge_wavelengths[edge_id].set(w);
                initial_edge_wavelengths[edge_id].set(w);
            }
        }
    }
}

// Function to compute the min-cut between two nodes using Dinic's algorithm
vector<int> ComputeMinCut(int s, int d) {
    MaxFlow mf(N);
    for (int i = 1; i <= M; ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        mf.add_edge(u, v, 1);
        mf.add_edge(v, u, 1);
    }
    mf.max_flow(s, d);

    // Find reachable nodes from s in the residual graph
    vector<bool> visited(N + 1, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (auto& e : mf.graph[u]) {
            if (e.cap > 0 && !visited[e.to]) {
                visited[e.to] = true;
                q.push(e.to);
            }
        }
    }

    // Edges that go from visited to unvisited are in the min-cut
    vector<int> min_cut_edges;
    for (int i = 1; i <= M; ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        if (visited[u] && !visited[v]) {
            min_cut_edges.push_back(i);
        }
        if (visited[v] && !visited[u]) {
            min_cut_edges.push_back(i);
        }
    }
    return min_cut_edges;
}

// Function to compute Jaccard similarity between two sets
double JaccardSimilarity(const set<int>& a, const set<int>& b) {
    if (a.empty() && b.empty()) return 0.0;
    int intersection_size = 0;
    for (auto& elem : a) {
        if (b.find(elem) != b.end()) intersection_size++;
    }
    int union_size = a.size() + b.size() - intersection_size;
    return (double)intersection_size / union_size;
}

// Function to generate edge failure scenarios using min-cut
void GenerateFailureScenarios() {
    T1 = min(30, K);  // Ensure T1 is between 0 and 30

    failure_scenarios.clear();

    // Sort services by value
    vector<int> service_indices(K);
    for (int i = 0; i < K; ++i) {
        service_indices[i] = i + 1;
    }

    sort(service_indices.begin(), service_indices.end(), [&](int a, int b) {
        return services[a].V > services[b].V;
    });

    set<set<int>> scenario_edge_sets;

    for (auto idx : service_indices) {
        if ((int)failure_scenarios.size() >= T1) {
            break;
        }

        const Service& srv = services[idx];
        int s = srv.s;
        int d = srv.d;

        vector<int> min_cut_edges = ComputeMinCut(s, d);

        if (min_cut_edges.empty()) continue;

        // Limit the number of edge failures per scenario
        if ((int)min_cut_edges.size() > 60) {
            min_cut_edges.resize(60);
        }

        set<int> current_edges(min_cut_edges.begin(), min_cut_edges.end());

        // Check for duplicates or similar scenarios
        bool similar = false;
        for (auto& edge_set : scenario_edge_sets) {
            double similarity = JaccardSimilarity(edge_set, current_edges);
            if (similarity > 0.5) {
                similar = true;
                break;
            }
        }
        if (similar) continue;

        FailureScenario fs;
        fs.ci = min_cut_edges.size();
        fs.edge_ids = min_cut_edges;

        failure_scenarios.push_back(fs);
        scenario_edge_sets.insert(current_edges);
    }

    T1 = failure_scenarios.size();  // Update T1 in case we have fewer scenarios
}

// Function to output edge failure scenarios
void OutputEdgeFailureScenarios() {
    // Output T1, the number of edge failure test scenarios we provide
    cout << T1 << "\n";
    // Output each scenario as per the required format
    for (auto& fs : failure_scenarios) {
        cout << fs.ci << "\n";
        for (int i = 0; i < fs.edge_ids.size(); ++i) {
            cout << fs.edge_ids[i] << (i < fs.edge_ids.size() - 1 ? " " : "\n");
        }
    }
    cout.flush();
}

// Function to reset the environment to its initial state
void ResetEnvironment() {
    // Reset services to their initial state
    for (int i = 1; i <= K; ++i) {
        Service& srv = services[i];
        srv.alive = true;
        srv.need_to_replan = false;
        srv.path = srv.initial_path;
        srv.wavelengths = srv.initial_wavelengths;
    }

    // Reset Pi[i]
    for (int i = 1; i <= N; ++i) {
        Pi[i] = initialPi[i];
    }

    // Reset adjacency list
    adjacency_list.assign(N + 1, vector<pair<int, int>>());
    for (int i = 1; i <= M; ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        adjacency_list[u].emplace_back(v, i);
        adjacency_list[v].emplace_back(u, i);
    }

    // Reset edge wavelengths
    edge_wavelengths = initial_edge_wavelengths;
}

// Function to handle an edge failure in the network
void HandleEdgeFailure(int e) {
    // Remove the failed edge from the adjacency list
    int u = edges[e].first;
    int v = edges[e].second;
    // Remove edge e from u's adjacency list
    adjacency_list[u].erase(
        remove_if(adjacency_list[u].begin(), adjacency_list[u].end(),
                  [&](const pair<int, int>& p) { return p.first == v && p.second == e; }),
        adjacency_list[u].end()
    );
    // Remove edge e from v's adjacency list
    adjacency_list[v].erase(
        remove_if(adjacency_list[v].begin(), adjacency_list[v].end(),
                  [&](const pair<int, int>& p) { return p.first == u && p.second == e; }),
        adjacency_list[v].end()
    );
}

// Function to detect services affected by an edge failure
vector<int> DetectAffectedServices(int e) {
    vector<int> affected_services;
    for (int i = 1; i <= K; ++i) {
        if (!services[i].alive) continue;
        if (find(services[i].path.begin(), services[i].path.end(), e) != services[i].path.end()) {
            services[i].need_to_replan = true;
            affected_services.push_back(i);
        }
    }
    return affected_services;
}

// Function to assign wavelengths using GA with temporary edge_wavelengths
vector<int> AssignWavelengthsUsingGA(vector<int>& affected_services) {
    vector<int> successfully_replanned_services;

    // Create a temporary copy of edge_wavelengths to avoid conflicts
    vector<bitset<MAX_K + 1>> temp_edge_wavelengths = edge_wavelengths;

    // Create a temporary copy of Pi
    vector<int> temp_Pi = Pi;

    // Sort services by value (higher value first)
    sort(affected_services.begin(), affected_services.end(), [&](int a, int b) {
        return services[a].V > services[b].V;
    });

    // Run GA for each affected service
    for (auto service_id : affected_services) {
        Service& srv = services[service_id];
        // Run GA to find the best path and wavelength assignment using temp_edge_wavelengths and temp_Pi
        bool replanned = RunGAWithTemp(srv, temp_edge_wavelengths, temp_Pi);
        if (replanned) {
            successfully_replanned_services.push_back(service_id);
            // Update temp_edge_wavelengths with the new assignments
            for (int j = 0; j < srv.path.size(); ++j) {
                int edge_id = srv.path[j];
                for (int w = srv.wavelengths[j]; w <= srv.wavelengths[j] + (srv.R - srv.L); ++w) {
                    temp_edge_wavelengths[edge_id].set(w);
                }
            }
            // Channel conversion opportunities updated in RunGAWithTemp
        }
        else {
            // Service could not be replanned
            srv.alive = false;
            srv.need_to_replan = false;
        }
    }

    // After all services are replanned, update the global edge_wavelengths and Pi
    edge_wavelengths = temp_edge_wavelengths;
    Pi = temp_Pi;

    return successfully_replanned_services;
}

// Function to check if a path is valid (no cycles or duplicate edges/nodes)
bool IsValidPath(const vector<int>& chromosome, const Service& srv) {
    return !HasDuplicateEdgesOrNodes(chromosome, srv);
}

bool HasDuplicateEdgesOrNodes(const vector<int>& chromosome, const Service& srv) {
    unordered_set<int> visited_nodes; // Use unordered_set for faster lookup
    unordered_set<int> visited_edges;

    int current_node = srv.s;
    visited_nodes.insert(current_node);

    for (auto edge_id : chromosome) {
        if (visited_edges.count(edge_id)) {
            return true; // Duplicate edge
        }
        visited_edges.insert(edge_id);

        int u = edges[edge_id].first;
        int v = edges[edge_id].second;
        int next_node;
        if (u == current_node) {
            next_node = v;
        }
        else if (v == current_node) {
            next_node = u;
        }
        else {
            return true; // Disconnected edge
        }

        if (visited_nodes.count(next_node)) {
            return true; // Cycle detected
        }
        visited_nodes.insert(next_node);
        current_node = next_node;
    }
    return false;
}

// Function to run GA for a single service using temporary edge_wavelengths and temp_Pi
bool RunGAWithTemp(Service& srv, vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, vector<int>& temp_Pi) {
    // Initialize population
    vector<vector<int>> population;
    vector<double> fitness_values;

    // Generate initial population
    int attempts = 0;
    while (population.size() < POPULATION_SIZE && attempts < POPULATION_SIZE * 2) {
        vector<int> chromosome = GenerateInitialChromosome(srv);
        attempts++;
        if (chromosome.empty()) continue;
        if (!IsValidPath(chromosome, srv)) continue;
        population.push_back(chromosome);
        fitness_values.push_back(FitnessFunction(chromosome, srv, temp_edge_wavelengths, temp_Pi));
    }

    if (population.empty()) return false;

    // Evolutionary loop
    for (int generation = 0; generation < MAX_GENERATIONS; ++generation) {
        // Sort population based on fitness values
        vector<pair<double, vector<int>>> pop_fitness;
        for (int i = 0; i < population.size(); ++i) {
            pop_fitness.emplace_back(fitness_values[i], population[i]);
        }
        sort(pop_fitness.begin(), pop_fitness.end());
        // Elitism - keep the best N_ELITE individuals
        vector<vector<int>> new_population;
        for (int i = 0; i < N_ELITE && i < pop_fitness.size(); ++i) {
            new_population.push_back(pop_fitness[i].second);
        }
        // Selection and reproduction
        while (new_population.size() < POPULATION_SIZE) {
            // Tournament selection
            int idx1 = rand() % population.size();
            int idx2 = rand() % population.size();
            vector<int> parent1 = fitness_values[idx1] < fitness_values[idx2] ? population[idx1] : population[idx2];

            idx1 = rand() % population.size();
            idx2 = rand() % population.size();
            vector<int> parent2 = fitness_values[idx1] < fitness_values[idx2] ? population[idx1] : population[idx2];

            // Crossover
            vector<int> child1, child2;
            tie(child1, child2) = CrossoverChromosomes(parent1, parent2, srv);

            // Mutation
            if (((double)rand() / RAND_MAX) < MUTATION_RATE) {
                child1 = MutateChromosome(child1, srv);
            }
            if (((double)rand() / RAND_MAX) < MUTATION_RATE) {
                child2 = MutateChromosome(child2, srv);
            }

            // Validate and add to new population
            if (!child1.empty() && IsValidPath(child1, srv)) new_population.push_back(child1);
            if (new_population.size() >= POPULATION_SIZE) break;
            if (!child2.empty() && IsValidPath(child2, srv)) new_population.push_back(child2);
        }

        // Evaluate new population
        population = new_population;
        fitness_values.clear();
        for (auto& chrom : population) {
            fitness_values.push_back(FitnessFunction(chrom, srv, temp_edge_wavelengths, temp_Pi));
        }
    }

    // Get the best chromosome
    int best_idx = distance(fitness_values.begin(), min_element(fitness_values.begin(), fitness_values.end()));
    vector<int> best_chromosome = population[best_idx];

    // Decode the best chromosome
    vector<int> path;
    vector<int> wavelengths;
    map<int, int> converters_needed;
    bool valid = DecodeChromosomeWithTemp(best_chromosome, srv, path, wavelengths, converters_needed, temp_edge_wavelengths, temp_Pi);
    if (valid) {
        srv.path = path;
        srv.wavelengths = wavelengths;
        // Update temp_Pi with converters_used
        for (auto& kv : converters_needed) {
            temp_Pi[kv.first] -= kv.second;
        }
        return true;
    }
    return false;
}

// Fitness function for the GA with temporary edge_wavelengths and temp_Pi
double FitnessFunction(const vector<int>& chromosome, const Service& srv, const vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, const vector<int>& temp_Pi) {
    // Decode chromosome to get path and wavelength assignment
    vector<int> path;
    vector<int> wavelengths;
    map<int, int> converters_needed;
    bool valid = DecodeChromosomeWithTemp(chromosome, srv, path, wavelengths, converters_needed, temp_edge_wavelengths, temp_Pi);
    if (!valid) return 1e9;  // High penalty for invalid chromosome

    // Fitness considers path length and number of wavelength changes
    int num_conversions = 0;
    for (auto& kv : converters_needed) {
        num_conversions += kv.second;
    }

    return path.size() + num_conversions * 10.0;
}

// Function to generate an initial chromosome for a service using randomized shortest paths
vector<int> GenerateInitialChromosome(const Service& srv) {
    // Assign random weights to edges
    vector<double> edge_weights(M + 1);
    for (int i = 1; i <= M; ++i) {
        edge_weights[i] = ((double)rand() / RAND_MAX) + 1.0; // Random weight between 1 and 2
    }

    // Run Dijkstra's algorithm
    vector<double> dist(N + 1, 1e9);
    vector<int> prev_edge(N + 1, -1);
    vector<int> prev_node(N + 1, -1);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    dist[srv.s] = 0;
    pq.push({0, srv.s});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;

        if (u == srv.d) break;

        for (auto& [v, edge_id] : adjacency_list[u]) {
            double w = edge_weights[edge_id];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                prev_edge[v] = edge_id;
                prev_node[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    if (dist[srv.d] >= 1e9) return {};

    // Reconstruct path
    vector<int> path;
    int u = srv.d;
    while (u != srv.s) {
        int edge_id = prev_edge[u];
        if (edge_id == -1) break;
        path.push_back(edge_id);
        u = prev_node[u];
    }
    reverse(path.begin(), path.end());
    return path;
}

// Function to generate an initial chromosome between two nodes
vector<int> GenerateInitialChromosomeBetweenNodes(int s, int d) {
    // Assign random weights to edges
    vector<double> edge_weights(M + 1);
    for (int i = 1; i <= M; ++i) {
        edge_weights[i] = ((double)rand() / RAND_MAX) + 1.0; // Random weight between 1 and 2
    }

    // Run Dijkstra's algorithm
    vector<double> dist(N + 1, 1e9);
    vector<int> prev_edge(N + 1, -1);
    vector<int> prev_node(N + 1, -1);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    dist[s] = 0;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;

        if (u == d) break;

        for (auto& [v, edge_id] : adjacency_list[u]) {
            double w = edge_weights[edge_id];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                prev_edge[v] = edge_id;
                prev_node[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    if (dist[d] >= 1e9) return {};

    // Reconstruct path
    vector<int> path;
    int u = d;
    while (u != s) {
        int edge_id = prev_edge[u];
        if (edge_id == -1) break;
        path.push_back(edge_id);
        u = prev_node[u];
    }
    reverse(path.begin(), path.end());
    return path;
}

// Function to mutate a chromosome
vector<int> MutateChromosome(const vector<int>& chromosome, const Service& srv) {
    if (chromosome.empty()) return chromosome;

    int path_length = chromosome.size();
    if (path_length <= 1) return chromosome;

    // Select a segment to mutate
    int start_idx = rand() % path_length;
    int end_idx = start_idx + rand() % (path_length - start_idx);

    // Get the nodes at the start and end of the segment
    int start_node = srv.s;
    for (int i = 0; i < start_idx; ++i) {
        int edge_id = chromosome[i];
        if (edges[edge_id].first == start_node) {
            start_node = edges[edge_id].second;
        }
        else {
            start_node = edges[edge_id].first;
        }
    }

    int end_node = start_node;
    for (int i = start_idx; i <= end_idx; ++i) {
        int edge_id = chromosome[i];
        if (edges[edge_id].first == end_node) {
            end_node = edges[edge_id].second;
        }
        else {
            end_node = edges[edge_id].first;
        }
    }

    // Generate a new path between start_node and end_node
    vector<int> new_subpath = GenerateInitialChromosomeBetweenNodes(start_node, end_node);

    if (new_subpath.empty()) return chromosome; // Mutation failed

    // Construct the new chromosome
    vector<int> mutated;
    mutated.insert(mutated.end(), chromosome.begin(), chromosome.begin() + start_idx);
    mutated.insert(mutated.end(), new_subpath.begin(), new_subpath.end());
    mutated.insert(mutated.end(), chromosome.begin() + end_idx + 1, chromosome.end());

    // Validate the mutated chromosome
    if (!IsValidPath(mutated, srv)) {
        return chromosome; // Return original if mutation results in invalid path
    }
    return mutated;
}

// Function to perform crossover between two chromosomes
pair<vector<int>, vector<int>> CrossoverChromosomes(const vector<int>& parent1, const vector<int>& parent2, const Service& srv) {
    if (((double)rand() / RAND_MAX) > CROSSOVER_RATE) {
        return {parent1, parent2};
    }

    // Find common nodes between the two paths
    vector<int> nodes1, nodes2;
    int current_node = srv.s;
    nodes1.push_back(current_node);
    for (int edge_id : parent1) {
        if (edges[edge_id].first == current_node) {
            current_node = edges[edge_id].second;
        }
        else {
            current_node = edges[edge_id].first;
        }
        nodes1.push_back(current_node);
    }
    current_node = srv.s;
    nodes2.push_back(current_node);
    for (int edge_id : parent2) {
        if (edges[edge_id].first == current_node) {
            current_node = edges[edge_id].second;
        }
        else {
            current_node = edges[edge_id].first;
        }
        nodes2.push_back(current_node);
    }

    // Find common nodes
    unordered_set<int> nodes_set(nodes1.begin(), nodes1.end());
    vector<int> common_nodes;
    for (int node : nodes2) {
        if (nodes_set.find(node) != nodes_set.end()) {
            common_nodes.push_back(node);
        }
    }
    if (common_nodes.empty()) {
        // No common node, cannot crossover
        return {parent1, parent2};
    }

    // Select a common node at random
    int common_node = common_nodes[rand() % common_nodes.size()];

    // Find positions of the common node in both paths
    auto it1 = find(nodes1.begin(), nodes1.end(), common_node);
    auto it2 = find(nodes2.begin(), nodes2.end(), common_node);

    int idx1 = distance(nodes1.begin(), it1);
    int idx2 = distance(nodes2.begin(), it2);

    // Build new child paths
    vector<int> child1_edges, child2_edges;
    // For child1, take edges from parent1 up to idx1-1, then edges from parent2 from idx2 to end
    child1_edges.insert(child1_edges.end(), parent1.begin(), parent1.begin() + idx1);
    child1_edges.insert(child1_edges.end(), parent2.begin() + idx2, parent2.end());
    // For child2, take edges from parent2 up to idx2-1, then edges from parent1 from idx1 to end
    child2_edges.insert(child2_edges.end(), parent2.begin(), parent2.begin() + idx2);
    child2_edges.insert(child2_edges.end(), parent1.begin() + idx1, parent1.end());

    // Validate the child chromosomes
    if (!IsValidPath(child1_edges, srv)) {
        child1_edges = parent1; // Revert to parent if invalid
    }
    if (!IsValidPath(child2_edges, srv)) {
        child2_edges = parent2; // Revert to parent if invalid
    }

    return {child1_edges, child2_edges};
}

// Function to decode a chromosome into a path and wavelength assignment using temporary edge_wavelengths and temp_Pi
bool DecodeChromosomeWithTemp(const vector<int>& chromosome, const Service& srv, vector<int>& path, vector<int>& wavelengths, map<int, int>& converters_needed, const vector<bitset<MAX_K + 1>>& temp_edge_wavelengths, const vector<int>& temp_Pi) {
    path.clear();
    wavelengths.clear();
    converters_needed.clear();
    // Convert chromosome into a path
    int current_node = srv.s;
    for (auto edge_id : chromosome) {
        int u = edges[edge_id].first;
        int v = edges[edge_id].second;
        if (u == current_node) {
            path.push_back(edge_id);
            current_node = v;
        }
        else if (v == current_node) {
            path.push_back(edge_id);
            current_node = u;
        }
        else {
            // Invalid path
            return false;
        }
        if (current_node == srv.d) break;
    }
    if (current_node != srv.d) return false;

    // Assign wavelengths
    int width = srv.R - srv.L + 1;
    int prev_wavelength = -1;
    for (int i = 0; i < path.size(); ++i) {
        int edge_id = path[i];
        // Find available wavelength
        int assigned_wavelength = -1;
        // Try to maintain the same wavelength as previous if possible
        if (prev_wavelength != -1) {
            bool available = true;
            for (int w = prev_wavelength; w < prev_wavelength + width; ++w) {
                if (w > MAX_K || temp_edge_wavelengths[edge_id].test(w)) {
                    available = false;
                    break;
                }
            }
            if (available) {
                assigned_wavelength = prev_wavelength;
            }
        }
        // If not possible, find the first available wavelength
        if (assigned_wavelength == -1) {
            for (int w = 1; w <= MAX_K - width + 1; ++w) {
                bool available = true;
                for (int offset = 0; offset < width; ++offset) {
                    if (temp_edge_wavelengths[edge_id].test(w + offset)) {
                        available = false;
                        break;
                    }
                }
                if (available) {
                    assigned_wavelength = w;
                    break;
                }
            }
        }
        if (assigned_wavelength == -1) {
            return false; // No available wavelength
        }
        wavelengths.push_back(assigned_wavelength);
        prev_wavelength = assigned_wavelength;
    }

    // Collect the number of converters needed at each node
    prev_wavelength = wavelengths.empty() ? -1 : wavelengths[0];
    for (int i = 1; i < wavelengths.size(); ++i) {
        if (wavelengths[i] != wavelengths[i - 1]) {
            // Channel conversion needed at node between edge i-1 and i
            int edge_prev = path[i - 1];
            int edge_curr = path[i];
            // Find the common node
            int common_node = -1;
            if (edges[edge_prev].first == edges[edge_curr].first || edges[edge_prev].first == edges[edge_curr].second) {
                common_node = edges[edge_prev].first;
            }
            else {
                common_node = edges[edge_prev].second;
            }
            if (common_node == -1) {
                return false; // Invalid path
            }
            converters_needed[common_node]++;
            prev_wavelength = wavelengths[i];
        }
    }

    // Now check if we have enough channel conversion opportunities at each node
    for (auto& kv : converters_needed) {
        int node = kv.first;
        int required = kv.second;
        if (temp_Pi[node] < required) {
            return false; // Not enough channel conversion opportunities
        }
    }

    return true;
}

// Function to handle test scenarios provided by the system
void HandleTestScenarios() {
    while (true) {
        int T;
        if (!(cin >> T)) {
            break; // No more input
        }
        for (int t = 0; t < T; ++t) {
            // Reset the environment to the initial state before each scenario
            ResetEnvironment();

            while (true) {
                int e;
                cin >> e;
                if (e == -1) {
                    break; // End of the test scenario
                }
                // Handle the edge failure in the network
                HandleEdgeFailure(e);

                // Detect services affected by this edge failure
                vector<int> affected_services = DetectAffectedServices(e);

                // Replan services affected by the current edge failure
                vector<int> successfully_replanned_services = AssignWavelengthsUsingGA(affected_services);

                // Output the number of services successfully replanned
                int K_prime = successfully_replanned_services.size();
                cout << K_prime << "\n";

                // Output details of each successfully replanned service
                for (auto service_id : successfully_replanned_services) {
                    Service& srv = services[service_id];
                    cout << service_id << " " << srv.path.size() << "\n";
                    // Output path and wavelength intervals
                    for (int j = 0; j < srv.path.size(); ++j) {
                        cout << srv.path[j] << " " << srv.wavelengths[j] << " " << srv.wavelengths[j] + (srv.R - srv.L) << (j < srv.path.size() - 1 ? " " : "\n");
                    }
                }
                cout.flush();
            }
        }
    }
}