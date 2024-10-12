#include <vector>

#ifndef _STRUCTS_H_
#define _STRUCTS_H_

// Service structure representing an optical service in the network
struct Service {
  int id;                     // Service ID
  int s;                      // Source node
  int d;                      // Destination node
  int S;                      // Number of edges traversed (path length)
  int L;                      // Occupied wavelength range start
  int R;                      // Occupied wavelength range end
  int V;                      // Service value
  std::vector<int> path;      // Sequence of edge indices the service traverses
  bool alive{true};           // Whether the service is active
  bool need_to_replan{false}; // Whether the service needs to be replanned
  std::vector<int> wavelengths; // Start wavelength on each edge

  // Initial paths and wavelengths
  std::vector<int> initial_path;
  std::vector<int> initial_wavelengths;
};
// Structure representing a failure scenario with a sequence of edge failures
struct FailureScenario {
  int ci;
  std::vector<int> edge_ids;
};

#endif
#include <bitset>
#include <vector>

#ifndef _GLOBALS_H_
#define _GLOBALS_H_
// Constants
constexpr int MAX_K = 40;
constexpr int MAX_N = 200;
constexpr int MAX_M = 1000;

// GA parameters
constexpr int POPULATION_SIZE = 50;
constexpr int MAX_GENERATIONS = 50;
constexpr double CROSSOVER_RATE = 0.9;
constexpr double MUTATION_RATE = 0.1;
// Global variables representing the network environment
int N = 0, M = 0;
std::vector<int> Pi;
std::vector<int> initialPi;
std::vector<std::vector<std::pair<int, int>>>
    adjacency_list; // Adjacency list of the network graph (neighbor node, edge
                    // ID)
std::vector<std::pair<int, int>>
    edges; // List of edges (edge index to node pairs)
std::vector<std::bitset<MAX_K + 1>>
    edge_wavelengths; // Edge ID to occupied wavelengths
std::vector<std::bitset<MAX_K + 1>>
    initial_edge_wavelengths; // Initial wavelengths on edges
int K = 0;                    // Number of initial services
std::vector<Service> services;
int T1 = 0;
std::vector<FailureScenario>
    failure_scenarios; // Edge failure scenarios we generate

#endif // _GLOBALS_H_
#include <iostream>
#include <vector>

// Function to initialize the environment by reading input data
void InitializeEnvironment() {
  // Read N (number of nodes) and M (number of edges)
  std::cin >> N >> M;

  // Read Pi (channel conversion opportunities at each node)
  Pi.resize(N + 1);
  initialPi.resize(N + 1);
  for (int i = 1; i <= N; ++i) {
    std::cin >> Pi[i];
    initialPi[i] = Pi[i];
  }

  // Initialize adjacency list and read edges
  adjacency_list.resize(N + 1);
  edges.resize(M + 1);
  for (int i = 1; i <= M; ++i) {
    int u, v;
    std::cin >> u >> v;
    adjacency_list[u].emplace_back(v, i);
    adjacency_list[v].emplace_back(u, i);
    edges[i] = {u, v};
  }

  // Read K (number of initial services)
  std::cin >> K;
  services.resize(K + 1); // 1-based indexing

  // Initialize edge wavelengths
  edge_wavelengths.resize(M + 1, std::bitset<MAX_K + 1>());
  initial_edge_wavelengths.resize(M + 1, std::bitset<MAX_K + 1>());

  // Read details for each service
  for (int i = 1; i <= K; ++i) {
    Service &srv = services[i];
    srv.id = i;
    std::cin >> srv.s >> srv.d >> srv.S >> srv.L >> srv.R >> srv.V;
    srv.path.resize(srv.S);
    srv.wavelengths.resize(srv.S, srv.L); // Initialize wavelengths

    // Read the sequence of edge indices
    for (int j = 0; j < srv.S; ++j) {
      std::cin >> srv.path[j];
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
// Function to reset the environment to its initial state
void ResetEnvironment() {
  // Reset services to their initial state
  for (int i = 1; i <= K; ++i) {
    Service &srv = services[i];
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
  adjacency_list.assign(N + 1, std::vector<std::pair<int, int>>());
  for (int i = 1; i <= M; ++i) {
    int u = edges[i].first;
    int v = edges[i].second;
    adjacency_list[u].emplace_back(v, i);
    adjacency_list[v].emplace_back(u, i);
  }

  // Reset edge wavelengths
  edge_wavelengths = initial_edge_wavelengths;
}
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <queue>
#include <set>
#include <stdint.h>
#include <vector>

// Fast Max-Flow using Dinic's Algorithm
struct EdgeFlow {
  int to;
  int rev;
  int cap;
};

class MaxFlow {
public:
  int N;
  std::vector<std::vector<EdgeFlow>> graph;
  std::vector<int> level;
  std::vector<int> ptr;

  explicit MaxFlow(int N_)
      : N(N_), graph(N_ + 1), level(N_ + 1, -1), ptr(N_ + 1, 0) {}

  void AddEdge(int from, int to, int cap) {
    EdgeFlow a{to, static_cast<int>(graph[to].size()), cap};
    EdgeFlow b{from, static_cast<int>(graph[from].size()), 0};
    graph[from].emplace_back(a);
    graph[to].emplace_back(b);
  }

  bool BFS(int s, int t) {
    std::fill(level.begin(), level.end(), -1);
    std::queue<int> q;
    q.push(s);
    level[s] = 0;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (const auto &e : graph[u]) {
        if (e.cap > 0 && level[e.to] == -1) {
          level[e.to] = level[u] + 1;
          q.push(e.to);
          if (e.to == t)
            return true;
        }
      }
    }
    return false;
  }

  int DFS(int u, int t, int pushed) {
    if (u == t)
      return pushed;
    while (ptr[u] < static_cast<int>(graph[u].size())) {
      EdgeFlow &e = graph[u][ptr[u]];
      if (e.cap > 0 && level[e.to] == level[u] + 1) {
        int tr = DFS(e.to, t, std::min(pushed, e.cap));
        if (tr > 0) {
          graph[u][ptr[u]].cap -= tr;
          graph[e.to][e.rev].cap += tr;
          return tr;
        }
      }
      ptr[u]++;
    }
    return 0;
  }

  int MaxFlowValue(int s, int t) {
    int flow = 0;
    while (BFS(s, t)) {
      std::fill(ptr.begin(), ptr.end(), 0);
      while (int pushed = DFS(s, t, INT32_MAX)) {
        flow += pushed;
      }
    }
    return flow;
  }
};
// Function to compute the min-cut between two nodes using Dinic's algorithm
std::vector<int> ComputeMinCut(int s, int d) {
  MaxFlow mf(N);
  for (int i = 1; i <= M; ++i) {
    int u = edges[i].first;
    int v = edges[i].second;
    mf.AddEdge(u, v, 1);
    mf.AddEdge(v, u, 1);
  }
  mf.MaxFlowValue(s, d);

  // Find reachable nodes from s in the residual graph
  std::vector<bool> visited(N + 1, false);
  std::queue<int> q;
  q.push(s);
  visited[s] = true;
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (const auto &e : mf.graph[u]) {
      if (e.cap > 0 && !visited[e.to]) {
        visited[e.to] = true;
        q.push(e.to);
      }
    }
  }

  // Edges that go from visited to unvisited are in the min-cut
  std::vector<int> min_cut_edges;
  for (int i = 1; i <= M; ++i) {
    int u = edges[i].first;
    int v = edges[i].second;
    if (visited[u] && !visited[v]) {
      min_cut_edges.emplace_back(i);
    }
    if (visited[v] && !visited[u]) {
      min_cut_edges.emplace_back(i);
    }
  }
  return min_cut_edges;
}

// Function to compute Jaccard similarity between two sets
double JaccardSimilarity(const std::set<int> &a, const std::set<int> &b) {
  if (a.empty() && b.empty())
    return 0.0;
  int intersection_size = 0;
  for (const auto &elem : a) {
    if (b.find(elem) != b.end()) {
      ++intersection_size;
    }
  }
  int union_size = static_cast<int>(a.size()) + static_cast<int>(b.size()) -
                   intersection_size;
  return static_cast<double>(intersection_size) / union_size;
}

// Function to generate edge failure scenarios using min-cut
void GenerateFailureScenarios() {
  T1 = std::min(0, K);

  failure_scenarios.clear();

  // Sort services by value in descending order
  std::vector<int> service_indices(K);
  for (int i = 0; i < K; ++i) {
    service_indices[i] = i + 1;
  }

  std::sort(service_indices.begin(), service_indices.end(),
            [&](int a, int b) { return services[a].V > services[b].V; });

  std::set<std::set<int>> scenario_edge_sets;

  for (const auto &idx : service_indices) {
    if (static_cast<int>(failure_scenarios.size()) >= T1) {
      break;
    }

    const Service &srv = services[idx];
    int s = srv.s;
    int d = srv.d;

    std::vector<int> min_cut_edges = ComputeMinCut(s, d);

    if (min_cut_edges.empty())
      continue;

    // Limit the number of edge failures per scenario
    if (static_cast<int>(min_cut_edges.size()) > 60) {
      min_cut_edges.resize(60);
    }

    std::set<int> current_edges(min_cut_edges.begin(), min_cut_edges.end());

    // Check for duplicates or similar scenarios
    bool similar = false;
    for (const auto &edge_set : scenario_edge_sets) {
      double similarity = JaccardSimilarity(edge_set, current_edges);
      if (similarity > 0.5) {
        similar = true;
        break;
      }
    }
    if (similar)
      continue;

    FailureScenario fs;
    fs.ci = static_cast<int>(min_cut_edges.size());
    fs.edge_ids = min_cut_edges;

    failure_scenarios.emplace_back(fs);
    scenario_edge_sets.emplace(current_edges);
  }

  T1 = static_cast<int>(failure_scenarios.size());
}

void OutputEdgeFailureScenarios() {
  // Output T1, the number of edge failure test scenarios we provide
  std::cout << T1 << "\n";

  // Output each scenario as per the required format
  for (const auto &fs : failure_scenarios) {
    std::cout << fs.ci << "\n";
    for (std::size_t i = 0; i < fs.edge_ids.size(); ++i) {
      std::cout << fs.edge_ids[i] << (i < fs.edge_ids.size() - 1 ? " " : "\n");
    }
  }
  std::cout.flush();
}
#include <bitset>
#include <map>
#include <set>
#include <vector>

// Function declarations
void InitializeEnvironment();
void OutputEdgeFailureScenarios();
void GenerateFailureScenarios();
void HandleTestScenarios();
void ResetEnvironment();
void HandleEdgeFailure(int e);
std::vector<int> DetectAffectedServices(int e);
std::vector<int>
AssignWavelengthsUsingGA(const std::vector<int> &affected_services);
bool RunGAWithTemp(Service &srv,
                   std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
                   std::vector<int> &temp_Pi);
double FitnessFunction(
    const std::vector<int> &chromosome, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi);
std::vector<int> GenerateInitialChromosome(const Service &srv);
std::vector<int> MutateChromosome(const std::vector<int> &chromosome,
                                  const Service &srv);
std::pair<std::vector<int>, std::vector<int>>
CrossoverChromosomes(const std::vector<int> &parent1,
                     const std::vector<int> &parent2, const Service &srv);
bool DecodeChromosomeWithTemp(
    const std::vector<int> &chromosome, const Service &srv,
    std::vector<int> &path, std::vector<int> &wavelengths,
    std::map<int, int> &converters_needed,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi);
std::vector<std::pair<int, int>> GetAvailableIntervalsWithTemp(
    int edge_id, int width, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths);
double JaccardSimilarity(const std::set<int> &a, const std::set<int> &b);
std::vector<int> ComputeMinCut(int s, int d);
#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <map>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

// Random number generator setup
std::random_device rd;
std::mt19937 gen(rd());

// Function to assign wavelengths using GA with temporary edge_wavelengths
std::vector<int>
AssignWavelengthsUsingGA(const std::vector<int> &affected_services) {
  std::vector<int> successfully_replanned_services;

  // Create a temporary copy of edge_wavelengths to avoid conflicts
  std::vector<std::bitset<MAX_K + 1>> temp_edge_wavelengths = edge_wavelengths;

  // Create a temporary copy of Pi
  std::vector<int> temp_Pi = Pi;

  // Sort services by value (higher value first)
  std::vector<int> sorted_services = affected_services;
  std::sort(sorted_services.begin(), sorted_services.end(),
            [&](int a, int b) { return services[a].V > services[b].V; });

  // Run GA for each affected service
  for (const auto service_id : sorted_services) {
    Service &srv = services[service_id];
    // Run GA to find the best path and wavelength assignment using
    // temp_edge_wavelengths and temp_Pi
    bool replanned = RunGAWithTemp(srv, temp_edge_wavelengths, temp_Pi);
    if (replanned) {
      successfully_replanned_services.emplace_back(service_id);
      // Update temp_edge_wavelengths with the new assignments
      for (std::size_t j = 0; j < srv.path.size(); ++j) {
        int edge_id = srv.path[j];
        for (int w = srv.wavelengths[j];
             w <= srv.wavelengths[j] + (srv.R - srv.L); ++w) {
          temp_edge_wavelengths[edge_id].set(w);
        }
      }
      // Channel conversion opportunities updated in RunGAWithTemp
    } else {
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

// Function to run GA for a single service using temporary edge_wavelengths and temp_Pi
bool RunGAWithTemp(Service &srv,
                   std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
                   std::vector<int> &temp_Pi) {
  // Initialize population
  std::vector<std::vector<int>> population;
  std::vector<double> fitness_values;

  // Generate initial population
  for (int i = 0; i < POPULATION_SIZE; ++i) {
    std::vector<int> chromosome = GenerateInitialChromosome(srv);
    if (chromosome.empty())
      continue;
    population.emplace_back(chromosome);
    fitness_values.emplace_back(
        FitnessFunction(chromosome, srv, temp_edge_wavelengths, temp_Pi));
  }

  if (population.empty())
    return false;

  // Evolutionary loop
  for (int generation = 0; generation < MAX_GENERATIONS; ++generation) {
    // Selection (Tournament selection)
    std::vector<std::vector<int>> new_population;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> dist_idx(
        0, static_cast<int>(population.size()) - 1);

    while (static_cast<int>(new_population.size()) < POPULATION_SIZE) {
      // Tournament selection for parent1
      int idx1 = dist_idx(gen);
      int idx2 = dist_idx(gen);
      const std::vector<int> &parent1 =
          (fitness_values[idx1] < fitness_values[idx2]) ? population[idx1]
                                                        : population[idx2];

      // Tournament selection for parent2
      idx1 = dist_idx(gen);
      idx2 = dist_idx(gen);
      const std::vector<int> &parent2 =
          (fitness_values[idx1] < fitness_values[idx2]) ? population[idx1]
                                                        : population[idx2];

      // Crossover
      std::vector<int> child1, child2;
      std::tie(child1, child2) = CrossoverChromosomes(parent1, parent2, srv);

      // Mutation
      std::uniform_real_distribution<> mutation_dis(0.0, 1.0);
      if (mutation_dis(gen) < MUTATION_RATE) {
        child1 = MutateChromosome(child1, srv);
      }
      if (mutation_dis(gen) < MUTATION_RATE) {
        child2 = MutateChromosome(child2, srv);
      }

      // Add to new population
      if (!child1.empty())
        new_population.emplace_back(child1);
      if (!child2.empty())
        new_population.emplace_back(child2);
    }

    // Evaluate new population
    population = std::move(new_population);
    fitness_values.clear();
    fitness_values.reserve(population.size());
    for (const auto &chrom : population) {
      fitness_values.emplace_back(
          FitnessFunction(chrom, srv, temp_edge_wavelengths, temp_Pi));
    }
  }

  // Get the best chromosome
  auto best_iter =
      std::min_element(fitness_values.begin(), fitness_values.end());
  int best_idx =
      static_cast<int>(std::distance(fitness_values.begin(), best_iter));
  const std::vector<int> &best_chromosome = population[best_idx];

  // Decode the best chromosome
  std::vector<int> path;
  std::vector<int> wavelengths;
  std::map<int, int> converters_needed;
  bool valid = DecodeChromosomeWithTemp(best_chromosome, srv, path, wavelengths,
                                        converters_needed,
                                        temp_edge_wavelengths, temp_Pi);
  if (valid) {
    srv.path = path;
    srv.wavelengths = wavelengths;
    // Update temp_Pi with converters_used
    for (const auto &[node, count] : converters_needed) {
      temp_Pi[node] -= count;
    }
    return true;
  }
  return false;
}

// Fitness function for the GA with temporary edge_wavelengths and temp_Pi
double FitnessFunction(
    const std::vector<int> &chromosome, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi) {
  // Decode chromosome to get path and wavelength assignment
  std::vector<int> path;
  std::vector<int> wavelengths;
  std::map<int, int> converters_needed;
  bool valid = DecodeChromosomeWithTemp(chromosome, srv, path, wavelengths,
                                        converters_needed,
                                        temp_edge_wavelengths, temp_Pi);
  if (!valid)
    return 1e9; // High penalty for invalid chromosome

  // Fitness considers path length and number of wavelength changes (channel
  // conversions)
  int num_conversions = 0;
  for (const auto &[node, count] : converters_needed) {
    num_conversions += count;
  }
  return static_cast<double>(path.size()) + num_conversions * 10.0;
}

// Function to generate an initial chromosome for a service
std::vector<int> GenerateInitialChromosome(const Service &srv) {
  // Generate a random simple path from s to d
  int max_attempts = 100;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    std::vector<int> path_nodes;
    std::unordered_set<int> visited;
    std::vector<int> edge_path;

    int current_node = srv.s;
    path_nodes.push_back(current_node);
    visited.insert(current_node);

    while (current_node != srv.d) {
      const auto &neighbors = adjacency_list[current_node];
      if (neighbors.empty()) {
        break; // Dead end
      }
      // Randomly select a neighbor
      std::vector<std::pair<int, int>> unvisited_neighbors;
      for (const auto &[neighbor, edge_id] : neighbors) {
        if (visited.find(neighbor) == visited.end()) {
          unvisited_neighbors.emplace_back(neighbor, edge_id);
        }
      }
      if (unvisited_neighbors.empty()) {
        break; // No unvisited neighbors, dead end
      }
      std::uniform_int_distribution<> dist(
          0, static_cast<int>(unvisited_neighbors.size()) - 1);
      int idx = dist(gen);
      int next_node = unvisited_neighbors[idx].first;
      int edge_id = unvisited_neighbors[idx].second;

      path_nodes.push_back(next_node);
      edge_path.push_back(edge_id);
      visited.insert(next_node);
      current_node = next_node;
    }

    if (current_node == srv.d) {
      return edge_path;
    }
    // If not successful, try again
  }
  // If no path found after max_attempts
  return {};
}

// Function to get nodes from edges
std::vector<int> GetNodesFromEdges(const std::vector<int> &edges_sequence, int start_node) {
  std::vector<int> nodes;
  nodes.push_back(start_node);
  int current_node = start_node;

  for (const auto &edge_id : edges_sequence) {
    int u = edges[edge_id].first;
    int v = edges[edge_id].second;
    int next_node;
    if (u == current_node) {
      next_node = v;
    } else if (v == current_node) {
      next_node = u;
    } else {
      // Invalid path
      return {};
    }
    nodes.push_back(next_node);
    current_node = next_node;
  }
  return nodes;
}

// Function to convert nodes to edges
std::vector<int> ConvertNodesToEdges(const std::vector<int> &nodes) {
  std::vector<int> edges_path;
  for (size_t i = 1; i < nodes.size(); ++i) {
    int u = nodes[i - 1];
    int v = nodes[i];
    // Find an edge between u and v
    bool found = false;
    for (const auto &[neighbor, edge_id] : adjacency_list[u]) {
      if (neighbor == v) {
        edges_path.emplace_back(edge_id);
        found = true;
        break;
      }
    }
    if (!found) {
      // No edge between u and v
      return {};
    }
  }
  return edges_path;
}

// Function to perform crossover between two chromosomes
std::pair<std::vector<int>, std::vector<int>>
CrossoverChromosomes(const std::vector<int> &parent1,
                     const std::vector<int> &parent2, const Service &srv) {
  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) >= CROSSOVER_RATE) {
    // No crossover, return parents
    return {parent1, parent2};
  }

  // Find common nodes between the two parents
  std::vector<int> parent1_nodes = GetNodesFromEdges(parent1, srv.s);
  std::vector<int> parent2_nodes = GetNodesFromEdges(parent2, srv.s);

  if (parent1_nodes.empty() || parent2_nodes.empty()) {
    return {parent1, parent2};
  }

  std::unordered_map<int, int> node_indices_p1;
  for (size_t i = 0; i < parent1_nodes.size(); ++i) {
    node_indices_p1[parent1_nodes[i]] = static_cast<int>(i);
  }

  std::vector<int> common_nodes;
  for (size_t i = 0; i < parent2_nodes.size(); ++i) {
    int node = parent2_nodes[i];
    if (node_indices_p1.find(node) != node_indices_p1.end()) {
      common_nodes.push_back(node);
    }
  }

  if (common_nodes.empty()) {
    // No common nodes, cannot crossover
    return {parent1, parent2};
  }

  // Randomly select a common node to crossover
  std::uniform_int_distribution<> dist(0, static_cast<int>(common_nodes.size()) - 1);
  int idx = dist(gen);
  int crossover_node = common_nodes[idx];

  // Get indices of crossover node in parents
  int p1_idx = node_indices_p1[crossover_node];
  int p2_idx = 0;
  for (size_t i = 0; i < parent2_nodes.size(); ++i) {
    if (parent2_nodes[i] == crossover_node) {
      p2_idx = static_cast<int>(i);
      break;
    }
  }

  // Create offspring by swapping subpaths
  std::vector<int> child1_nodes(parent1_nodes.begin(), parent1_nodes.begin() + p1_idx + 1);
  child1_nodes.insert(child1_nodes.end(), parent2_nodes.begin() + p2_idx + 1, parent2_nodes.end());

  std::vector<int> child2_nodes(parent2_nodes.begin(), parent2_nodes.begin() + p2_idx + 1);
  child2_nodes.insert(child2_nodes.end(), parent1_nodes.begin() + p1_idx + 1, parent1_nodes.end());

  // Remove possible cycles in child paths
  std::unordered_set<int> visited_nodes;
  std::vector<int> pruned_child1_nodes;
  for (int node : child1_nodes) {
    if (visited_nodes.count(node) == 0) {
      pruned_child1_nodes.push_back(node);
      visited_nodes.insert(node);
    } else {
      // Cycle detected, stop adding nodes
      break;
    }
  }

  visited_nodes.clear();
  std::vector<int> pruned_child2_nodes;
  for (int node : child2_nodes) {
    if (visited_nodes.count(node) == 0) {
      pruned_child2_nodes.push_back(node);
      visited_nodes.insert(node);
    } else {
      // Cycle detected, stop adding nodes
      break;
    }
  }

  // Convert node paths to edge paths
  std::vector<int> child1_edges = ConvertNodesToEdges(pruned_child1_nodes);
  std::vector<int> child2_edges = ConvertNodesToEdges(pruned_child2_nodes);

  // Validate paths
  if (child1_edges.empty()) {
    child1_edges = parent1;
  }
  if (child2_edges.empty()) {
    child2_edges = parent2;
  }

  return {child1_edges, child2_edges};
}

// Function to generate a random path for mutation
std::vector<int> GenerateRandomPath(const Service &srv) {
  // Similar to GenerateInitialChromosome, but from srv.s to srv.d
  int max_attempts = 100;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    std::vector<int> path_nodes;
    std::unordered_set<int> visited;
    std::vector<int> edge_path;

    int current_node = srv.s;
    path_nodes.push_back(current_node);
    visited.insert(current_node);

    while (current_node != srv.d) {
      const auto &neighbors = adjacency_list[current_node];
      if (neighbors.empty()) {
        break; // Dead end
      }
      // Randomly select a neighbor
      std::vector<std::pair<int, int>> unvisited_neighbors;
      for (const auto &[neighbor, edge_id] : neighbors) {
        if (visited.find(neighbor) == visited.end()) {
          unvisited_neighbors.emplace_back(neighbor, edge_id);
        }
      }
      if (unvisited_neighbors.empty()) {
        break; // No unvisited neighbors, dead end
      }
      std::uniform_int_distribution<> dist(0, static_cast<int>(unvisited_neighbors.size()) - 1);
      int idx = dist(gen);
      int next_node = unvisited_neighbors[idx].first;
      int edge_id = unvisited_neighbors[idx].second;

      path_nodes.push_back(next_node);
      edge_path.push_back(edge_id);
      visited.insert(next_node);
      current_node = next_node;
    }

    if (current_node == srv.d) {
      return edge_path;
    }
    // If not successful, try again
  }
  // If no path found after max_attempts
  return {};
}

// Function to mutate a chromosome
std::vector<int> MutateChromosome(const std::vector<int> &chromosome,
                                  const Service &srv) {
  // Convert chromosome to nodes
  std::vector<int> nodes = GetNodesFromEdges(chromosome, srv.s);

  if (nodes.size() <= 2) {
    // Can't mutate a path of length <= 2
    return chromosome;
  }

  // Randomly select a mutation point
  std::uniform_int_distribution<> dist_idx(1, static_cast<int>(nodes.size()) - 2);
  int mutation_point = dist_idx(gen);

  int mutation_node = nodes[mutation_point];

  // Generate a new subpath from mutation_node to destination
  Service sub_srv;
  sub_srv.s = mutation_node;
  sub_srv.d = srv.d;

  std::vector<int> new_subpath = GenerateRandomPath(sub_srv);
  if (new_subpath.empty()) {
    // Mutation failed, return original chromosome
    return chromosome;
  }

  // Convert new subpath to node sequence
  std::vector<int> new_subpath_nodes = GetNodesFromEdges(new_subpath, mutation_node);

  // Build new nodes sequence
  std::vector<int> mutated_nodes;
  mutated_nodes.insert(mutated_nodes.end(), nodes.begin(), nodes.begin() + mutation_point + 1);
  mutated_nodes.insert(mutated_nodes.end(), new_subpath_nodes.begin() + 1, new_subpath_nodes.end());

  // Remove possible cycles
  std::unordered_set<int> visited_nodes;
  std::vector<int> pruned_mutated_nodes;
  for (int node : mutated_nodes) {
    if (visited_nodes.count(node) == 0) {
      pruned_mutated_nodes.push_back(node);
      visited_nodes.insert(node);
    } else {
      // Cycle detected, stop adding nodes
      break;
    }
  }

  // Convert nodes back to edges
  std::vector<int> mutated_edges = ConvertNodesToEdges(pruned_mutated_nodes);
  if (mutated_edges.empty()) {
    // Invalid path, return original chromosome
    return chromosome;
  }
  return mutated_edges;
}

// Function to decode a chromosome into a path and wavelength assignment using
// temporary edge_wavelengths and temp_Pi
bool DecodeChromosomeWithTemp(
    const std::vector<int> &chromosome, const Service &srv,
    std::vector<int> &path, std::vector<int> &wavelengths,
    std::map<int, int> &converters_needed,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi) {
  path.clear();
  wavelengths.clear();
  converters_needed.clear();

  // Convert chromosome into a path
  int current_node = srv.s;
  for (const auto edge_id : chromosome) {
    int u = edges[edge_id].first;
    int v = edges[edge_id].second;
    if (u == current_node) {
      path.emplace_back(edge_id);
      current_node = v;
    } else if (v == current_node) {
      path.emplace_back(edge_id);
      current_node = u;
    } else {
      // Invalid path
      return false;
    }
  }
  if (current_node != srv.d)
    return false;

  // Assign wavelengths
  int width = srv.R - srv.L + 1;
  int prev_wavelength = -1;
  for (std::size_t i = 0; i < path.size(); ++i) {
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
    wavelengths.emplace_back(assigned_wavelength);
    prev_wavelength = assigned_wavelength;
  }

  // Collect the number of converters needed at each node
  prev_wavelength = wavelengths.empty() ? -1 : wavelengths[0];
  for (std::size_t i = 1; i < wavelengths.size(); ++i) {
    if (wavelengths[i] != wavelengths[i - 1]) {
      // Channel conversion needed at node between edge i-1 and i
      int edge_prev = path[i - 1];
      int edge_curr = path[i];
      // Find the common node
      int u1 = edges[edge_prev].first;
      int v1 = edges[edge_prev].second;
      int u2 = edges[edge_curr].first;
      int v2 = edges[edge_curr].second;

      int common_node = -1;
      if (u1 == u2 || u1 == v2) {
        common_node = u1;
      } else if (v1 == u2 || v1 == v2) {
        common_node = v1;
      } else {
        return false; // Invalid path
      }

      if (common_node == srv.s) {
        // Source node cannot perform channel conversion
        return false;
      }

      converters_needed[common_node]++;
      prev_wavelength = wavelengths[i];
    }
  }

  // Now check if we have enough channel conversion opportunities at each node
  for (const auto &[node, required] : converters_needed) {
    if (temp_Pi[node] < required) {
      return false; // Not enough channel conversion opportunities
    }
  }

  return true;
}
#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <utility>
#include <vector>

// Main function
int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

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
// Function to handle an edge failure in the network
void HandleEdgeFailure(int e) {
  // Remove the failed edge from the adjacency list
  int u = edges[e].first;
  int v = edges[e].second;

  // Remove edge e from u's adjacency list
  auto &adj_u = adjacency_list[u];
  adj_u.erase(std::remove_if(adj_u.begin(), adj_u.end(),
                             [&](const std::pair<int, int> &p) {
                               return p.first == v && p.second == e;
                             }),
              adj_u.end());

  // Remove edge e from v's adjacency list
  auto &adj_v = adjacency_list[v];
  adj_v.erase(std::remove_if(adj_v.begin(), adj_v.end(),
                             [&](const std::pair<int, int> &p) {
                               return p.first == u && p.second == e;
                             }),
              adj_v.end());
}

// Function to detect services affected by an edge failure
std::vector<int> DetectAffectedServices(int e) {
  std::vector<int> affected_services;
  for (int i = 1; i <= K; ++i) {
    if (!services[i].alive)
      continue;
    if (std::find(services[i].path.begin(), services[i].path.end(), e) !=
        services[i].path.end()) {
      services[i].need_to_replan = true;
      affected_services.emplace_back(i);
    }
  }
  return affected_services;
}
// Function to get available intervals of wavelengths of size 'width' on an edge
// using temporary edge_wavelengths
std::vector<std::pair<int, int>> GetAvailableIntervalsWithTemp(
    int edge_id, int width, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths) {
  std::vector<std::pair<int, int>> intervals;
  // Find continuous available wavelengths
  int count = 0;
  int start = 1;
  for (int w = 1; w <= MAX_K; ++w) {
    if (!temp_edge_wavelengths[edge_id].test(w)) {
      if (count == 0) {
        start = w;
      }
      count++;
      if (count >= width) {
        intervals.emplace_back(start, w);
        // Move to the next possible interval
        count--;
        start++;
      }
    } else {
      count = 0;
    }
  }
  return intervals;
}

// Function to handle test scenarios provided by the system
void HandleTestScenarios() {
  while (true) {
    int T;
    if (!(std::cin >> T)) {
      break; // No more input
    }
    for (int t = 0; t < T; ++t) {
      // Reset the environment to the initial state before each scenario
      ResetEnvironment();

      while (true) {
        int e;
        std::cin >> e;
        if (e == -1) {
          break; // End of the test scenario
        }
        // Handle the edge failure in the network
        HandleEdgeFailure(e);

        // Detect services affected by this edge failure
        std::vector<int> affected_services = DetectAffectedServices(e);

        // Replan services affected by the current edge failure
        std::vector<int> successfully_replanned_services =
            AssignWavelengthsUsingGA(affected_services);

        // Output the number of services successfully replanned
        int K_prime = static_cast<int>(successfully_replanned_services.size());
        std::cout << K_prime << "\n";

        // Output details of each successfully replanned service
        for (const auto &service_id : successfully_replanned_services) {
          const Service &srv = services[service_id];
          std::cout << service_id << " " << srv.path.size() << "\n";
          // Output path and wavelength intervals
          for (std::size_t j = 0; j < srv.path.size(); ++j) {
            std::cout << srv.path[j] << " " << srv.wavelengths[j] << " "
                      << srv.wavelengths[j] + (srv.R - srv.L)
                      << (j < srv.path.size() - 1 ? " " : "\n");
          }
        }
        std::cout.flush();
      }
    }
  }
}
