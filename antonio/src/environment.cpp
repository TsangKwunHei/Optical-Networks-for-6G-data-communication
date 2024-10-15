#include "globals.cpp"
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
