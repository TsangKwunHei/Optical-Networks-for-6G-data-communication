#include "environment.cpp"
#include "failures.cpp"
#include "ga.cpp"
#include "m.hpp"
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
