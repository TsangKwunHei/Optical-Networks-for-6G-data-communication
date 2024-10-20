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
