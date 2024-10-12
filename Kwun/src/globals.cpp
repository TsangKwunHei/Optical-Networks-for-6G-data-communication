#include "structs.h"
#include <bitset>
#include <vector>

#ifndef _GLOBALS_H_
#define _GLOBALS_H_
// Constants
constexpr int MAX_K = 40;
constexpr int MAX_N = 200;
constexpr int MAX_M = 1000;

// GA parameters
constexpr int POPULATION_SIZE = 100;
constexpr int MAX_GENERATIONS = 100;
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
