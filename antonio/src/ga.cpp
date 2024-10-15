#include "globals.cpp"
#include "m.hpp"
#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <map>
#include <queue>
#include <random>
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

// Function to run GA for a single service using temporary edge_wavelengths and
// temp_Pi
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

// Function to generate an initial chromosome for a service using BFS for a
// single path
std::vector<int> GenerateInitialChromosome(const Service &srv) {
  // BFS to find the shortest path
  std::queue<std::vector<int>> q;
  std::vector<bool> visited(N + 1, false);
  q.push({srv.s});
  visited[srv.s] = true;

  while (!q.empty()) {
    std::vector<int> path_nodes = q.front();
    q.pop();
    int current_node = path_nodes.back();

    if (current_node == srv.d) {
      // Convert node path to edge path
      std::vector<int> edge_path;
      for (std::size_t i = 1; i < path_nodes.size(); ++i) {
        int u = path_nodes[i - 1];
        int v = path_nodes[i];
        // Find the edge ID
        bool found = false;
        for (const auto &[neighbor, edge_id] : adjacency_list[u]) {
          if (neighbor == v) {
            edge_path.emplace_back(edge_id);
            found = true;
            break;
          }
        }
        if (!found) {
          edge_path.clear();
          break;
        }
      }
      return edge_path;
    }

    for (const auto &[neighbor, edge_id] : adjacency_list[current_node]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        std::vector<int> new_path = path_nodes;
        new_path.emplace_back(neighbor);
        q.push(new_path);
      }
    }
  }
  // If no path found
  return {};
}

// Function to mutate a chromosome
std::vector<int> MutateChromosome(const std::vector<int> &chromosome,
                                  const Service &srv) {
  std::vector<int> mutated = chromosome;
  if (mutated.empty())
    return mutated;

  std::uniform_int_distribution<> dist_idx(0, static_cast<int>(mutated.size()) -
                                                  1);
  int idx = dist_idx(gen);
  int current_edge = mutated[idx];
  int u = edges[current_edge].first;
  int v = edges[current_edge].second;

  // Determine the current node
  int current_node = srv.s;
  for (int i = 0; i < idx; ++i) {
    int edge_id = mutated[i];
    if (edges[edge_id].first == current_node) {
      current_node = edges[edge_id].second;
    } else {
      current_node = edges[edge_id].first;
    }
  }

  // Get all possible edges from current_node
  const auto &possible_edges = adjacency_list[current_node];
  if (possible_edges.empty())
    return mutated;

  // Replace with a random adjacent edge
  std::uniform_int_distribution<> dist_edge(
      0, static_cast<int>(possible_edges.size()) - 1);
  int new_edge = possible_edges[dist_edge(gen)].second;
  mutated[idx] = new_edge;
  return mutated;
}

// Function to perform crossover between two chromosomes
std::pair<std::vector<int>, std::vector<int>>
CrossoverChromosomes(const std::vector<int> &parent1,
                     const std::vector<int> &parent2, const Service &srv) {
  std::vector<int> child1 = parent1;
  std::vector<int> child2 = parent2;
  if (parent1.empty() || parent2.empty())
    return {child1, child2};

  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) < CROSSOVER_RATE) {
    std::uniform_int_distribution<> dist_point(
        0, static_cast<int>(std::min(parent1.size(), parent2.size())) - 1);
    int crossover_point = dist_point(gen);
    // Perform single-point crossover
    child1.assign(parent1.begin(), parent1.begin() + crossover_point);
    child1.insert(child1.end(), parent2.begin() + crossover_point,
                  parent2.end());

    child2.assign(parent2.begin(), parent2.begin() + crossover_point);
    child2.insert(child2.end(), parent1.begin() + crossover_point,
                  parent1.end());
  }
  return {child1, child2};
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
    if (current_node == srv.d)
      break;
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
      int common_node = -1;
      if (edges[edge_prev].first == edges[edge_curr].first ||
          edges[edge_prev].first == edges[edge_curr].second) {
        common_node = edges[edge_prev].first;
      } else {
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
  for (const auto &[node, required] : converters_needed) {
    if (temp_Pi[node] < required) {
      return false; // Not enough channel conversion opportunities
    }
  }

  return true;
}
