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
#include <set>
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
        int w = srv.wavelengths[j];
        int width = srv.R - srv.L + 1;
        for (int offset = 0; offset < width; ++offset) {
          temp_edge_wavelengths[edge_id].set(w + offset);
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

  // Generate initial population using random walks
  int initial_population_size = 0;
  while (initial_population_size < POPULATION_SIZE * 2) {
    std::vector<int> chromosome = GenerateInitialChromosome(srv);
    if (chromosome.empty())
      continue;
    population.emplace_back(chromosome);
    fitness_values.emplace_back(
        FitnessFunction(chromosome, srv, temp_edge_wavelengths, temp_Pi));
    initial_population_size++;
    if (static_cast<int>(population.size()) >= POPULATION_SIZE)
      break;
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
                                        converters_needed, temp_edge_wavelengths,
                                        temp_Pi);
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
                                        converters_needed, temp_edge_wavelengths,
                                        temp_Pi);
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
// Function to generate an initial chromosome for a service using BFS
std::vector<int> GenerateInitialChromosome(const Service &srv) {
  std::vector<int> parent(N + 1, -1);
  std::vector<int> edge_to(N + 1, -1);
  std::queue<int> q;
  std::vector<bool> visited(N + 1, false);

  q.push(srv.s);
  visited[srv.s] = true;

  while (!q.empty()) {
    int current_node = q.front();
    q.pop();

    if (current_node == srv.d)
      break;

    for (const auto &[neighbor, edge_id] : adjacency_list[current_node]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        parent[neighbor] = current_node;
        edge_to[neighbor] = edge_id;
        q.push(neighbor);
      }
    }
  }

  if (!visited[srv.d]) {
    // No path found
    return {};
  }

  // Reconstruct path from s to d
  std::vector<int> path_edges;
  int current_node = srv.d;
  while (current_node != srv.s) {
    path_edges.push_back(edge_to[current_node]);
    current_node = parent[current_node];
  }

  std::reverse(path_edges.begin(), path_edges.end());
  return path_edges;
}

// Function to mutate a chromosome while ensuring a valid path
std::vector<int> MutateChromosome(const std::vector<int> &chromosome,
                                  const Service &srv) {
  if (chromosome.empty())
    return chromosome;

  // Select a mutation point
  std::uniform_int_distribution<> dist_idx(0, static_cast<int>(chromosome.size()) - 1);
  int mutation_point = dist_idx(gen);

  // Extract sub-paths
  std::vector<int> prefix_edges(chromosome.begin(), chromosome.begin() + mutation_point);
  int mutation_node = srv.s;
  for (int edge_id : prefix_edges) {
    int u = edges[edge_id].first;
    int v = edges[edge_id].second;
    mutation_node = (u == mutation_node) ? v : u;
  }

  // Reroute from mutation_node to destination using BFS
  Service temp_srv = srv;
  temp_srv.s = mutation_node;
  std::vector<int> new_suffix = GenerateInitialChromosome(temp_srv);

  if (new_suffix.empty())
    return chromosome; // Return original if mutation fails

  // Combine prefix and new suffix
  prefix_edges.insert(prefix_edges.end(), new_suffix.begin(), new_suffix.end());
  return prefix_edges;
}

// Function to perform crossover between two chromosomes ensuring valid paths
std::pair<std::vector<int>, std::vector<int>>
CrossoverChromosomes(const std::vector<int> &parent1,
                     const std::vector<int> &parent2, const Service &srv) {
  if (parent1.empty() || parent2.empty())
    return {parent1, parent2};

  // Map nodes to indices for quick lookup
  std::unordered_map<int, int> parent1_nodes;
  int node = srv.s;
  parent1_nodes[node] = 0;
  for (size_t i = 0; i < parent1.size(); ++i) {
    int edge_id = parent1[i];
    int u = edges[edge_id].first;
    int v = edges[edge_id].second;
    node = (u == node) ? v : u;
    parent1_nodes[node] = i + 1;
  }

  node = srv.s;
  std::vector<int> common_nodes;
  if (parent1_nodes.count(node))
    common_nodes.push_back(node);
  for (size_t i = 0; i < parent2.size(); ++i) {
    int edge_id = parent2[i];
    int u = edges[edge_id].first;
    int v = edges[edge_id].second;
    node = (u == node) ? v : u;
    if (parent1_nodes.count(node))
      common_nodes.push_back(node);
  }

  if (common_nodes.size() <= 1)
    return {parent1, parent2}; // No crossover possible

  // Choose a random crossover node
  std::uniform_int_distribution<> dist(1, static_cast<int>(common_nodes.size()) - 1);
  int crossover_node = common_nodes[dist(gen)];

  // Split parents at crossover node
  int idx1 = parent1_nodes[crossover_node];
  int idx2 = parent1_nodes[crossover_node];

  std::vector<int> child1_edges(parent1.begin(), parent1.begin() + idx1);
  std::vector<int> child2_edges(parent2.begin(), parent2.begin() + idx2);

  // Append the rest from the other parent
  child1_edges.insert(child1_edges.end(), parent2.begin() + idx2, parent2.end());
  child2_edges.insert(child2_edges.end(), parent1.begin() + idx1, parent1.end());

  return {child1_edges, child2_edges};
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
      int node_u1 = edges[edge_prev].first;
      int node_v1 = edges[edge_prev].second;
      int node_u2 = edges[edge_curr].first;
      int node_v2 = edges[edge_curr].second;
      int common_node = -1;
      if (node_u1 == node_u2 || node_u1 == node_v2) {
        common_node = node_u1;
      } else if (node_v1 == node_u2 || node_v1 == node_v2) {
        common_node = node_v1;
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
