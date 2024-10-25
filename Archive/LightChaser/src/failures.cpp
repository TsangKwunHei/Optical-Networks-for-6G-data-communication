#include "globals.cpp"
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
