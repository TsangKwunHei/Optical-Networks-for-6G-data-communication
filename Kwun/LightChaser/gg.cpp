#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <random>
#include <ctime>
#include <cstdio>  // For fflush

using namespace std;

struct Service {
    int id;
    int source;
    int destination;
    int traversed_number;
    int left_channel;
    int right_channel;
    int service_value;
    vector<int> path;           // Node IDs
    vector<int> passed_edges;   // Edge IDs
    vector<int> left_channel_per_edge;
    vector<int> right_channel_per_edge;
    vector<bool> hold_resource_edges;
    bool alive = true;
    bool need_to_replan = false;
};

struct Node {
    int id;
    int switching_times = 0;
};

struct Node_for_bfs {
    int id;
    vector<int> edgeIndices;
    vector<int> nodeIndices;
    float cost = 0.0;
    vector<int> left_channel_list;
    vector<int> right_channel_list;
    vector<bool> using_switch_list;
    int left_channel;
    int right_channel;
    Node_for_bfs(int _id, vector<int> _edgeIndices)
        : id(_id), edgeIndices(_edgeIndices){};
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices)
        : id(_id), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices) {}
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices, int _left_channel,
                 int _right_channel, float _cost)
        : id(_id),
          edgeIndices(_edgeIndices),
          nodeIndices(_nodeIndices),
          left_channel(_left_channel),
          right_channel(_right_channel),
          cost(_cost) {}
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices, float _cost,
                 vector<int> _left_channel_list, vector<int> _right_channel_list,
                 vector<bool> _using_switch_list, int _left_channel, int _right_channel)
        : id(_id),
          edgeIndices(_edgeIndices),
          nodeIndices(_nodeIndices),
          cost(_cost),
          left_channel_list(_left_channel_list),
          right_channel_list(_right_channel_list),
          using_switch_list(_using_switch_list),
          left_channel(_left_channel),
          right_channel(_right_channel) {}
};

struct Edge {
    int connecting_node[2];
    bool is_broken = false;
    float importance = 0;
    vector<bool> wave_length_occupied;
    Edge() {
        wave_length_occupied.resize(40);
        for (int i = 0; i < 40; i++) {
            wave_length_occupied[i] = false;
        }
    }
};

// Environment Data
int node_number;
int edge_number;
int service_number;
vector<Node> init_nodes;
vector<Edge> init_edges;
vector<Service> init_services;

int minWavelength = 40;  // Used in costCalc function

// Calculate edge importance using BFS
void calcEdgeImportance() {
    multimap<int, int> graph;
    map<pair<int, int>, int> edgeIndexMap;
    for (int i = 0; i < edge_number; i++) {
        graph.insert({init_edges[i].connecting_node[0], init_edges[i].connecting_node[1]});
        graph.insert({init_edges[i].connecting_node[1], init_edges[i].connecting_node[0]});
        edgeIndexMap[{init_edges[i].connecting_node[0], init_edges[i].connecting_node[1]}] = i;
        edgeIndexMap[{init_edges[i].connecting_node[1], init_edges[i].connecting_node[0]}] = i;
    }
    for (int sn = 0; sn < service_number; sn++) {
        int start = init_services[sn].source;
        int goal = init_services[sn].destination;

        queue<Node_for_bfs> q;
        unordered_set<int> visited;
        vector<vector<int>> allPaths;
        q.push(Node_for_bfs(start, {}));
        visited.insert(start);
        bool found = false;

        while (!q.empty()) {
            int levelSize = q.size();

            for (int i = 0; i < levelSize; ++i) {
                Node_for_bfs current = q.front();
                q.pop();
                if (current.id == goal) {
                    found = true;
                    allPaths.push_back(current.edgeIndices);
                }

                if (!found) {
                    auto range = graph.equal_range(current.id);
                    for (auto it = range.first; it != range.second; ++it) {
                        int neighbor = it->second;
                        if (visited.find(neighbor) == visited.end()) {
                            vector<int> newEdgeIndices = current.edgeIndices;
                            int edge_idx = edgeIndexMap[{current.id, neighbor}];
                            newEdgeIndices.push_back(edge_idx);
                            q.push(Node_for_bfs(neighbor, newEdgeIndices));
                            visited.insert(neighbor);
                        }
                    }
                }
            }
            if (found) {
                break;
            }
        }
        for (size_t i = 0; i < allPaths.size(); i++) {
            for (size_t j = 0; j < allPaths[i].size(); j++) {
                init_edges[allPaths[i][j]].importance +=
                    float(init_services[sn].service_value) *
                    float(init_services[sn].right_channel - init_services[sn].left_channel + 1) /
                    40 / allPaths.size();
            }
        }
    }
}

// Read environment data
void Initialize_Environment() {
    // Read N & M.
    cin >> node_number >> edge_number;

    // Pre-allocate size
    init_nodes.resize(node_number);
    init_edges.resize(edge_number);

    // Read Channel switching times
    for (int i = 0; i < node_number; i++) {
        init_nodes[i].id = i + 1;
        cin >> init_nodes[i].switching_times;
    }
    // Read edge connections
    for (int i = 0; i < edge_number; i++) {
        int node1;
        int node2;
        cin >> node1 >> node2;
        init_edges[i].connecting_node[0] = node1;
        init_edges[i].connecting_node[1] = node2;
    }

    // Read number of services
    cin >> service_number;

    // Pre-allocate Service_list
    init_services.resize(service_number);

    // Read services details
    for (int i = 0; i < service_number; i++) {
        int S;
        cin >> init_services[i].source >> init_services[i].destination >>
            init_services[i].traversed_number >> init_services[i].left_channel >>
            init_services[i].right_channel >> init_services[i].service_value;
        if (minWavelength > init_services[i].right_channel - init_services[i].left_channel + 1) {
            minWavelength = init_services[i].right_channel - init_services[i].left_channel + 1;
        }
        S = init_services[i].traversed_number;
        init_services[i].id = i + 1;
        init_services[i].passed_edges.resize(S);
        init_services[i].path.resize(S + 1);
        init_services[i].left_channel_per_edge.resize(S);
        init_services[i].right_channel_per_edge.resize(S);
        init_services[i].hold_resource_edges.resize(S);
        for (int j = 0; j < S; j++) {
            cin >> init_services[i].passed_edges[j];
            init_services[i].left_channel_per_edge[j] = init_services[i].left_channel;
            init_services[i].right_channel_per_edge[j] = init_services[i].right_channel;
            init_services[i].hold_resource_edges[j] = true;
            int edge_number = init_services[i].passed_edges[j];
            int node1 = init_edges[edge_number - 1].connecting_node[0];
            int node2 = init_edges[edge_number - 1].connecting_node[1];
            if (j == 0) {
                init_services[i].path[j] = node1;
            }
            init_services[i].path[j + 1] = node2;
            for (int k = init_services[i].left_channel - 1; k < init_services[i].right_channel;
                 k++) {
                init_edges[edge_number - 1].wave_length_occupied[k] = true;
            }
        }
    }

    // Evaluate edge importance
    calcEdgeImportance();
}

int T1 = 0;
int T = 0;

float randf() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }

int randi(int min, int max) { return (rand() % (max - min + 1)) + min; }

// Function to calculate Jaccard similarity between two sets
double jaccard_similarity(const unordered_set<int>& set1, const unordered_set<int>& set2) {
    if (set1.empty() && set2.empty()) return 1.0;
    int intersection = 0;
    for (const auto& elem : set1) {
        if (set2.find(elem) != set2.end()) {
            intersection++;
        }
    }
    int union_size = set1.size() + set2.size() - intersection;
    return static_cast<double>(intersection) / static_cast<double>(union_size);
}

void Output_Scenarios() {
    // Seed the random number generator
    srand(time(0));

    // Maximum number of scenarios
    const int MAX_T1 = 30;

    // Maximum number of edge failures per scenario
    int max_failure_number = 60;
    if (edge_number / 2 < max_failure_number) {
        max_failure_number = edge_number / 2;
    }

    // List to store all generated scenarios as sets
    vector<unordered_set<int>> scenarios;
    scenarios.reserve(MAX_T1);

    // Maximum number of attempts to generate a valid scenario
    const int MAX_ATTEMPTS = 10000;

    // Generate all edge IDs
    vector<int> all_edges(edge_number);
    for (int i = 0; i < edge_number; i++) {
        all_edges[i] = i + 1;
    }

    // Function to shuffle and select a subset
    auto generate_random_subset = [&](int& c_i, unordered_set<int>& subset) -> bool {
        // Randomly decide the number of failures
        c_i = randi(1, max_failure_number);

        // Shuffle the edge list
        vector<int> shuffled_edges = all_edges;
        shuffle(shuffled_edges.begin(), shuffled_edges.end(), default_random_engine(rand()));

        // Select the first c_i edges
        subset.clear();
        for (int i = 0; i < c_i; i++) {
            subset.insert(shuffled_edges[i]);
        }

        return true;
    };

    int generated = 0;
    int attempts = 0;

    while (generated < MAX_T1 && attempts < MAX_ATTEMPTS) {
        attempts++;
        int c_i;
        unordered_set<int> subset;
        if (!generate_random_subset(c_i, subset)) {
            continue;
        }

        bool valid = true;
        for (const auto& existing_subset : scenarios) {
            double sim = jaccard_similarity(subset, existing_subset);
            if (sim > 0.5) {
                valid = false;
                break;
            }
        }

        if (valid) {
            // Add to scenarios
            scenarios.emplace_back(subset);
            generated++;
        }
    }

    // Now, output the number of generated scenarios
    T1 = generated;
    cout << T1 << endl;
    fflush(stdout);

    // Now, output the scenarios
    for (const auto& scenario_set : scenarios) {
        int c_i = scenario_set.size();
        cout << c_i << endl;
        int count = 0;
        for (const auto& edge_id : scenario_set) {
            cout << edge_id << " ";
            count++;
            if (count == c_i) break;
        }
        cout << endl;
        fflush(stdout);
    }
}

void detect_need_to_replan(vector<Service>& services, int e_failed, int& num_need_to_replan) {
    num_need_to_replan = 0;
    for (size_t i = 0; i < services.size(); i++) {
        Service& service = services[i];
        service.need_to_replan = false;
        if (!service.alive) continue;
        for (size_t j = 0; j < service.passed_edges.size(); j++) {
            if (service.passed_edges[j] == e_failed) {
                service.need_to_replan = true;
                service.hold_resource_edges[j] = false;
                break;  // no need to check further edges
            }
        }
        if (service.need_to_replan) {
            num_need_to_replan++;
        }
    }
}

int inspect_edge(int current_left_channel, int current_right_channel, int edge_idx,
                 vector<Edge>& edges_current)
// Check if the edge can accommodate the current service, and whether channel switching is needed.
// Return -1 if cannot accommodate, 0 if can accommodate without switching, 1 if can with switching.
{
    int inspection_result = 0;
    int wavelength = current_right_channel - current_left_channel + 1;
    for (int i = current_left_channel - 1; i < current_right_channel; i++) {
        if (edges_current[edge_idx].wave_length_occupied[i]) {
            inspection_result = -1;
            break;
        }
    }

    if (inspection_result == -1) {
        bool open = false;
        int startWL = -1;
        int endWL = -1;
        for (int i = 0; i < 40; i++) {
            if (!edges_current[edge_idx].wave_length_occupied[i] && !open) {
                open = true;
                startWL = i;
                endWL = i;
            }
            if (edges_current[edge_idx].wave_length_occupied[i] && open) {
                open = false;
                int WL = endWL - startWL + 1;
                if (WL >= wavelength) {
                    inspection_result = 1;
                    break;
                }
            }
            if (open && !edges_current[edge_idx].wave_length_occupied[i]) {
                endWL = i;
            }
        }
    }
    return inspection_result;
}

void plan_channel_switch(vector<bool>& edge_wave_length_occupied, int& changed_left_channel,
                         int& changed_right_channel) {
    int current_WL = changed_right_channel - changed_left_channel + 1;

    bool open = false;
    int startWL = -1;
    int endWL = -1;
    int occupied_part = 0;
    int still_useful_part = 0;
    vector<int> insert_space_left;
    vector<int> insert_space_right;
    for (int i = 0; i < 40; i++) {
        if (!(edge_wave_length_occupied[i]) && !open) {
            open = true;
            startWL = i;
            endWL = i;
        }
        if ((edge_wave_length_occupied[i]) && open) {
            open = false;
            int WL = endWL - startWL + 1;
            if (WL >= current_WL) {
                insert_space_left.push_back(startWL);
                insert_space_right.push_back(endWL);
            }
            if (WL > minWavelength + 5) {
                still_useful_part += WL;
            }
        }
        if (open && !(edge_wave_length_occupied[i])) {
            endWL = i;
        }
        if (edge_wave_length_occupied[i]) {
            occupied_part++;
        }
    }

    float a = randf();
    if (a < 0.5) {
        reverse(insert_space_left.begin(), insert_space_left.end());
        reverse(insert_space_right.begin(), insert_space_right.end());
    }

    if (!insert_space_left.empty()) {
        float best_utilization = 0;
        int choose_space = -1;
        for (size_t i = 0; i < insert_space_left.size(); i++) {
            int original_WL = insert_space_right[i] - insert_space_left[i] + 1;
            int insert_WL_surplus = original_WL - current_WL;
            int still_useful_part1 = still_useful_part;
            int occupied_part1 = occupied_part;
            if (original_WL > minWavelength + 5) {
                still_useful_part1 -= original_WL;
            }
            if (insert_WL_surplus > minWavelength + 5) {
                still_useful_part1 += insert_WL_surplus;
            }
            occupied_part1 += current_WL;
            float current_utilization = float(still_useful_part1 + occupied_part1) / 40;
            if (current_utilization > best_utilization) {
                best_utilization = current_utilization;
                choose_space = i;
            }
        }
        if (choose_space != -1) {
            float b = randf();
            changed_left_channel = insert_space_left[choose_space] + 1;
            changed_right_channel = changed_left_channel + current_WL - 1;
            if (b < 0.5) {
                changed_right_channel = insert_space_right[choose_space] + 1;
                changed_left_channel = changed_right_channel - current_WL + 1;
            }
        }
    }
}

float costCalc(float edge_importance, int current_left_channel, int current_right_channel,
               bool need_switch, int surplus_switch_times, vector<bool>& edge_wave_length_occupied,
               float w_edge_importance, float w_path_distance, float w_channel_switch,
               float w_utilization, float standardize_edge_importance) {
    // Edge importance cost
    float cost = 0.0;
    cost += edge_importance / standardize_edge_importance * w_edge_importance;

    // Path distance cost
    cost += w_path_distance;

    // Channel switching cost
    if (need_switch) {
        if (surplus_switch_times > 0) {
            cost += float(1.0) / surplus_switch_times * w_channel_switch;
        } else {
            cost += 1000.0;  // Penalize if no switch times left
        }
    }

    // Line utilization cost
    for (int i = current_left_channel - 1; i < current_right_channel; i++) {
        edge_wave_length_occupied[i] = true;
    }
    bool open = false;
    int startWL = -1;
    int endWL = -1;
    int still_useful_part = 0;
    int occupied_part = 0;
    for (int i = 0; i < 40; i++) {
        if (!(edge_wave_length_occupied[i]) && !open) {
            open = true;
            startWL = i;
            endWL = i;
        }
        if ((edge_wave_length_occupied[i]) && open) {
            open = false;
            int WL = endWL - startWL + 1;
            if (WL > minWavelength + 5) {
                still_useful_part += WL;
            }
        }
        if (open && !(edge_wave_length_occupied[i])) {
            endWL = i;
        }
        if (edge_wave_length_occupied[i]) {
            occupied_part++;
        }
    }
    cost += float(still_useful_part + occupied_part) / 40 * w_utilization;

    return cost;
}

void priorityQueueSearch(vector<Service>& services, vector<Edge>& edges, vector<Node>& nodes,
                         float w_edge_importance, float w_path_distance, float w_channel_switch,
                         float w_utilization, float standardize_edge_importance,
                         vector<int>& need_replan_services, vector<int>& success_replan,
                         vector<vector<int>>& Path, vector<vector<int>>& left_channel_Path,
                         vector<vector<int>>& right_channel_Path)  // Custom cost function
{
    // Record the services that need to be replanned
    vector<vector<float>> replan_idx_priority;
    int max_value = 0;
    for (int idx : need_replan_services) {
        vector<float> svc_id_priority;
        svc_id_priority.resize(2);
        svc_id_priority[0] = idx;
        svc_id_priority[1] =
            0.4 * (float(services[idx].right_channel - services[idx].left_channel + 1) / 40) +
            0.6 * (float(services[idx].service_value));
        replan_idx_priority.push_back(svc_id_priority);
        if (services[idx].service_value > max_value) {
            max_value = services[idx].service_value;
        }
    }

    // Sort by priority
    sort(replan_idx_priority.begin(), replan_idx_priority.end(),
         [](const vector<float>& a, const vector<float>& b) { return a[1] > b[1]; });

    // Build graph for path planning
    multimap<int, int> graph;
    map<pair<int, int>, int> edgeIndexMap;
    for (int i = 0; i < edge_number; i++) {
        if (edges[i].is_broken) {
            continue;
        }
        int node1 = edges[i].connecting_node[0];
        int node2 = edges[i].connecting_node[1];
        graph.insert({node1, node2});
        graph.insert({node2, node1});
        edgeIndexMap[{node1, node2}] = i;
        edgeIndexMap[{node2, node1}] = i;
    }

    // Priority queue comparison function
    auto compare = [](const Node_for_bfs& a, const Node_for_bfs& b) { return a.cost > b.cost; };

    // Plan for each service needing replanning
    for (size_t rp = 0; rp < replan_idx_priority.size(); rp++) {
        int svc_idx = replan_idx_priority[rp][0];
        int start = services[svc_idx].source;
        int goal = services[svc_idx].destination;

        // Copy current edge occupancy
        vector<Edge> edges_current(edges);
        for (size_t i = 0; i < services[svc_idx].passed_edges.size(); i++) {
            int edge_idx = services[svc_idx].passed_edges[i] - 1;
            for (int j = services[svc_idx].left_channel_per_edge[i] - 1;
                 j < services[svc_idx].right_channel_per_edge[i]; j++) {
                edges_current[edge_idx].wave_length_occupied[j] = false;
            }
        }

        priority_queue<Node_for_bfs, vector<Node_for_bfs>, decltype(compare)> pq(compare);
        unordered_set<int> visited;

        int current_left_channel = services[svc_idx].left_channel;
        int current_right_channel = services[svc_idx].right_channel;

        pq.push(
            Node_for_bfs(start, {}, {start}, current_left_channel, current_right_channel, 0));
        bool found = false;
        while (!pq.empty()) {
            Node_for_bfs current = pq.top();
            pq.pop();

            current_left_channel = current.left_channel;
            current_right_channel = current.right_channel;

            if (visited.find(current.id) != visited.end()) {
                continue;
            }
            visited.insert(current.id);

            if (current.id == goal) {
                found = true;
                Path.push_back(current.edgeIndices);
                left_channel_Path.push_back(current.left_channel_list);
                right_channel_Path.push_back(current.right_channel_list);
                success_replan.push_back(services[svc_idx].id);
                break;
            }

            auto range = graph.equal_range(current.id);
            for (auto it = range.first; it != range.second; ++it) {
                int neighbor = it->second;

                // Check if we can plan on this edge
                int neighbor_edge = edgeIndexMap[{current.id, neighbor}];
                int inspection_result = inspect_edge(current_left_channel, current_right_channel,
                                                     neighbor_edge, edges_current);
                if (inspection_result == -1) {
                    continue;
                }
                if (inspection_result == 0) {
                    // Without channel switching
                    float edge_importance = edges_current[neighbor_edge].importance;
                    int surplus_switch_times = nodes[current.id - 1].switching_times;
                    vector<int> newEdgeIndices = current.edgeIndices;
                    vector<int> newNodeIndices = current.nodeIndices;
                    vector<int> newLeftChannelList = current.left_channel_list;
                    vector<int> newRightChannelList = current.right_channel_list;
                    vector<bool> newUsingSwitch = current.using_switch_list;
                    newEdgeIndices.push_back(neighbor_edge);
                    newNodeIndices.push_back(neighbor);
                    newLeftChannelList.push_back(current_left_channel);
                    newRightChannelList.push_back(current_right_channel);
                    newUsingSwitch.push_back(false);
                    float newCost = current.cost +
                                    costCalc(edge_importance, current_left_channel,
                                             current_right_channel, false, surplus_switch_times,
                                             edges_current[neighbor_edge].wave_length_occupied,
                                             w_edge_importance, w_path_distance, w_channel_switch,
                                             w_utilization, standardize_edge_importance);
                    pq.push(Node_for_bfs(neighbor, newEdgeIndices, newNodeIndices, newCost,
                                         newLeftChannelList, newRightChannelList, newUsingSwitch,
                                         current_left_channel, current_right_channel));
                }
                // With channel switching
                if (nodes[current.id - 1].switching_times > 0) {
                    float edge_importance = edges_current[neighbor_edge].importance;
                    int surplus_switch_times = nodes[current.id - 1].switching_times - 1;
                    vector<int> newEdgeIndices = current.edgeIndices;
                    vector<int> newNodeIndices = current.nodeIndices;
                    vector<int> newLeftChannelList = current.left_channel_list;
                    vector<int> newRightChannelList = current.right_channel_list;
                    vector<bool> newUsingSwitch = current.using_switch_list;
                    newEdgeIndices.push_back(neighbor_edge);
                    newNodeIndices.push_back(neighbor);
                    newUsingSwitch.push_back(true);
                    // Plan channel switch
                    int changed_left_channel = current_left_channel;
                    int changed_right_channel = current_right_channel;
                    plan_channel_switch(edges_current[neighbor_edge].wave_length_occupied,
                                        changed_left_channel, changed_right_channel);
                    newLeftChannelList.push_back(changed_left_channel);
                    newRightChannelList.push_back(changed_right_channel);
                    float newCost = current.cost +
                                    costCalc(edge_importance, changed_left_channel,
                                             changed_right_channel, true, surplus_switch_times,
                                             edges_current[neighbor_edge].wave_length_occupied,
                                             w_edge_importance, w_path_distance, w_channel_switch,
                                             w_utilization, standardize_edge_importance);
                    pq.push(Node_for_bfs(neighbor, newEdgeIndices, newNodeIndices, newCost,
                                         newLeftChannelList, newRightChannelList, newUsingSwitch,
                                         changed_left_channel, changed_right_channel));
                }
            }
        }
        if (!found) {
            success_replan.push_back(-services[svc_idx].id);
        }
    }
}

void rePlan(vector<Service>& services, vector<Edge>& edges, vector<Node>& nodes,
            vector<int>& need_replan_services) {
    // Calculate standardization variable for cost function
    float standardize_edge_importance = 0.0;
    for (int i = 0; i < edge_number; i++) {
        if (edges[i].importance > standardize_edge_importance) {
            standardize_edge_importance = edges[i].importance;
        }
    }

    // Weights
    float w_edge_importance1 = 0.4;
    float w_path_distance1 = 0.1;
    float w_channel_switch1 = 0.3;
    float w_utilization1 = 0.2;
    vector<vector<int>> Path1;
    vector<vector<int>> left_channel_Path1;
    vector<vector<int>> right_channel_Path1;
    vector<int> success_replan;  // -1 indicates failure, else service id

    priorityQueueSearch(services, edges, nodes, w_edge_importance1, w_path_distance1,
                        w_channel_switch1, w_utilization1, standardize_edge_importance,
                        need_replan_services, success_replan, Path1, left_channel_Path1,
                        right_channel_Path1);

    // Apply best plan
    int total_num_success_replan = 0;
    for (size_t i = 0; i < success_replan.size(); i++) {
        if (success_replan[i] > 0) {
            total_num_success_replan++;
            int svc_idx = success_replan[i] - 1;
            services[svc_idx].passed_edges.clear();
            services[svc_idx].left_channel_per_edge.clear();
            services[svc_idx].right_channel_per_edge.clear();
            for (size_t j = 0; j < Path1[i].size(); j++) {
                int edge_idx = Path1[i][j];  // zero-based index
                services[svc_idx].passed_edges.push_back(edge_idx + 1);  // edge IDs start from 1
                services[svc_idx].left_channel_per_edge.push_back(left_channel_Path1[i][j]);
                services[svc_idx].right_channel_per_edge.push_back(right_channel_Path1[i][j]);
                for (int k = left_channel_Path1[i][j] - 1; k < right_channel_Path1[i][j]; k++) {
                    edges[edge_idx].wave_length_occupied[k] = true;
                }
            }
        } else {
            int svc_idx = -success_replan[i] - 1;
            services[svc_idx].alive = false;
        }
    }

    // Output replanning details
    cout << total_num_success_replan << endl;
    fflush(stdout);
    if (total_num_success_replan > 0) {
        for (size_t i = 0; i < success_replan.size(); i++) {
            if (success_replan[i] > 0) {
                int svc_idx = success_replan[i] - 1;
                cout << success_replan[i] << " " << services[svc_idx].passed_edges.size() << endl;
                fflush(stdout);
                for (size_t j = 0; j < services[svc_idx].passed_edges.size(); j++) {
                    cout << services[svc_idx].passed_edges[j] << " "
                         << services[svc_idx].left_channel_per_edge[j] << " "
                         << services[svc_idx].right_channel_per_edge[j] << " ";
                }
                cout << endl;
                fflush(stdout);
            }
        }
    }
}

int main() {
    // Initialize Environment
    Initialize_Environment();

    // Output failure scenarios
    Output_Scenarios();

    // Read total number of scenarios
    cin >> T;

    // Execute scenarios
    for (int t = 0; t < T; t++) {
        // Reset services, edges, and nodes
        vector<Service> services(init_services);
        vector<Edge> edges(init_edges);
        vector<Node> nodes(init_nodes);

        while (true) {
            // Read failed edge
            int e_failed;
            cin >> e_failed;
            // Check for end of scenario
            if (e_failed < 0) {
                break;
            }

            // Update edge status
            edges[e_failed - 1].is_broken = true;
            // Detect services needing replanning
            int num_need_to_replan = 0;
            detect_need_to_replan(services, e_failed, num_need_to_replan);

            // Collect indices of services needing replanning
            vector<int> need_replan_services;
            for (size_t i = 0; i < services.size(); i++) {
                if (services[i].need_to_replan) {
                    need_replan_services.push_back(i);
                }
            }

            // Execute replanning
            if (num_need_to_replan > 0) {
                rePlan(services, edges, nodes, need_replan_services);
            } else {
                cout << 0 << endl;
                fflush(stdout);
            }
        }
    }

    return 0;
}
