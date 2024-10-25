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

using namespace std;

struct Service {
    int id;
    int source;
    int destination;
    int traversed_number;
    int left_channel;
    int right_channel;
    int service_value;
    vector<int> path;
    vector<int> passed_edges;
    vector<int> left_channel_per_edge;
    vector<int> right_channel_per_edge;
    vector<bool> hold_resource_edges;
    bool alive = true;
    bool need_to_replan = false;
    vector<vector<int>> allPaths;

};

struct replan_details {
    int server_id = -1;
    int S = 0;
    vector<int> path_and_wavelengths;
};

struct Node {
    int id;
    int switching_times = 0;
    vector<int> connected_edges;
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
    Node_for_bfs(int _id, vector<int> _edgeIndices) :id(_id), edgeIndices(_edgeIndices) {};
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices) : id(_id), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices) {}
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices, int _left_channel, int _right_channel, float _cost) : id(_id), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), left_channel(_left_channel), right_channel(_right_channel), cost(_cost) {}
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices, float _cost) : id(_id), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), cost(_cost) {}
    Node_for_bfs(int _id, vector<int> _edgeIndices, vector<int> _nodeIndices, float _cost,vector<int> _left_channel_list,vector<int> _right_channel_list,vector<bool> _using_switch_list,int _left_channel, int _right_channel):
        id(_id),edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), cost(_cost), left_channel_list(_left_channel_list), right_channel_list(_right_channel_list), using_switch_list(_using_switch_list), left_channel(_left_channel), right_channel(_right_channel) {}
};

struct Edge_for_bfs {
    int id;
    int from;
    int to;
    vector<int> edgeIndices;
    vector<int> nodeIndices;
    float cost = 0.0;
    vector<int> left_channel_list;
    vector<int> right_channel_list;
    vector<bool> using_switch_list;
    int left_channel;
    int right_channel;
    Edge_for_bfs(int _id, int _from, int _to, vector<int> _edgeIndices) :id(_id), from(_from), to(_to), edgeIndices(_edgeIndices) {};
    Edge_for_bfs(int _id, int _from, int _to, vector<int> _edgeIndices, vector<int> _nodeIndices) : id(_id), from(_from), to(_to), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices) {}
    Edge_for_bfs(int _id, int _from, int _to, vector<int> _edgeIndices, vector<int> _nodeIndices, int _left_channel, int _right_channel, float _cost) : id(_id), from(_from), to(_to), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), left_channel(_left_channel), right_channel(_right_channel), cost(_cost) {}
    Edge_for_bfs(int _id, int _from, int _to, vector<int> _edgeIndices, vector<int> _nodeIndices, float _cost) : id(_id), from(_from), to(_to), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), cost(_cost) {}
    Edge_for_bfs(int _id, int _from, int _to, vector<int> _edgeIndices, vector<int> _nodeIndices, float _cost, vector<int> _left_channel_list, vector<int> _right_channel_list, vector<bool> _using_switch_list, int _left_channel, int _right_channel) :
        id(_id), from(_from), to(_to), edgeIndices(_edgeIndices), nodeIndices(_nodeIndices), cost(_cost), left_channel_list(_left_channel_list), right_channel_list(_right_channel_list), using_switch_list(_using_switch_list), left_channel(_left_channel), right_channel(_right_channel) {}
};

struct Edge {
    int id;
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
int N = 5;
int M = 6;
// Read Data Section

vector<int> path_net;
vector<vector<bool>> adjacency_list;


int J = 2;


float randf()
{    
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int randi(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}



//vector<vector<int>> path_net;

// 全局初始环境参数
int node_number;
int edge_number;
int service_number;
vector<Node>init_nodes;
vector<Edge>init_edges;
vector<Service> init_services;

int minWavelength = 40;//用于costCalc函数

// 广度优先搜索 计算边的重要度
void calcEdgeImportance() {

    multimap<int, int> graph;
    multimap<pair<int, int>, int> edgeIndexMap;
    for (int i = 0; i < edge_number; i++) {
        graph.insert({ init_edges[i].connecting_node[0],init_edges[i].connecting_node[1] });
        graph.insert({ init_edges[i].connecting_node[1],init_edges[i].connecting_node[0] });
        edgeIndexMap.insert({ {init_edges[i].connecting_node[0], init_edges[i].connecting_node[1]}, i });
        edgeIndexMap.insert({ {init_edges[i].connecting_node[1],init_edges[i].connecting_node[0]}, i });
    }
    for (int sn = 0; sn < service_number; sn++) {
        int start = init_services[sn].source;
        int goal = init_services[sn].destination;

        queue<Edge_for_bfs> q;
        unordered_set<int> visited;
        vector<vector<int>> allPaths;
        q.push(Edge_for_bfs(-1, start, start, {}));
        bool found = false;

        while (!q.empty()) {
            int levelSize = q.size();
            vector<vector<int>> currentLevelPaths;

            for (int i = 0; i < levelSize; ++i) {
                Edge_for_bfs current = q.front();
                q.pop();

                if (visited.find(current.id) != visited.end()) {
                    continue;
                }
                visited.insert(current.id);

                if (current.to == goal) {
                    found = true;
                    allPaths.push_back(current.edgeIndices);
                }
                int node_id = current.to;
                if (!found) {
                    auto range = graph.equal_range(node_id);
                    for (auto it = range.first; it != range.second; ++it) {
                        int neighbor = it->second;
                        auto edge_range = edgeIndexMap.equal_range({ node_id, neighbor });
                        for (auto it2 = edge_range.first; it2 != edge_range.second; ++it2) {
                            int neighbor_edge = it2->second;
                            if (visited.find(neighbor_edge) == visited.end()) {
                                vector<int> newEdgeIndices = current.edgeIndices;
                                newEdgeIndices.push_back(neighbor_edge);
                                q.push(Edge_for_bfs(neighbor_edge, node_id, neighbor, newEdgeIndices));
                            }
                        }
                    }
                }
            }
            if (found) {
                if (sn == 45) {
                    int a = 1;
                }
                break;
            }
        }
        for (int i = 0; i < allPaths.size(); i++) {
            for (int j = 0; j < allPaths[i].size(); j++) {
                init_edges[allPaths[i][j]].importance += float(init_services[sn].service_value) * float(init_services[sn].right_channel - init_services[sn].left_channel + 1) / 40 / allPaths.size();
            }
        }
    }
}


void Initialize_Environment() 
//读取环境数据 Read environment data 
{
    // Read N & M.
    cin >> node_number >> edge_number;

    // 预分配内存 pre-allocate size
    init_nodes.resize(node_number);
    init_edges.resize(edge_number);

    // 读取信道转换次数 Read Channel switching times
    for (int i = 0; i < node_number; i++) {
        init_nodes[i].id = i + 1;
        cin >> init_nodes[i].switching_times;
    }
    // 读取边连结信息 Read edge connections
    for (int i = 0; i < edge_number; i++) {
        int node1;
        int node2;
        cin >> node1 >> node2;
        init_edges[i].id = i + 1;
        init_edges[i].connecting_node[0] = node1;
        init_edges[i].connecting_node[1] = node2;
        init_nodes[node1 - 1].connected_edges.push_back(i);
        init_nodes[node2 - 1].connected_edges.push_back(i);
    }

    // 读取服务数量 Read J
    cin >> service_number;

    // 预分配内存 Pre-allocate Service_list
    init_services.resize(service_number);

    // 读取服务信息 Read services details
    for (int i = 0; i < service_number; i++) {
        int S;
        cin >> init_services[i].source >> init_services[i].destination >> init_services[i].traversed_number >> init_services[i].left_channel >> init_services[i].right_channel >> init_services[i].service_value;
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
            if (j == 0) {
                if (init_services[i].source == init_edges[edge_number - 1].connecting_node[0]) {
                    init_services[i].path[j] = init_edges[edge_number - 1].connecting_node[0];
                }
                else {
                    init_services[i].path[j] = init_edges[edge_number - 1].connecting_node[1];
                }
            }
            if (init_services[i].path[j] == init_edges[edge_number - 1].connecting_node[0]) {
                init_services[i].path[j + 1] = init_edges[edge_number - 1].connecting_node[1];
            }
            else {
                init_services[i].path[j + 1] = init_edges[edge_number - 1].connecting_node[0];
            }
            
            for (int k = init_services[i].left_channel - 1; k < init_services[i].right_channel; k++) {
                init_edges[edge_number - 1].wave_length_occupied[k] = true;
            }
        }
    }

    /////////////////////////////////////////评估每条边的重要性 evaluate_edges_importance;
    calcEdgeImportance();
}


int T1 = 0;
int T = 0;

void Output_Scenarios() {
    T1 = 0;
    cout << T1 << endl;
    fflush(stdout);  // 刷新输出缓冲区
    int max_failure_number = 60;
    if (edge_number / 2 < max_failure_number) {
        max_failure_number = edge_number / 2;
    }
    for (int i = 0; i < T1; i++) {
        int failure_nuber = randi(1, max_failure_number);
        cout << failure_nuber << endl;
        fflush(stdout);  // 刷新输出缓冲区
        vector<int> randlist;
        randlist.resize(edge_number);
        for (int j = 0; j < edge_number; j++) {
            randlist[j] = j + 1;
        }
        random_shuffle(randlist.begin(), randlist.end());
        for (int j = 0; j < failure_nuber; j++) {
            cout << randlist[j] << " ";
        }
        cout << endl;
        fflush(stdout);  // 刷新输出缓冲区
    }
}

void detect_need_to_replan(vector<Service>* services, int e_failed, int* num_need_to_replan) {
    for (vector<Service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        (*iter).need_to_replan = false;
        if (!(*iter).alive) {
            continue;
        }
        for (vector<int>::iterator iter2 = (*iter).passed_edges.begin(); iter2 != (*iter).passed_edges.end(); iter2++) {
            if (e_failed == (*iter2)) {
                (*iter).need_to_replan = true;
            }
        }
        if ((*iter).need_to_replan) {
            (*num_need_to_replan)++;
        }
    }
}

int inspect_edge(int current_left_channel, int current_right_channel, int edge_idx, vector<Edge>* edges_current) 
// 检查边是否能存放当前服务，若能，是否需要信道转换. 返回-1代表不能存放，返回0代表可以存放，返回1代表可以存放但需要信道转换
{
    int inspection_result = 0;
    int wavelength = current_right_channel - current_left_channel + 1;
    for (int i = current_left_channel - 1; i < current_right_channel; i++) {
        if ((*edges_current)[edge_idx].wave_length_occupied[i]) {
            inspection_result = -1;
        }
    }

    if (inspection_result == -1) {
        bool open = false;
        int startWL = -1;
        int endWL = -1;
        for (int i = 0; i < 40; i++) {
            if (!(*edges_current)[edge_idx].wave_length_occupied[i] && !open) {
                open = true;
                startWL = i;
                endWL = i;
            }
            if ((*edges_current)[edge_idx].wave_length_occupied[i] && open) {
                open = false;
                int WL = endWL - startWL + 1;
                if (WL > wavelength) {
                    inspection_result = 1;
                    break;
                }
            }
            if (open && !(*edges_current)[edge_idx].wave_length_occupied[i]) {
                endWL = i;
            }
        }
    }
    return inspection_result;
}

void plan_channel_switch(vector<bool> edge_wave_length_occupied, int* changed_left_channel, int* changed_right_channel) {

    int current_WL = *changed_right_channel - *changed_left_channel + 1;
    
  
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
        for (int i = 0; i < insert_space_left.size(); i++) {
            int original_WL = insert_space_right[i] - insert_space_left[i]+1;
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
            *changed_left_channel = insert_space_left[choose_space] + 1;
            *changed_right_channel = *changed_left_channel + current_WL - 1;
            if (b < 0.5) {
                *changed_right_channel = insert_space_right[choose_space] + 1;
                *changed_left_channel = *changed_right_channel - current_WL + 1;
            }
        }
    }
}

float costCalc(
    float edge_importance,
    int current_left_channel,
    int current_right_channel,
    bool need_switch,
    int surplus_switch_times,
    vector<bool> edge_wave_length_occupied,
    float w_edge_importance,
    float w_path_distance,
    float w_channel_switch,
    float w_utilization,
    float standardize_edge_importance)
{
    //追加边重要性目标
    float cost = 0.0;
    cost += edge_importance / standardize_edge_importance * w_edge_importance;

    //追加路径距离目标
    cost += w_path_distance;

    //追加信道转换目标
    if (need_switch) {
        cost += float(1.0)/surplus_switch_times * w_channel_switch;
    }

    //计算并追加线路利用率目标
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
            if (WL > minWavelength+5) {
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

void priorityQueueSearch(
    vector<Service>* services,
    vector<Edge>* edges,
    vector<Node>* nodes,
    float w_edge_importance,
    float w_path_distance,
    float w_channel_switch,
    float w_utilization,
    float standardize_edge_importance,
    int num_need_to_replan,
    vector<vector<int>>* Path,
    vector<vector<int>>* left_channel_Path,
    vector<vector<int>>* right_channel_Path,
    vector<vector<bool>>* switch_channel_Path,
    vector<vector<int>>* node_Path,
    vector<int>* success_replan)//自定义代价函数的优先队列搜索 
{
    (*Path).resize(num_need_to_replan);
    (*node_Path).resize(num_need_to_replan);
    (*left_channel_Path).resize(num_need_to_replan);
    (*right_channel_Path).resize(num_need_to_replan);
    (*switch_channel_Path).resize(num_need_to_replan);
    (*success_replan).resize(num_need_to_replan);
    //先整理需要重规划的服务器
    vector<int> replan_idx1;
    vector<int> replan_idx;
    vector<float>replan_priority;
    replan_idx1.resize(num_need_to_replan);
    replan_idx.resize(num_need_to_replan);
    replan_priority.resize(num_need_to_replan);
    int max_value = 0;
    int idx = -1;
    for (int i = 0; i < service_number; i++) {
        if ((*services)[i].need_to_replan) {
            idx++;
            replan_idx1[idx] = i;
            replan_priority[idx] = (*services)[i].right_channel - (*services)[i].left_channel + 1;
            if ((*services)[i].service_value > max_value) {
                max_value = (*services)[i].service_value;
            }
        }
    }

    for (int i = 0; i < num_need_to_replan; i++) {
        int id = replan_idx1[i];
        replan_priority[i]= 
            0.4 * (float(replan_priority[i]) / 40) + 
            0.6 * (float((*services)[id].service_value) / max_value);
    }
    
    vector<int> indices(replan_priority.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }


    // 按优先度进行排序
    sort(indices.begin(), indices.end(),
        [&replan_priority](int a, int b) {
            return replan_priority[a] > replan_priority[b];
        });

    for (int i = 0; i < num_need_to_replan; i++) {
        replan_idx[i] = replan_idx1[indices[i]];
    }

    //刻画路网信息，用于后续路径规划
    multimap<int, int> graph;
    multimap<pair<int, int>, int> edgeIndexMap;
    for (int i = 0; i < edge_number; i++) {
        if ((*edges)[i].is_broken) {
            continue;
        }
        graph.insert({ (*edges)[i].connecting_node[0],(*edges)[i].connecting_node[1] });
        graph.insert({ (*edges)[i].connecting_node[1],(*edges)[i].connecting_node[0] });
        edgeIndexMap.insert({ {(*edges)[i].connecting_node[0], (*edges)[i].connecting_node[1]},i });
        edgeIndexMap.insert({ {(*edges)[i].connecting_node[1], (*edges)[i].connecting_node[0]},i });
    }
    
    //定义的优先队列比较函数
    auto compare = [](const Edge_for_bfs& a, const Edge_for_bfs& b) {
        return a.cost > b.cost;
    };

    // 按排序后的重规划顺序进行规划
    for (int rp = 0; rp < num_need_to_replan; rp++) {
        int svc_id = replan_idx[rp];
        int start = (*services)[svc_id].source;
        int goal = (*services)[svc_id].destination;
        (*success_replan)[rp] = -1;

        // 所有的边的占据情况记录
        vector<Edge> edges_current(*edges);
        for (int i = 0; i < (*services)[svc_id].passed_edges.size(); i++) {
            int edge_idx = (*services)[svc_id].passed_edges[i] - 1;
            for (int j = (*services)[svc_id].left_channel_per_edge[i] - 1; j < (*services)[svc_id].right_channel_per_edge[i]; j++) {
                edges_current[edge_idx].wave_length_occupied[j] = false;
            }
        }

        priority_queue<Edge_for_bfs, vector<Edge_for_bfs>, decltype(compare)> pq(compare);
        unordered_set<int> visited_edge;
        unordered_set<int> visited_node;

        int current_left_channel = (*services)[svc_id].left_channel;
        int current_right_channel = (*services)[svc_id].right_channel;
        
        pq.push(Edge_for_bfs(-1, start, start, {}, {start}, current_left_channel, current_right_channel, 0));
        bool found = false;
        while (!pq.empty()) {
            Edge_for_bfs current = pq.top();
            pq.pop();
            current_left_channel = current.left_channel;
            current_right_channel = current.right_channel;

            if (visited_edge.find(current.id) != visited_edge.end()) {
                continue;
            }
            if (visited_node.find(current.to) != visited_node.end()) {
                continue;
            }
            visited_edge.insert(current.id);
            visited_node.insert(current.to);

            if (current.to == goal) {
                found = true;
                (*Path)[rp] = current.edgeIndices;
                (*left_channel_Path)[rp] = current.left_channel_list;
                (*right_channel_Path)[rp] = current.right_channel_list;
                (*switch_channel_Path)[rp] = current.using_switch_list;
                (*success_replan)[rp] = (*services)[svc_id].id;
                (*node_Path)[rp] = current.nodeIndices;
            }

            if (!found) {
                auto range = graph.equal_range(current.to);
                for (auto it = range.first; it != range.second; ++it) {
                    int neighbor = it->second;
                    if (visited_node.find(neighbor) != visited_node.end()) {
                        continue;
                    }
                    auto edge_range = edgeIndexMap.equal_range({ current.to, neighbor });
                    for (auto it2 = edge_range.first; it2 != edge_range.second; ++it2) {
                        //先检查是否是否可以规划在当前边，以及是否需要转换信道
                        int neighbor_edge = it2->second;
                        if (visited_edge.find(neighbor_edge) != visited_edge.end()) {
                            continue;
                        }
                        int inspection_result = inspect_edge(current_left_channel, current_right_channel, neighbor_edge, &edges_current);
                        if (inspection_result == -1) {
                            continue;
                        }
                        else if (inspection_result == 0) {
                            //信道不转换的方案
                            float edge_importance = edges_current[neighbor_edge].importance;
                            int surplus_switch_times = (*nodes)[current.to - 1].switching_times;
                            //bool edge_occuping_status[40];
                            //copy(begin(edges_current[neighbor_edge].wave_length_occupied), end(edges_current[neighbor_edge].wave_length_occupied), begin(edge_occuping_status)); = {  };
                            vector<int> newEdgeIndices = current.edgeIndices;
                            vector<int> newNodeIndices = current.nodeIndices;
                            vector<int> newLeftChannelList = current.left_channel_list;
                            vector<int> newRightChannelLiet = current.right_channel_list;
                            vector<bool> newUsingSwitch = current.using_switch_list;
                            newEdgeIndices.push_back(neighbor_edge);
                            newNodeIndices.push_back(neighbor);
                            newLeftChannelList.push_back(current_left_channel);
                            newRightChannelLiet.push_back(current_right_channel);
                            newUsingSwitch.push_back(false);
                            float newCost = current.cost + costCalc(
                                edge_importance,
                                current_left_channel,
                                current_right_channel,
                                false,
                                surplus_switch_times,
                                edges_current[neighbor_edge].wave_length_occupied,
                                w_edge_importance,
                                w_path_distance,
                                w_channel_switch,
                                w_utilization,
                                standardize_edge_importance);
                            pq.push(Edge_for_bfs(neighbor_edge, current.to, neighbor, newEdgeIndices, newNodeIndices, newCost, newLeftChannelList, newRightChannelLiet, newUsingSwitch, current_left_channel, current_right_channel));
                        }
                        //信道转换的方案
                        if ((*nodes)[current.to - 1].switching_times > 0) {
                            float edge_importance = edges_current[neighbor_edge].importance;
                            int surplus_switch_times = (*nodes)[current.to - 1].switching_times;
                            vector<int> newEdgeIndices = current.edgeIndices;
                            vector<int> newNodeIndices = current.nodeIndices;
                            vector<int> newLeftChannelList = current.left_channel_list;
                            vector<int> newRightChannelLiet = current.right_channel_list;
                            vector<bool> newUsingSwitch = current.using_switch_list;
                            newEdgeIndices.push_back(neighbor_edge);
                            newNodeIndices.push_back(neighbor);
                            newUsingSwitch.push_back(true);
                            // 计算最佳信道切换方案
                            int changed_left_channel = current_left_channel;
                            int changed_right_channel = current_right_channel;
                            plan_channel_switch(edges_current[neighbor_edge].wave_length_occupied, &changed_left_channel, &changed_right_channel);
                            newLeftChannelList.push_back(changed_left_channel);
                            newRightChannelLiet.push_back(changed_right_channel);
                            float newCost = current.cost + costCalc(
                                edge_importance,
                                changed_left_channel,
                                changed_right_channel,
                                true,
                                surplus_switch_times,
                                edges_current[neighbor_edge].wave_length_occupied,
                                w_edge_importance,
                                w_path_distance,
                                w_channel_switch,
                                w_utilization,
                                standardize_edge_importance);
                            pq.push(Edge_for_bfs(neighbor_edge, current.to, neighbor, newEdgeIndices, newNodeIndices, newCost, newLeftChannelList, newRightChannelLiet, newUsingSwitch, changed_left_channel, changed_right_channel));
                        }
                    }
                }
            }
            if (found) {
                break;
            }
        }
        if ((*success_replan)[rp] < 0) {
            (*success_replan)[rp] = -(*services)[svc_id].id;
        }

        for (int i = 0; i < (*Path)[rp].size(); i++) {
            (*Path)[rp][i]++;
        }
        //规划成功，占据新路径的资源
        for (int j = 0; j < (*Path)[rp].size(); j++) {
            int edge_idx = (*Path)[rp][j] - 1;
            int left_channel = (*left_channel_Path)[rp][j];
            int right_channel = (*right_channel_Path)[rp][j];
            for (int k = left_channel - 1; k < right_channel; k++) {
                (*edges)[edge_idx].wave_length_occupied[k] = true;
            }
            //扣除相应节点的信道转换次数
            if ((*switch_channel_Path)[rp][j]) {
                (*nodes)[(*node_Path)[rp][j] - 1].switching_times--;
            }
        }
    }
}

void rePlan(vector<Service>* services, vector<Edge>* edges, vector<Node>* nodes, int num_need_to_replan) {



    //计算cost函数的归一化变量
    float standardize_edge_importance = 0.0;
    for (int i = 0; i < edge_number; i++) {
        if ((*edges)[i].importance > standardize_edge_importance) {
            standardize_edge_importance = (*edges)[i].importance;
        }
    }

    //尝试权重1
    float w_edge_importance1 = 0.4;
    float w_path_distance1 = 0.1;
    float w_channel_switch1 = 0.3;
    float w_utilization1 = 0.2;
    vector<vector<int>> Path1;
    vector<vector<int>> left_channel_Path1;
    vector<vector<int>> right_channel_Path1;
    vector<vector<bool>> switch_channel_Path1;
    vector<vector<int>> node_Path1;
    vector<int> success_replan; // -1表示规划失败，否则给出服务id
    priorityQueueSearch(
        services,
        edges,
        nodes,
        w_edge_importance1,
        w_path_distance1,
        w_channel_switch1,
        w_utilization1,
        standardize_edge_importance,
        num_need_to_replan,
        &Path1,
        &left_channel_Path1,
        &right_channel_Path1,
        &switch_channel_Path1,
        &node_Path1,
        &success_replan);

    //检查loop
    for (int i = 0; i < success_replan.size(); i++){
        if (success_replan[i] > 0) {
            for (int j = 0; j < node_Path1[i].size()-1; j++) {
                for (int k = j + 1; k < node_Path1[i].size(); k++) {
                    if (node_Path1[i][j] == node_Path1[i][k]) {
                        throw(runtime_error("成环了兄弟"));
                    }
                }
            }
        }
    }



    //应用最好的方案
    int total_num_success_replan = 0;
    vector<int> replan_edge_num;
    replan_edge_num.resize(num_need_to_replan);
    for (int i = 0; i < num_need_to_replan; i++) {
        replan_edge_num[i] = left_channel_Path1[i].size();
        if (success_replan.size()>0){
            if (success_replan[i] >= 0) {
                total_num_success_replan++;
                int svc_idx = success_replan[i] - 1;
                //释放资源
                for (int j = 0; j < (*services)[svc_idx].passed_edges.size(); j++) {
                    int edge_idx = (*services)[svc_idx].passed_edges[j] - 1;
                    int left_channel = (*services)[svc_idx].left_channel_per_edge[j];
                    int right_channel = (*services)[svc_idx].right_channel_per_edge[j];
                    for (int k = left_channel - 1; k < right_channel; k++) {
                        (*edges)[edge_idx].wave_length_occupied[k] = false;
                    }
                }
                (*services)[svc_idx].path.resize(replan_edge_num[i] + 1);
                (*services)[svc_idx].passed_edges.resize(replan_edge_num[i]);
                (*services)[svc_idx].left_channel_per_edge.resize(replan_edge_num[i]);
                (*services)[svc_idx].right_channel_per_edge.resize(replan_edge_num[i]);
                for (int j = 0; j < replan_edge_num[i]; j++) {
                    int node_idx = node_Path1[i][j] - 1;
                    (*services)[svc_idx].path[j] = node_Path1[i][j];
                    (*services)[svc_idx].passed_edges[j] = Path1[i][j];
                    (*services)[svc_idx].left_channel_per_edge[j] = left_channel_Path1[i][j];
                    (*services)[svc_idx].right_channel_per_edge[j] = right_channel_Path1[i][j];
                    //if (switch_channel_Path1[i][j]) {
                    //    (*nodes)[node_idx].switching_times--;
                    //}
                }
                //更新占据的资源
                for (int j = 0; j < (*services)[svc_idx].passed_edges.size(); j++) {
                    int edge_idx = (*services)[svc_idx].passed_edges[j] - 1;
                    int left_channel = (*services)[svc_idx].left_channel_per_edge[j];
                    int right_channel = (*services)[svc_idx].right_channel_per_edge[j];
                    for (int k = left_channel - 1; k < right_channel; k++) {
                        (*edges)[edge_idx].wave_length_occupied[k] = true;
                    }
                }
            }
            else {
                int svc_idx = -success_replan[i] - 1;
                (*services)[svc_idx].alive = false;
            }
        }
    }

    // 输出重规划细节
    cout << total_num_success_replan << endl;
    fflush(stdout);  // 刷新输出缓冲区
    for (int i = 0; i < num_need_to_replan; i++) {
        if (success_replan.size() > 0) {
            if (success_replan[i] > 0) {
                cout << success_replan[i] << " " << replan_edge_num[i] << endl;
                fflush(stdout);  // 刷新输出缓冲区
                for (int j = 0; j < replan_edge_num[i]; j++) {
                    cout << Path1[i][j] << " " << left_channel_Path1[i][j] << " " << right_channel_Path1[i][j] << " ";
                }
                cout << endl;
                fflush(stdout);  // 刷新输出缓冲区
            }
        }
        else {
            cout << 0 << endl;
            fflush(stdout);  // 刷新输出缓冲区
        }
    }
    //cout << 

}



int main(){

    // 初始化环境 Initialize_Environment
    Initialize_Environment();

    // 输出故障场景 Output_Scenarios
    Output_Scenarios();

    // 接收故障场景总数
    cin >> T;

    // 开始执行scenarios
    for (int t = 0; t < T; t++) {
        
        // 重置服务状态 reset services
        vector<Service> services(init_services);
        vector<Edge> edges(init_edges);
        vector<Node> nodes(init_nodes);

        while (true) {
            // 读取故障边 read failure_edge
            int e_failed;
            cin >> e_failed;
            // 识别结束符号
            if (e_failed < 0) {
                break;
            }
            
            // 更新故障发生后边的状态 update_edge_status
            edges[e_failed - 1].is_broken = true;
            // 检查哪些服务需要重新规划
            int num_need_to_replan = 0;
            detect_need_to_replan(&services, e_failed, &num_need_to_replan);
            
            // 执行重规划
            rePlan(&services, &edges, &nodes, num_need_to_replan);
        }
    }


    return 0;
}
