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

struct Node_for_dfs {
    int id;
    vector<int> edgeIndices;
    Node_for_dfs(int _id, vector<int> _edgeIndices) : id(_id), edgeIndices(_edgeIndices) {}
};

struct Edge {
    int connecting_node[2];
    bool is_broken = false;
    float importance = 0;
    bool wave_length_occupied[40] = { false };
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


// 广度优先搜索 计算边的重要度
void calcEdgeImportance() {
    for (int sn = 0; sn < service_number; sn++) {
        int start = init_services[sn].source;
        int goal = init_services[sn].destination;

        multimap<int, int> graph;
        map<pair<int, int>, int> edgeIndexMap;
        for (int i = 0; i < edge_number; i++) {
            graph.insert({ init_edges[i].connecting_node[0],init_edges[i].connecting_node[1] });
            graph.insert({ init_edges[i].connecting_node[1],init_edges[i].connecting_node[0] });
            edgeIndexMap[{init_edges[i].connecting_node[0], init_edges[i].connecting_node[1]}] = i;
            edgeIndexMap[{init_edges[i].connecting_node[1], init_edges[i].connecting_node[0]}] = i;
        }

        queue<Node_for_dfs> q;
        unordered_set<int> visited;
        vector<vector<int>> allPaths;
        q.push(Node_for_dfs(start, { start }));
        visited.insert(start);
        bool found = false;

        while (!q.empty()) {
            int levelSize = q.size();
            vector<vector<int>> currentLevelPaths;

            for (int i = 0; i < levelSize; ++i) {
                Node_for_dfs current = q.front();
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
                            newEdgeIndices.push_back(edgeIndexMap[{current.id, neighbor}]);
                            q.push(Node_for_dfs(neighbor, newEdgeIndices));
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
                init_edges[allPaths[i][j]].importance += float(init_services[sn].right_channel - init_services[sn].left_channel + 1)/allPaths.size();
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
        S = init_services[i].traversed_number;
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
                init_services[i].path[j] = init_edges[edge_number - 1].connecting_node[0];
            }
            init_services[i].path[j + 1] = init_edges[edge_number - 1].connecting_node[1];
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
    T1 = 30;
    cout << T1 << endl;
    for (int i = 0; i < T1; i++) {
        int failure_nuber = randi(1, 60);
        vector<int> randlist;
        randlist.resize(edge_number);
        for (int j = 0; j < edge_number; j++) {
            randlist[j] = j;
        }
        random_shuffle(randlist.begin(), randlist.end());
        for (int j = 0; j < failure_nuber; j++) {
            cout << randlist[j] << " ";
        }
        cout << endl;
    }
}

void detect_need_to_replan(vector<Service>* services, int e_failed) {
    for (vector<Service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        int idx = 0;
        for (vector<int>::iterator iter2 = (*iter).path.begin(); iter2 != (*iter).path.end(); iter2++) {
            if (e_failed == (*iter2)) {
                (*iter).need_to_replan = true;
                (*iter).hold_resource_edges[idx] = false;
            }
            idx++;
        }
    }
}


// 深度优先搜索（DFS）找到所有路径
void dfs(int currentNode, int endP, const vector<vector<bool>>& adjacency_list,
    vector<bool>& visited, vector<int>& path, vector<vector<int>>& allPaths) {
    // 标记当前节点为已访问
    visited[currentNode] = true;
    path.push_back(currentNode);

    // 如果当前节点是终点，保存路径
    if (currentNode == endP) {
        allPaths.push_back(path);
    }
    else {
        // 遍历所有邻接节点
        for (int neighbor = 0; neighbor < adjacency_list.size(); ++neighbor) {
            if (adjacency_list[currentNode][neighbor] && !visited[neighbor]) {
                // 如果邻接节点未被访问，继续递归
                dfs(neighbor, endP, adjacency_list, visited, path, allPaths);
            }
        }
    }

    // 回溯：取消当前节点的访问状态，继续其他可能的路径
    path.pop_back();
    visited[currentNode] = false;
}

// 找到所有从startP到endP的路径
vector<vector<int>> findAllPaths(int startP, int endP, const vector<vector<bool>>& adjacency_list) {
    int N = adjacency_list.size();
    vector<vector<int>> allPaths;  // 存储所有路径
    vector<bool> visited(N, false);  // 标记节点是否已访问
    vector<int> path;  // 当前路径

    // 调用DFS从起点开始搜索
    dfs(startP, endP, adjacency_list, visited, path, allPaths);

    return allPaths;
}

void rePlan(vector<Service>* services, vector<Edge>* edges, vector<Node>* nodes) {

}


void available_path_search(vector<Service>* services, vector<vector<bool>> adjacency_list) {
    //for (int i = 0; i < scene1.ci; i++) {
    //    int p1 = path_net[2 * i];
    //    int p2 = path_net[2 * i + 1];
    //    adjacency_list[p1][p2] = false;
    //    adjacency_list[p2][p1] = false;
    //}

    //for (vector<Service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
    //    int startP = (*iter).source - 1;
    //    int endP = (*iter).destination - 1;
    //    /*
    //    * 输入例子
    //    int startP = 0, endP = 4;
    //    vector<vector<bool>> adjacency_list = {
    //        {false, true, false, false, true},  // 0号节点与1号和4号节点相邻
    //        {true, false, true, false, false},  // 1号节点与0号和2号节点相邻
    //        {false, true, false, true, false},  // 2号节点与1号和3号节点相邻
    //        {false, false, true, false, true},  // 3号节点与2号和4号节点相邻
    //        {true, false, false, true, false}   // 4号节点与0号和3号节点相邻
    //    };
    //    */

    //    // dfs法获取所有路径
    //    (*iter).allPaths = findAllPaths(startP, endP, adjacency_list);
    //    // 路径全局优化
    //    rePlan(services, adjacency_list);
    //}

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
        // 读取故障总数 read failure_number
        int ci;
        cin >> ci;
        for (int c = 0; c < ci; c++) {
            // 读取故障边 read failure_edge
            int e_failed;
            cin >> e_failed;
            
            // 更新故障发生后边的状态 update_edge_status
            edges[e_failed - 1].is_broken = true;
            // 检查哪些服务需要重新规划
            detect_need_to_replan(&services, e_failed);
            
            // 执行重规划
            rePlan(&services, &edges, &nodes);
        }
    }


    return 0;
}
