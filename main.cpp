<<<<<<< HEAD
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

using namespace std;

struct service {
    int s;
    int d;
    int S;
    int L;
    int R;
    int V;
    vector<int> path;
    bool alive = true;
    bool need_to_replan = false;
    vector<bool> hold_resource_path;
    vector<vector<int>> allPaths

};

struct failure_scenario {
    int ci;
    vector<int> edge_ids;
};
struct replan_details {
    int server_id = -1;
    int S = 0;
    vector<int> path_and_wavelengths;
};


// Environment Data
int N = 5;
int M = 6;
// Read Data Section

vector<int> Pi;

vector<int> path_net;
vector<vector<bool>> adjacency_list;




int J = 2;
vector<service> init_Services;


// 故障发生次数
const int T1 = 30;
const int T2 = 0;

float randf()
{    
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int randi(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}


void Initialize_Environment() {
    for (int i = 0; i < N; i++) {
        Pi.push_back(1);
    }
    for (int j = 0; j < N; j++) {
        vector<bool> A;
        adjacency_list.push_back(A);
        for (int j2 = 0; j2 < N; j2++) {
            adjacency_list[j].push_back(false);
        }
    }
    adjacency_list[0][1] = true;
    adjacency_list[1][4] = true;
    adjacency_list[0][3] = true;
    adjacency_list[3][4] = true;
    adjacency_list[0][2] = true;
    adjacency_list[2][4] = true;
    path_net.push_back(0);
    path_net.push_back(1);
    path_net.push_back(1);
    path_net.push_back(4);
    path_net.push_back(0);
    path_net.push_back(3);
    path_net.push_back(3);
    path_net.push_back(4);
    path_net.push_back(0);
    path_net.push_back(2);
    path_net.push_back(2);
    path_net.push_back(4);

    service one_service;
    one_service.s = 1;
    one_service.d = 5;
    one_service.S = 2;
    one_service.L = 1;
    one_service.R = 20;
    one_service.V = 1;
    one_service.path.push_back(1);
    one_service.hold_resource_path.push_back(true);
    one_service.path.push_back(2);
    one_service.hold_resource_path.push_back(true);
    init_Services.push_back(one_service);
    service two_service;
    two_service.s = 1;
    two_service.d = 5;
    two_service.S = 2;
    two_service.L = 21;
    two_service.R = 40;
    two_service.V = 1;
    two_service.path.push_back(1);
    two_service.hold_resource_path.push_back(true);
    two_service.path.push_back(2);
    two_service.hold_resource_path.push_back(true);

    init_Services.push_back(two_service);
}


void detect_need_to_replan(vector<service>* services, failure_scenario scene1) {

    vector<bool> fiber_is_broken;
    for (int i = 0; i < M; i++) {
        fiber_is_broken.push_back(false);
    }
    for (int i = 0; i < scene1.ci; i++) {
        fiber_is_broken[scene1.edge_ids[i]] = true;
    }

    for (vector<service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        int idx = 0;
        for (vector<int>::iterator iter2 = (*iter).path.begin(); iter2 != (*iter).path.end(); iter2++) {
            if (fiber_is_broken[*iter2]) {
                (*iter).need_to_replan = true;
                (*iter).hold_resource_path[idx] = false;
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

void rePlan(vector<service>* services, vector<vector<bool>> adjacency_list) {

}


void available_path_search(vector<service>* services, vector<vector<bool>> adjacency_list, failure_scenario scene1) {
    for (int i = 0; i < scene1.ci; i++) {
        int p1 = path_net[2 * i];
        int p2 = path_net[2 * i + 1];
        adjacency_list[p1][p2] = false;
        adjacency_list[p2][p1] = false;
    }

    for (vector<service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        int startP = (*iter).s - 1;
        int endP = (*iter).d - 1;
        /*
        * 输入例子
        int startP = 0, endP = 4;
        vector<vector<bool>> adjacency_list = {
            {false, true, false, false, true},  // 0号节点与1号和4号节点相邻
            {true, false, true, false, false},  // 1号节点与0号和2号节点相邻
            {false, true, false, true, false},  // 2号节点与1号和3号节点相邻
            {false, false, true, false, true},  // 3号节点与2号和4号节点相邻
            {true, false, false, true, false}   // 4号节点与0号和3号节点相邻
        };
        */

        // dfs法获取所有路径
        (*iter).allPaths = findAllPaths(startP, endP, adjacency_list);
        // 路径全局优化
        rePlan(services, adjacency_list);
    }

}

int main(){

    // 初始化环境 Initialize_Environment
    Initialize_Environment();
    
    // 开始执行scenarios
    int T = T1 + T2;
    for (int t = 0; t < T; t++) {
        vector<service> services(init_Services);

        // 模拟一次损坏
        failure_scenario scene1;
        scene1.ci = randi(1, 2);
        vector<int> randperm;
        for (int m = 0; m < M; m++) {
            randperm.push_back(m);
        }
        random_shuffle(randperm.begin(), randperm.end());
        for (int c = 0; c < scene1.ci; c++) {
            scene1.edge_ids.push_back(randperm[c]);
        }
        // 检查哪些服务需要重新规划
        detect_need_to_replan(&services,scene1);
        available_path_search(&services, adjacency_list, scene1);

 

    }


    return 0;
}
=======
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

using namespace std;

struct service {
    int s;
    int d;
    int S;
    int L;
    int R;
    int V;
    vector<int> path;
    bool alive = true;
    bool need_to_replan = false;
    vector<bool> hold_resource_path;
    vector<vector<int>> allPaths

};

struct failure_scenario {
    int ci;
    vector<int> edge_ids;
};
struct replan_details {
    int server_id = -1;
    int S = 0;
    vector<int> path_and_wavelengths;
};


// Environment Data
int N = 5;
int M = 6;
// Read Data Section

vector<int> Pi;

vector<int> path_net;
vector<vector<bool>> adjacency_list;




int J = 2;
vector<service> init_Services;


// 故障发生次数
const int T1 = 30;
const int T2 = 0;

float randf()
{    
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int randi(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}


void Initialize_Environment() {
    for (int i = 0; i < N; i++) {
        Pi.push_back(1);
    }
    for (int j = 0; j < N; j++) {
        vector<bool> A;
        adjacency_list.push_back(A);
        for (int j2 = 0; j2 < N; j2++) {
            adjacency_list[j].push_back(false);
        }
    }
    adjacency_list[0][1] = true;
    adjacency_list[1][4] = true;
    adjacency_list[0][3] = true;
    adjacency_list[3][4] = true;
    adjacency_list[0][2] = true;
    adjacency_list[2][4] = true;
    path_net.push_back(0);
    path_net.push_back(1);
    path_net.push_back(1);
    path_net.push_back(4);
    path_net.push_back(0);
    path_net.push_back(3);
    path_net.push_back(3);
    path_net.push_back(4);
    path_net.push_back(0);
    path_net.push_back(2);
    path_net.push_back(2);
    path_net.push_back(4);

    service one_service;
    one_service.s = 1;
    one_service.d = 5;
    one_service.S = 2;
    one_service.L = 1;
    one_service.R = 20;
    one_service.V = 1;
    one_service.path.push_back(1);
    one_service.hold_resource_path.push_back(true);
    one_service.path.push_back(2);
    one_service.hold_resource_path.push_back(true);
    init_Services.push_back(one_service);
    service two_service;
    two_service.s = 1;
    two_service.d = 5;
    two_service.S = 2;
    two_service.L = 21;
    two_service.R = 40;
    two_service.V = 1;
    two_service.path.push_back(1);
    two_service.hold_resource_path.push_back(true);
    two_service.path.push_back(2);
    two_service.hold_resource_path.push_back(true);

    init_Services.push_back(two_service);
}


void detect_need_to_replan(vector<service>* services, failure_scenario scene1) {

    vector<bool> fiber_is_broken;
    for (int i = 0; i < M; i++) {
        fiber_is_broken.push_back(false);
    }
    for (int i = 0; i < scene1.ci; i++) {
        fiber_is_broken[scene1.edge_ids[i]] = true;
    }

    for (vector<service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        int idx = 0;
        for (vector<int>::iterator iter2 = (*iter).path.begin(); iter2 != (*iter).path.end(); iter2++) {
            if (fiber_is_broken[*iter2]) {
                (*iter).need_to_replan = true;
                (*iter).hold_resource_path[idx] = false;
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

void rePlan(vector<service>* services, vector<vector<bool>> adjacency_list) {

}


void available_path_search(vector<service>* services, vector<vector<bool>> adjacency_list, failure_scenario scene1) {
    for (int i = 0; i < scene1.ci; i++) {
        int p1 = path_net[2 * i];
        int p2 = path_net[2 * i + 1];
        adjacency_list[p1][p2] = false;
        adjacency_list[p2][p1] = false;
    }

    for (vector<service>::iterator iter = (*services).begin(); iter != (*services).end(); iter++) {
        int startP = (*iter).s - 1;
        int endP = (*iter).d - 1;
        /*
        * 输入例子
        int startP = 0, endP = 4;
        vector<vector<bool>> adjacency_list = {
            {false, true, false, false, true},  // 0号节点与1号和4号节点相邻
            {true, false, true, false, false},  // 1号节点与0号和2号节点相邻
            {false, true, false, true, false},  // 2号节点与1号和3号节点相邻
            {false, false, true, false, true},  // 3号节点与2号和4号节点相邻
            {true, false, false, true, false}   // 4号节点与0号和3号节点相邻
        };
        */

        // dfs法获取所有路径
        (*iter).allPaths = findAllPaths(startP, endP, adjacency_list);
        // 路径全局优化
        rePlan(services, adjacency_list);
    }

}

int main(){

    // 初始化环境 Initialize_Environment
    Initialize_Environment();
    
    // 开始执行scenarios
    int T = T1 + T2;
    for (int t = 0; t < T; t++) {
        vector<service> services(init_Services);

        // 模拟一次损坏
        failure_scenario scene1;
        scene1.ci = randi(1, 2);
        vector<int> randperm;
        for (int m = 0; m < M; m++) {
            randperm.push_back(m);
        }
        random_shuffle(randperm.begin(), randperm.end());
        for (int c = 0; c < scene1.ci; c++) {
            scene1.edge_ids.push_back(randperm[c]);
        }
        // 检查哪些服务需要重新规划
        detect_need_to_replan(&services,scene1);
        available_path_search(&services, adjacency_list, scene1);

 

    }


    return 0;
}
>>>>>>> master
