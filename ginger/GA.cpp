#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;

vector<int> randperm(int perm_size) {
    vector<int> randlist;
    randlist.resize(perm_size);
    for (int i = 0; i < perm_size; i++) {
        randlist[i] = i;
    }
    shuffle(randlist.begin(), randlist.end(), default_random_engine(rand()));
    return randlist;
}

float randf()
{    
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int randi(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}

struct Packet {
    int ts;
    int pkt_size;
    int ddl;
    int time_consumption;
};

struct Slice
{
    int num_packets = 0;
    int slice_bw = 0;
    int max_delay = 0;
    int total_size = 0;
    vector<Packet>packets;
    vector<int> deadlines;
};


// 全局初始环境参数
int num_slices;
float port_bw;
vector<Slice> slices;
int num_packets_total = 0;

// 最佳结果
vector<int> best_solution;
float best_obj;

// 遗传算法参数
int population_size = 20;
int generations = 0;
float mutation_rate = 0.50;
float proportion_random_individual = 0.50;









// 将阿拉伯数字映射为中文数字字符
string numToChinese(int num) {
    // 定义中文数字字符表
    std::vector<std::string> chineseDigits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };

    // 将数字转换为字符串
    std::string numStr = std::to_string(num);
    std::string result = "";

    // 遍历每一位数字，将其转换为对应的中文字符
    for (char digit : numStr) {
        int digitValue = digit - '0';  // 将字符转换为对应的整数
        result += chineseDigits[digitValue];  // 查找并添加对应的中文字符
    }

    return result;
}








void Genetic_Mutate(vector<int>* popu) 
// 遗传算法变异函数
{
    for (int i = 0; i < num_packets_total; i++) {
        if (randf() < 0.05) {
            int a = randi(0, num_packets_total - 1);
            int temp = (*popu)[i];
            (*popu)[i] = (*popu)[a];
            (*popu)[a] = temp;
        }
    }

}

vector<vector<int>> Genetic_Copulate(vector<vector<int>>* population)
// 遗传算法交配函数
{
    vector<vector<int>> new_population;
    new_population.resize(population_size);

    for (int i = 0; i < population_size; i++) {
        new_population[i].resize(num_packets_total);
        for (int j = 0; j < num_packets_total; j++) {
            new_population[i][j] = (*population)[i][j];
        }
    }

    vector<int> randlist = randperm(population_size);

    int half_popu_size = population_size / 2;
    for (int ii = 0; ii < half_popu_size; ii++) {
        int i1 = randlist[ii * 2];
        int i2 = randlist[ii * 2 + 1];
        vector<int>father((*population)[i1]);
        vector<int>mother((*population)[i2]);
        vector<int>boy(father);
        vector<int>girl(mother);
        for (int j = 0; j < num_packets_total; j++) {
            if (randf() < 0.5) {
                int a = boy[j];
                int idx = -1;
                for (int i = 0; i < num_packets_total; i++) {
                    if (boy[i] == mother[j]) {
                        idx = i;
                    }
                }
                boy[j] = mother[j];
                boy[idx] = a;
            }
            
            if (randf() < 0.5) {
                int a = girl[j];
                int idx = -1;
                for (int i = 0; i < num_packets_total; i++) {
                    if (girl[i] == father[j]) {
                        idx = i;
                    }
                }
                girl[j] = father[j];
                girl[idx] = a;
            }

        }
        for (int j = 0; j < num_packets_total; j++) {
            new_population[i1][j] = boy[j];
            new_population[i2][j] = girl[j];
        }
    }
    return new_population;
}

float Genetic_AdaptationCalc(vector<int>* popu) 
// 遗传算法适应度计算函数
{
    vector<int> slice_pkt_idx;
    slice_pkt_idx.resize(num_slices);
    // 初始化每个slice中pkt的索引
    for (int i = 0; i < num_slices; i++) {
        slice_pkt_idx[i] = -1;
    }

    // 初始化last_pkt_finish_time
    int last_pkt_finish_time = 0;
    // 初始化max_delay
    float max_delay = 0;
    // 初始化slice_finished_on_time 和 slice_started
    vector<bool> slice_finished_on_time;
    slice_finished_on_time.resize(num_slices);
    for (int i = 0; i < num_slices; i++) {
        slice_finished_on_time[i] = true;
    }
    // 初始化slice_starting_time 和 slice_ending_time 和 slice_total_size
    vector<int> slice_starting_time;
    vector<int> slice_ending_time;
    slice_starting_time.resize(num_slices);
    slice_ending_time.resize(num_slices);
    for (int i = 0; i < num_slices; i++) {
        slice_starting_time[i] = slices[i].packets[0].ts;
        slice_ending_time[i] = 0;
    }


    // 开始读取solution并计算目标函数
    for (int i = 0; i < num_packets_total; i++) {
        int slice_idx = (*popu)[i];
        slice_pkt_idx[slice_idx]++;
        int pkt_idx = slice_pkt_idx[slice_idx];

        // 计算te
        int packet_ts = slices[slice_idx].packets[pkt_idx].ts;
        int packet_te = packet_ts;
        if (packet_ts < last_pkt_finish_time) {
            packet_te = last_pkt_finish_time;
        }
        // 更新slice_ending_time
        slice_ending_time[slice_idx] = packet_te;
        // 检测delay是否＞tolerance，更新slice_finished_on_time
        if (packet_te > slices[slice_idx].packets[pkt_idx].ddl) {
            slice_finished_on_time[slice_idx] = false;
        }
        // 计算delay
        int delay = packet_te - packet_ts;
        // 检测并更新max_delay
        if (max_delay < delay) {
            max_delay = delay;
        }

        // 更新last_pkt_finish_time
        last_pkt_finish_time = packet_te + slices[slice_idx].packets[pkt_idx].time_consumption;
    }

    // 计算最终目标函数
    float Obj = 0;
    if (max_delay == 0) {
        Obj = +999999;
    }
    // 检查约束2
    for (int i = 0; i < num_slices; i++) {
        if (slices[i].total_size / (slice_ending_time[i] - slice_starting_time[i]) < 0.95 * slices[i].slice_bw) {
            Obj = -999999;
        }
    }

    Obj += float(10000) / max_delay;
    float num_satisfied_user = 0.0;
    for (int j = 0; j < num_slices; j++) {
        if (slice_finished_on_time[j]) {
            num_satisfied_user += 1;
        }
    }
    Obj += num_satisfied_user / num_slices;
    return Obj;
}




void Genetic_Initialize(vector<int> *popu)
// 遗传算法初始化函数
{
    vector<int> randlist = randperm(num_packets_total);
    int idx = 0;
    for (int i = 0; i < num_slices; i++) {
        for (int j = 0; j < slices[i].num_packets; j++) {
            (*popu)[randlist[idx]] = i;
            idx++;
        }
    }
}



void Genetic_Main() 
// 遗传算法主要函数
{
    // Step0. 结构化种群信息
    vector<vector<int>> population;
    population.resize(population_size);
    vector<float> Obj_list;
    Obj_list.resize(population_size);

    for (int i = 0; i < population_size; i++) {
        population[i].resize(num_packets_total);
    }

    // STEP1.生成初始种群
    for (int i = 0; i < population_size; i++) {
        // 初始化
        Genetic_Initialize(&population[i]);
        // 合法化
        // Genetic_Legalize(slices, num_packets_total)
    }

    // STEP2.计算初始种群适应度
    for (int i = 0; i < population_size; i++) {
        Obj_list[i] = Genetic_AdaptationCalc(&population[i]);
    }

    // 遗传大循环
    for (int gene = 0; gene < generations; gene++) {
        
        //STEP_3.种群交配操作
        vector<vector<int>> new_population = Genetic_Copulate(&population);

        //STEP_4.种群变异操作
        for (int i = 0; i < population_size; i++) {
            Genetic_Mutate(&new_population[i]);
        }

        //STEP_5.新种群合法化操作
        for (int i = 0; i < population_size; i++) {
            //Genetic_leagalize(new_population[i], slices)
        }

        //STEP_6.新种群优化计算
        //  Genetic_Optimize(new_population{ i, 1 }, Known_Data);
        //

        //STEP_7.新种群适应度计算
        vector<float> new_Obj_list;
        new_Obj_list.resize(population_size);
        for (int i = 0; i < population_size; i++) {
            new_Obj_list[i] = Genetic_AdaptationCalc(&new_population[i]);
        }

        //STEP_8.合并种群

        //STEP_9.淘汰劣势个体
        vector<float> adaptation_list;
        adaptation_list.resize(2 * population_size);
        for (int i = 0; i < population_size; i++) {
            adaptation_list[i] = Obj_list[i];
            adaptation_list[i + population_size] = new_Obj_list[i];
        }

        vector<int> sort_sequence;
        sort_sequence.resize(2 * population_size);
        for (int i = 0; i < 2 * population_size; i++) {
            sort_sequence[i] = i;
        }
        sort(sort_sequence.begin(), sort_sequence.end(),
            [&](const int& a, const int& b) {
                return (adaptation_list[a] > adaptation_list[b]);
            }
        );


        vector<vector<int>> population1;
        population1.resize(population_size);
        int num_individual_remained = population_size - int(population_size * proportion_random_individual);
        vector<float> Obj1_list;
        Obj1_list.resize(population_size);
        //保留优势个体
        for (int i = 0; i < num_individual_remained; i++) {
            if (sort_sequence[i] < population_size) {
                population1[i] = population[sort_sequence[i]];
                Obj1_list[i] = adaptation_list[sort_sequence[i]];
            }
            else {
                population1[i] = new_population[sort_sequence[i] - population_size];
                Obj1_list[i] = adaptation_list[sort_sequence[i]];
            }
        }
        // 添加随机个体
        for (int i = num_individual_remained; i < population_size; i++) {
            population1[i].resize(num_packets_total);
            Genetic_Initialize(&population1[i]);
            Obj1_list[i] = Genetic_AdaptationCalc(&population1[i]);
        }
        //新生代种群的更新
        for (int i = 0; i < population_size; i++) {
            population[i] = population1[i];
            Obj_list[i] = Obj1_list[i];
        }

    }


    // 输出结果
    best_solution = population[0];
    best_obj = Obj_list[0];

}




void Initialize_Environment() 
//读取环境数据 Read environment data 
{
    // 读取切片数量和端口bandwidth
    cin >> num_slices >> port_bw;
    //throw runtime_error("第一个例子中num_slices=" + numToChinese(num_slices)+",port_bw=" + numToChinese(port_bw));
    cin.ignore(1000, '\n');
    // 预分配slices内存
    slices.resize(num_slices);

    for (int i = 0; i < num_slices+2060; i++) {
        // 读取切片信息
        cin >> slices[i].num_packets >> slices[i].slice_bw >> slices[i].max_delay;
        cin.ignore(1000, '\n');
        if (slices[i].num_packets == 0) {
            //throw runtime_error("找到一个num_packet=zero的slice,当前切片："+numToChinese(i));
            //throw runtime_error("num_packets=" + numToChinese(slices[i].num_packets)+",slice_bw = " + numToChinese(slices[i].slice_bw) + ",max_delay = " + numToChinese(slices[i].max_delay));
        }
        // 统计packet数量
        num_packets_total += slices[i].num_packets;
        // 预分配packets内存
        slices[i].packets.resize(slices[i].num_packets);
        for (int j = 0; j < slices[i].num_packets; j++) {
            cin >> slices[i].packets[j].ts >> slices[i].packets[j].pkt_size;
            slices[i].packets[j].ddl = slices[i].packets[j].ts + slices[i].max_delay;
            slices[i].packets[j].time_consumption = ceil(float(slices[i].packets[j].pkt_size) / port_bw);
            slices[i].total_size += slices[i].packets[j].pkt_size;
        }
    }

    best_solution.resize(num_packets_total);
    int idx = -1;
    for (int i = 0; i < num_slices; i++) {
        for (int j = 0; j < slices[i].num_packets; j++) {
            idx++;
            best_solution[idx] = i;
        }
    }

}

void Initialize_Environment_Debug()
//读取环境数据 Read environment data 
{
    // 读取切片数量和端口bandwidth

    string line;
    vector<int> line_int;
    getline(cin, line);  // 读取整行输入
    istringstream iss0(line);
    int number0;
    while (iss0 >> number0) {
        line_int.push_back(number0);  // 将每个数字存入 vector
    }
    num_slices = line_int[0];
    port_bw = line_int[1];

    string output_data = "";
    output_data += " " + numToChinese(line_int[0]);
    output_data += " " + numToChinese(line_int[1]) + " 换行";
    //throw runtime_error(output_data + numToChinese(num_slices));
    
    //throw runtime_error("第一个例子中num_slices=" + numToChinese(num_slices)+",port_bw=" + numToChinese(port_bw));
    //cin.ignore(1000, '\n');
    // 预分配slices内存
    slices.resize(num_slices);
    //throw runtime_error(output_data);

    for (int i = 0; i < num_slices + 2060; i++) {
        // 读取切片信息
        
        line.clear();
        line_int.clear();
        getline(cin, line);  // 读取整行输入
        istringstream iss1(line);
        int number1;
        while (iss1 >> number1) {
            line_int.push_back(number1);  // 将每个数字存入 vector
        }

        throw runtime_error("第一行: " + numToChinese(int(line_int.size())));
        //cin >> a;
        //slices[i].num_packets = a;
        //output_data += " " + numToChinese(a);
        //while (true) {
        //    cin >> a;
        //    if (a == 0) {
        //        continue;
        //    }
        //    else {
        //        break;
        //    }
        //}
        //slices[i].slice_bw = a;
        //output_data += " " + numToChinese(a);
        //cin >> a;
        //slices[i].max_delay = a;
        //output_data += " " + numToChinese(a) + " 换行";

        //throw runtime_error("num_packets=" + numToChinese(slices[i].num_packets) + ",slice_bw = " + numToChinese(slices[i].slice_bw) + ",max_delay = " + numToChinese(slices[i].max_delay));
        //cin.ignore(1000, '\n');
        if (i == 50) {
            //throw runtime_error("找到一个num_packet=zero的slice,当前切片："+numToChinese(i));
            //throw runtime_error("num_packets=" + numToChinese(slices[i].num_packets)+",slice_bw = " + numToChinese(slices[i].slice_bw) + ",max_delay = " + numToChinese(slices[i].max_delay));
        }
        // 统计packet数量
        num_packets_total += slices[i].num_packets;
        // 预分配packets内存
        slices[i].packets.resize(slices[i].num_packets);

        line.clear();
        line_int.clear();
        getline(cin, line);  // 读取整行输入
        istringstream iss2(line);
        int number2;
        while (iss2 >> number2) {
            line_int.push_back(number2);  // 将每个数字存入 vector
        }
        for (int j = 0; j < slices[i].num_packets; j++) {
            //cin >> a;
            //slices[i].packets[j].ts = a;
            //output_data += " " + numToChinese(a);
            //cin >> a;
            //slices[i].packets[j].pkt_size = a;
            //output_data += " " + numToChinese(a);
            slices[i].packets[j].ts = line_int[j * 2];
            slices[i].packets[j].pkt_size = line_int[j * 2 + 1];
            slices[i].packets[j].ddl = slices[i].packets[j].ts + slices[i].max_delay;
            slices[i].packets[j].time_consumption = ceil(float(slices[i].packets[j].pkt_size) / port_bw);
            slices[i].total_size += slices[i].packets[j].pkt_size;
        }
        output_data += " 换行";
        //throw runtime_error(output_data);
    }

    best_solution.resize(num_packets_total);
    int idx = -1;
    for (int i = 0; i < num_slices; i++) {
        for (int j = 0; j < slices[i].num_packets; j++) {
            idx++;
            best_solution[idx] = i;
        }
    }

}



/*

class packetSchedulingTask:
    def __init__(self, slices, port_bw, population_size=20, generations=50, mutation_rate=0.5):
        # 初始化调度器类，设置切片信息、端口带宽、种群大小、代数、变异率等参数
        self.slices = slices  # 切片信息
        self.port_bw = port_bw  # 端口带宽 (Gbps)

        self.population = []  # 种群列表

        self.best_solution = None
        self.best_Obj = None

    def optimizeSchedule(self): # 输入参数：plan_search_mde
        # 队列调度优化
        '''
        '''




    def __Genetic_leagalize(self, popu, slices):
        # 合法化方法
        '''
        :param
        '''
        packetSequence = [0] * len(popu)
        packet_ts = [0.0] * len(popu)
        packet_te = [0.0] * len(popu)
        packet_ts_limit = [0.0] * len(popu)
        packet_te_limit = [0.0] * len(popu)

        # 调整所有不符合约束的packet
        failure_scheduled = [0] * len(popu)
        while True:
            idx = 0
            idxes = [0] * len(slices)
            last_te = 0
            all_scheduled_success = True
            for slice_id in popu:
                packetSequence[idx] = slice_id
                packet_ts_limit[idx] = slices[slice_id]["packets"][idxes[slice_id]][0]
                if packet_ts_limit[idx] > last_te:
                    packet_ts[idx] = packet_ts_limit[idx]
                else:
                    packet_ts[idx] = last_te
                packet_te[idx] = packet_ts[idx] + slices[slice_id]["time_consumptions"][idxes[slice_id]]
                last_te = packet_te[idx]
                packet_te_limit[idx] = slices[slice_id]["deadlines"][idxes[slice_id]]
                if packet_te[idx] > packet_te_limit[idx]:
                    failure_scheduled[idx] = 1
                    all_scheduled_success = False
                idx += 1
                idxes[slice_id] += 1
            # 是否全部规划成功，是则退出循环
            if all_scheduled_success:
                break
            # 否则重规划
            idx = 0
            for reschedule in failure_scheduled:
                if reschedule:
                    ts_limit = packet_ts_limit[idx]
                    te_limit = packet_te_limit[idx]
                    time_consumption = packet_te[idx] - packet_ts[idx]
                    for idx2 in range(len(packet_te)):
                        if packet_te[idx2] + time_consumption >= te_limit:
                            temp = packetSequence[idx]
                            temp2 = popu[idx]
                            # for idx3 in range(idx2,idx):
                            for idx3 in range(idx-1, idx2-1, -1):
                                packetSequence[idx3+1] = packetSequence[idx3]
                                popu[idx3 + 1] = packetSequence[idx3]

                            packetSequence[idx2] = temp
                            popu[idx2] = temp2

                            temp = packet_ts_limit[idx]
                            for idx3 in range(idx-1, idx2-1, -1):
                                packet_ts_limit[idx3+1] = packet_ts_limit[idx3]
                            packet_ts_limit[idx2] = temp
                            temp = packet_te_limit[idx]
                            for idx3 in range(idx-1, idx2-1, -1):
                                packet_te_limit[idx3+1] = packet_te_limit[idx3]
                            packet_te_limit[idx2] = temp

                            if idx2 == 0:
                                current_te = time_consumption
                            else:
                                current_te = packet_te[idx2-1] + time_consumption
                            for idx3 in range(idx-1, idx2-1, -1):
                                packet_te[idx3 + 1] = packet_te[idx3]
                                packet_ts[idx3 + 1] = packet_ts[idx3]
                            if idx2 + 1 < len(packet_te):
                                if packet_ts[idx2 + 1] < current_te:
                                    delta = current_te - packet_ts[idx2 + 1]
                                    packet_ts[idx2 + 1] += delta
                                    packet_te[idx2 + 1] += delta
                                for idx3 in range(idx2+1, len(packet_te)-1):
                                    if packet_ts[idx3 + 1] < packet_te[idx3]:
                                        delta = packet_te[idx3] - packet_ts[idx3 + 1]
                                        packet_ts[idx3 + 1] += delta
                                        packet_te[idx3 + 1] += delta
                                    else:
                                        break
                            break
                    idx += 1
                else:
                    idx += 1
        return popu









def main():
    # 主程序入口，执行遗传算法调度
    n, port_bw, slices = parse_input()
    ga_scheduler = packetSchedulingTask(slices, port_bw)  # 创建遗传算法调度器
    best_schedule, best_obj = ga_scheduler.optimizeSchedule()  # 运行调度算法，获取最优的调度方案

    # 输出结果
    ga_scheduler.outputResult()


if __name__ == "__main__":
    # Start the main function in a separate thread to avoid potential recursion limits
    threading.Thread(target=main).start()
*/

void Output_Solution() {
    //输出数据包总数
    fflush(stdout);
    cout << num_packets_total<<endl;
    fflush(stdout);

    vector<int> slice_pkt_idx;
    slice_pkt_idx.resize(num_slices);
    // 初始化每个slice中pkt的索引
    for (int i = 0; i < num_slices; i++) {
        slice_pkt_idx[i] = -1;
    }

    // 初始化last_pkt_finish_time
    int last_pkt_finish_time = 0;


    // 开始读取solution并计算目标函数
    for (int i = 0; i < num_packets_total; i++) {
        int slice_idx = best_solution[i];
        slice_pkt_idx[slice_idx]++;
        int pkt_idx = slice_pkt_idx[slice_idx];

        // 计算te
        int packet_ts = slices[slice_idx].packets[pkt_idx].ts;
        int packet_te = packet_ts;
        if (packet_ts < last_pkt_finish_time) {
            packet_te = last_pkt_finish_time;
        }

        // 更新last_pkt_finish_time
        last_pkt_finish_time = packet_te + slices[slice_idx].packets[pkt_idx].time_consumption;

        cout << packet_te << " " << slice_idx << " " << pkt_idx << " ";
    }
    fflush(stdout);
}


int main(){

    // 初始化环境 Initialize_Environment
    Initialize_Environment_Debug();

    // 调用遗传算法进行调序
    //Genetic_Main();

    // 输出排序结果
    Output_Solution();

    return 0;
}
