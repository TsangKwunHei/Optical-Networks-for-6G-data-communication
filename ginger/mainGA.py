import random
import numpy as np
import sys



class packetSchedulingTask:
    def __init__(self, slices, port_bw, population_size=50, generations=1000, mutation_rate=0.5):
        # 初始化调度器类，设置切片信息、端口带宽、种群大小、代数、变异率等参数
        self.slices = slices  # 切片信息
        self.port_bw = port_bw  # 端口带宽 (Gbps)
        self.population_size = population_size  # 种群大小
        self.generations = generations  # 迭代次数
        self.mutation_rate = mutation_rate  # 变异概率
        self.population = []  # 种群列表

        self.best_solution = None
        self.best_Obj = None



    def optimizeSchedule(self): # 输入参数：plan_search_mde
        # 队列调度优化
        '''
        '''

        self.__genetic_Main()
        return self.best_solution, self.best_Obj

    def __genetic_Main(self):
        '''
        题目参数
        '''
        slices = self.slices
        port_bw = self.port_bw

        '''
        slices = [
            {"num_packets": 3, "slice_bw": 1, "max_delay": 30000, "packets": [(0, 8000), (1000, 16000), (3000, 8000)]},
            {"num_packets": 3, "slice_bw": 1, "max_delay": 30000, "packets": [(0, 8000), (1000, 16000), (3000, 8000)]}
        ]
        '''
        num_packets_total=0;
        for slice in slices:
            num_packets_total += slice["num_packets"]

        '''
        遗传算法参数
        '''
        generations = self.generations #遗传代数
        population_size = self.population_size #种群大小
        rate_tilt = 0.3 #交配参数——概率倾斜
        mutation_rate = self.mutation_rate #变异参数——变异概率
        rate_mutation_plus = 0.5 #变异参数——已经收敛后启用的变异概率
        proportion_random_individual = 0.5 #每次新生代种群中随机个体的比例。会向下取整。
        KillMode = 1 #淘汰规则

        #STEP_0.结构化种群信息
        population = np.zeros((population_size, num_packets_total), dtype=np.int32)
        Obj_list = np.zeros(population_size, dtype=np.double)


        #STEP_1.生成初始种群
        for i in range(population_size):
            population[i] = self.__Genetic_initialize(slices, num_packets_total)
            #合法化
            #population[i] = self.__Genetic_leagalize(population[i], slices)

        #STEP_2.计算初始种群解的适应度
        for i in range(population_size):
            Obj_list[i]= self.__Genetic_adaptationCalc(population[i], slices)


        #遗传大循环
        for gene in range(generations):

            #STEP_3.种群交配操作
            new_population = self.__Genetic_copulate(population, rate_tilt)
                #新种群命名为new_population

            #STEP_4.种群变异操作
            for i in range(population_size):
                if np.random.rand() < mutation_rate:
                    new_population[i]=self.__Genetic_mutate(new_population[i])

            #STEP_5.新种群合法化操作
            #for i in range(population_size):
                #new_population[i] = self.__Genetic_leagalize(new_population[i], slices)

            '''
            #STEP_6.
            新种群优化计算
            % for i=2:length(new_population(:, 1))
            %new_population
            {i, 1} = Optimize(new_population
            {i, 1}, Known_Data);
            '''

            #STEP_7.新种群适应度计算
            new_Obj_list = np.zeros(population_size, dtype=np.double)

            for i in range(population_size):
                new_Obj_list[i]=self.__Genetic_adaptationCalc(new_population[i], slices)

            #STEP_8.合并种群
            #population = [population;new_population]

            #STEP_9.淘汰劣势个体
            adaptation_list = np.zeros(2*population_size, dtype=np.double)
            for i in range(population_size):
                adaptation_list[i] = Obj_list[i]
                adaptation_list[i + population_size] = new_Obj_list[i]
            sort_sequence = np.argsort(adaptation_list)
            sort_sequence = sort_sequence[::-1]
            adaptation_list_sorted = adaptation_list[sort_sequence]
            #总种群个体适应度排序

            if KillMode == 1:
                population1 = np.zeros((population_size, num_packets_total), dtype=np.int32)
                num_individual_remained = np.int32(population_size - np.floor(population_size * proportion_random_individual))
                Obj1_list = np.zeros(population_size, dtype=np.double)
                #保留优势个体的数量
                for i in range(num_individual_remained):
                    if sort_sequence[i] < population_size:
                        population1[i] = population[sort_sequence[i]]
                        Obj1_list[i] = adaptation_list_sorted[i]
                    else:
                        population1[i] = new_population[sort_sequence[i] - population_size]
                        Obj1_list[i] = adaptation_list_sorted[i]
                #添加随机个体
                for i in range(num_individual_remained, population_size):
                    population1[i] = self.__Genetic_initialize(slices, num_packets_total)
                    #population1[i] = self.__Genetic_leagalize(population1[i], slices)
                    Obj1_list[i]= self.__Genetic_adaptationCalc(population1[i], slices)
                #新生代种群的更新
                population = population1
                Obj_list = Obj1_list

        #输出结果
        best_popu = population[0]
        Obj = self.__Genetic_adaptationCalc(best_popu, slices)
        self.best_solution = best_popu
        self.best_Obj = Obj

    def __Genetic_initialize(self,slices,num_packets_total):
        #初始化方法
        randlist = np.random.permutation(num_packets_total)
        individual = np.zeros(num_packets_total,dtype=np.int32)
        idx = 0
        for slice_num in range(len(slices)):
            for pkt in range(slices[slice_num]["num_packets"]):
                individual[randlist[idx]]=slice_num
                idx += 1

        return individual

    def __Genetic_leagalize(self, popu, slices):
        #合法化方法
        '''
        :param
        '''
        packetSequence = np.zeros((len(popu)),dtype=np.int32)
        packet_ts = np.zeros((len(popu)),dtype=np.float32)
        packet_te = np.zeros((len(popu)),dtype=np.float32)
        packet_ts_limit = np.zeros((len(popu)),dtype=np.float32)
        packet_te_limit = np.zeros((len(popu)), dtype=np.float32)

        # 调整所有不符合约束的packet
        failure_scheduled = np.zeros((len(popu)),dtype=np.int32)
        while True:
            idx = 0
            idxes = np.zeros((len(slices)), dtype=np.int32)
            last_te = 0
            all_scheduled_success = True
            for slice_id in popu:
                packetSequence[idx] = slice_id
                packet_ts_limit[idx] = slices[slice_id]["packets"][idxes[slice_id]][0]
                if packet_ts_limit[idx]>last_te:
                    packet_ts[idx] = packet_ts_limit[idx]
                else:
                    packet_ts[idx] = last_te
                packet_te[idx] = packet_ts[idx] + slices[slice_id]["time_consumptions"][idxes[slice_id]]
                last_te = packet_te[idx]
                packet_te_limit[idx] = slices[slice_id]["deadlines"][idxes[slice_id]]
                if packet_te[idx]>packet_te_limit[idx]:
                    failure_scheduled[idx] = 1
                    all_scheduled_success = False
                idx+=1
                idxes[slice_id]+=1
            #是否全部规划成功，是则退出循环
            if all_scheduled_success:
                break
            #否则重规划
            idx = 0
            for reschedule in failure_scheduled:
                if reschedule:
                    ts_limit = packet_ts_limit[idx]
                    te_limit = packet_te_limit[idx]
                    time_consumption = packet_te[idx] - packet_ts[idx]
                    for idx2 in range(len(packet_te)):
                        '''
                        if packet_te[idx2]>=ts_limit and packet_te[idx2]+time_consumption<=te_limit:
                            temp1 = packetSequence[idx]
                            temp2 = popu[idx]
                            for idx3 in range(idx-1,idx2,-1):
                                packetSequence[idx3+1] = packetSequence[idx3]
                                popu[idx3+1] = popu[idx3]
                            packetSequence[idx2+1] = temp1
                            popu[idx2 + 1] = temp2
                            temp = packet_ts_limit[idx]
                            for idx3 in range(idx-1,idx2,-1):
                                packet_ts_limit[idx3+1] = packet_ts_limit[idx3]
                            packet_ts_limit[idx2+1] = temp
                            temp = packet_te_limit[idx]
                            for idx3 in range(idx-1,idx2,-1):
                                packet_te_limit[idx3+1] = packet_te_limit[idx3]
                            packet_te_limit[idx2+1] = temp

                            current_te = packet_te[idx2] + time_consumption
                            for idx3 in range(idx-1,idx2,-1):
                                packet_te[idx3 + 1] = packet_te[idx3]
                                packet_ts[idx3 + 1] = packet_ts[idx3]
                            if packet_ts[idx2 + 2]<current_te:
                                delta = current_te - packet_ts[idx2 + 2]
                                packet_ts[idx2 + 2] += delta
                                packet_te[idx2 + 2] += delta
                            for idx3 in range(idx2+2,len(packet_te)-1):
                                if packet_ts[idx3 + 1]< packet_te[idx3]:
                                    delta = packet_te[idx3] - packet_ts[idx3 + 1]
                                    packet_ts[idx3 + 1] += delta
                                    packet_te[idx3 + 1] += delta
                                else:
                                    break
                            break
                        el
                        '''
                        if packet_te[idx2]+time_consumption>=te_limit:
                            temp =  packetSequence[idx]
                            temp2 = popu[idx]
                            #for idx3 in range(idx2,idx):
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

                            if idx2==0:
                                current_te = time_consumption
                            else:
                                current_te = packet_te[idx2-1] + time_consumption
                            for idx3 in range(idx-1, idx2-1, -1):
                                packet_te[idx3 + 1] = packet_te[idx3]
                                packet_ts[idx3 + 1] = packet_ts[idx3]
                            if idx2 + 1<len(packet_te):
                                if packet_ts[idx2 + 1]<current_te:
                                    delta = current_te - packet_ts[idx2 + 1]
                                    packet_ts[idx2 + 1] += delta
                                    packet_te[idx2 + 1] += delta
                                for idx3 in range(idx2+1,len(packet_te)-1):
                                    if packet_ts[idx3 + 1]< packet_te[idx3]:
                                        delta = packet_te[idx3] - packet_ts[idx3 + 1]
                                        packet_ts[idx3 + 1] += delta
                                        packet_te[idx3 + 1] += delta
                                    else:
                                        break
                            break
                    idx+=1
                else:
                    idx+=1
        return popu
    def __Genetic_adaptationCalc(self, popu, slices):
        '''
        :param popu:

        :return:
        '''

        packet_ts = np.zeros((len(popu)),dtype=np.float32)
        packet_te = np.zeros((len(popu)),dtype=np.float32)
        packet_ts_limit = np.zeros((len(popu)),dtype=np.float32)
        packet_te_limit = np.zeros((len(popu)), dtype=np.float32)

        idx = 0
        idxes = np.zeros((len(slices)), dtype=np.int32)
        last_te = 0
        max_delay = 0
        num_satisfied_user = np.ones((len(slices)), dtype=np.int32)
        for slice_id in popu:
            packet_ts_limit[idx] = slices[slice_id]["packets"][idxes[slice_id]][0]
            if packet_ts_limit[idx] > last_te:
                packet_ts[idx] = packet_ts_limit[idx]
            else:
                packet_ts[idx] = last_te
                if packet_ts[idx] - packet_ts_limit[idx] > max_delay:
                    max_delay = packet_ts[idx] - packet_ts_limit[idx]
            packet_te[idx] = packet_ts[idx] + slices[slice_id]["time_consumptions"][idxes[slice_id]]
            last_te = packet_te[idx]
            packet_te_limit[idx] = slices[slice_id]["deadlines"][idxes[slice_id]]
            if packet_te[idx] > packet_te_limit[idx]:
                num_satisfied_user[slice_id] = 0
            idx += 1
            idxes[slice_id] += 1


        return sum(num_satisfied_user)/len(num_satisfied_user) + 10000/max_delay
    def __Genetic_copulate(self, population, rate_tilt):
        '''

        :param population:
        :param rate_tilt:
        :return:
        '''

        population_size = len(population)
        packet_size = len(population[0])
        new_population = np.zeros((population_size,packet_size),dtype=np.int32)

        randlist = np.random.permutation(population_size)
        half_popu_size = np.int32(population_size/2)
        for ii in range(half_popu_size):
            i1 = randlist[ii * 2 - 1]
            i2 = randlist[ii * 2]
            father = population[i1]
            mother = population[i2]
            boy = np.copy(father)
            girl = np.copy(mother)
            for j in range(packet_size):
                if np.random.rand() < rate_tilt:
                    a = boy[j]
                    idx = boy.tolist().index(mother[j])
                    boy[j] = mother[j]
                    boy[idx] = a
                if np.random.rand() < rate_tilt:
                    a = girl[j]
                    idx = girl.tolist().index(father[j])
                    girl[j] = father[j]
                    girl[idx] = a
            new_population[i1] = boy
            new_population[i2] = girl
        return new_population

    def __Genetic_mutate(self, new_popu):
        packet_num = len(new_popu)
        for i in range(packet_num):
            a = np.random.rand()
            if a < 0.05:
                temp = new_popu[i];
                b=np.random.randint(0,packet_num-1)
                new_popu[i] = new_popu[b]
                new_popu[b] = temp

        return new_popu

    def outputResult(self):
        schedule = self.best_solution
        slices = self.slices

        total_packets = len(schedule)  # 数据包总数
        print(total_packets)

        packet_ts = np.zeros((len(schedule)),dtype=np.int32)
        packet_te = np.zeros((len(schedule)),dtype=np.int32)
        packet_ts_limit = np.zeros((len(schedule)),dtype=np.int32)
        packet_te_limit = np.zeros((len(schedule)), dtype=np.int32)

        idx = 0
        idxes = np.zeros((len(slices)), dtype=np.int32)
        last_te = np.int32(0)
        max_delay = 0
        num_satisfied_user = np.ones((len(slices)), dtype=np.int32)
        for slice_id in schedule:
            packet_ts_limit[idx] = slices[slice_id]["packets"][idxes[slice_id]][0]
            if packet_ts_limit[idx] > last_te:
                packet_ts[idx] = packet_ts_limit[idx]
            else:
                packet_ts[idx] = np.int32(last_te)
                if packet_ts[idx] - packet_ts_limit[idx] > max_delay:
                    max_delay = packet_ts[idx] - packet_ts_limit[idx]
            packet_te[idx] = np.int32(packet_ts[idx] + slices[slice_id]["time_consumptions"][idxes[slice_id]])
            last_te = np.int32(packet_te[idx])
            packet_te_limit[idx] = slices[slice_id]["deadlines"][idxes[slice_id]]
            print(f"{packet_ts[idx]} {slice_id} {idxes[slice_id]}", end=" ")
            idx += 1
            idxes[slice_id] += 1





class GeneticAlgorithmScheduler:


    def initialize_population(self):
        # 初始化种群，生成随机的解决方案
        self.population = [self.create_random_solution() for _ in range(self.population_size)]

    def create_random_solution(self):
        # 创建一个随机的调度解决方案，将所有切片的所有数据包混合随机排列
        all_packets = [(i, j, packet[0], packet[1]) for i, slice_info in enumerate(self.slices) for j, packet in enumerate(slice_info["packets"])]
        random.shuffle(all_packets)  # 随机打乱所有数据包
        return all_packets

    def fitness(self, solution):
        # 计算适应度分数，用于衡量解决方案的好坏
        leave_times = {}  # 存储每个数据包的离开时间
        slice_bandwidths = [s["slice_bw"] for s in self.slices]  # 各个切片的带宽需求
        max_delays = [s["max_delay"] for s in self.slices]  # 各个切片的最大延迟容忍度
        max_waiting_delay = [0] * len(self.slices)  # 每个切片的最大等待延迟

        last_leave_time = 0  # 上一个数据包的离开时间
        for pkt in solution:
            slice_id, pkt_id, arrival_time, size = pkt

            # 确保同一切片内的数据包按到达顺序依次离开
            if (slice_id, pkt_id - 1) in leave_times:
                last_leave_time = max(leave_times[(slice_id, pkt_id - 1)], last_leave_time)

            # 计算当前数据包的离开时间
            leave_time = max(last_leave_time, arrival_time) + (size / self.port_bw)
            leave_times[(slice_id, pkt_id)] = leave_time  # 记录离开时间
            last_leave_time = leave_time  # 更新上一个离开时间

            # 更新当前切片的最大等待延迟
            waiting_delay = leave_time - arrival_time
            max_waiting_delay[slice_id] = max(max_waiting_delay[slice_id], waiting_delay)

            # 检查是否超过切片的最大延迟容忍度
            if waiting_delay > max_delays[slice_id]:
                return float("inf")  # 超过最大延迟容忍度的方案直接淘汰

        # 计算适应度分数，适应度分数越低表示方案越好
        num_slices = len(self.slices)
        fitness_score = sum(max_waiting_delay) / num_slices + 10000 / max(max_waiting_delay)
        return fitness_score

    def selection(self):
        # 选择适应度最好的两个个体作为父代
        sorted_population = sorted(self.population, key=lambda sol: self.fitness(sol))
        return sorted_population[:2]

    def crossover(self, parent1, parent2):
        # 交叉操作，生成两个子代
        split_point = len(parent1) // 2  # 选择交叉点
        # 生成子代1，前一半来自父代1，剩余部分来自父代2（保证无重复数据包）
        child1 = parent1[:split_point] + [p for p in parent2 if p not in parent1[:split_point]]
        # 生成子代2，前一半来自父代2，剩余部分来自父代1（保证无重复数据包）
        child2 = parent2[:split_point] + [p for p in parent1 if p not in parent2[:split_point]]
        return child1, child2

    def mutate(self, solution):
        # 变异操作，随机交换两个数据包的位置
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(solution)), 2)  # 随机选择两个索引
            solution[idx1], solution[idx2] = solution[idx2], solution[idx1]  # 交换这两个数据包的位置

    def run(self):
        # 运行遗传算法，迭代多个代数寻找最优解
        self.initialize_population()  # 初始化种群

        for generation in range(self.generations):
            # 选择操作：选择适应度最好的个体作为父代
            parents = self.selection()

            # 交叉和变异操作，生成下一代种群
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)  # 随机选择两个父代
                child1, child2 = self.crossover(parent1, parent2)  # 交叉生成两个子代
                self.mutate(child1)  # 对子代1进行变异
                self.mutate(child2)  # 对子代2进行变异
                next_population.extend([child1, child2])  # 将子代加入到下一代种群中

            # 保证种群数量不超过设定的种群大小
            self.population = next_population[:self.population_size]

        # 返回找到的最优解
        best_solution = min(self.population, key=lambda sol: self.fitness(sol))
        return best_solution


# 输入解析函数
# 解析输入，生成用于调度的数据
def parse_input():
    '''
    n = 2  # 切片数量
    port_bw = 2  # 端口带宽 (Gbps)
    slices = [
        {"num_packets": 3, "slice_bw": 1, "max_delay": 25000, "packets": [(0, 8000), (1000, 16000), (3000, 16000)]},
        {"num_packets": 3, "slice_bw": 1, "max_delay": 25000, "packets": [(0, 8000), (8000, 16000), (16000, 16000)]}
    ]
    for slice in slices:
        slice["deadlines"] = []
        slice["time_consumptions"] = []
        for num in range(slice["num_packets"]):
            slice["deadlines"].append(slice["packets"][num][0]+slice["max_delay"])
            slice["time_consumptions"].append(slice["packets"][num][1]/port_bw)
    '''
    # Read all input data
    data = sys.stdin.read().split()
    idx = 0  # Count
    '''
    Example Input
    2 2 
    3 1 30000 
    0 8000 1000 16000 3000 8000 
    3 1 30000 
    0 8000 1000 16000 3000 8000 
    '''
    # Number of slices and PortBW
    # Output packet num has to equal input packet num
    n = int(data[idx]);
    idx += 1  # (1 < n <= 10000),
    port_bw = float(data[idx]);
    idx += 1  # in Gbps

    slices = []

    for slice_id in range(n):  # 2pkgs
        m_i = int(data[idx]);
        idx += 1  # num slice packets
        SliceBW_i = float(data[idx]);
        idx += 1  # SliceBW ranges from 0.01Gbps to 10Gbp
        UBD_i = int(data[idx]);
        idx += 1  # slice delay tolerance
        '''
        Slice   SliceBW_i   UBD     num_slice_packets(m_n)
        1       1           3000    3
        2       1           3000    3
        '''
        # sequence information about the indiviual slice
        packets = []
        for pkt_id in range(m_i):
            '''
            s.p     UBD         ts      Deadline
            s1.p1   30000       0       30000
            s2.p1   30000       0       30000
            s1.p2   30000       1000    31000
            s2.p2   30000       1000    31000
            s1.p3   30000       3000    33000
            s2.p3   30000       3000    33000
            '''
            ts = int(data[idx]);
            idx += 1  # arrival time in ns
            pkt_size = int(data[idx]);
            idx += 1  # 512 bit <= pkt_size <= 76800 bit.
            deadline = ts + UBD_i
            '''
                        slices = [
                {"num_packets": 3, "slice_bw": 1, "max_delay": 25000,
                 "packets": [(0, 8000), (1000, 16000), (3000, 16000)]},
                {"num_packets": 3, "slice_bw": 1, "max_delay": 25000,
                 "packets": [(0, 8000), (8000, 16000), (16000, 16000)]}
            ]
            '''
            packets.append((ts,pkt_size))
        slices.append({"num_packets":m_i,"slice_bw":SliceBW_i,"max_delay":UBD_i,"packets":packets})
        for slice in slices:
            slice["deadlines"] = []
            slice["time_consumptions"] = []
            for num in range(slice["num_packets"]):
                slice["deadlines"].append(slice["packets"][num][0] + slice["max_delay"])
                slice["time_consumptions"].append(slice["packets"][num][1] / port_bw)
    return n, port_bw, slices



if __name__ == "__main__":
    # 主程序入口，执行遗传算法调度
    n, port_bw, slices = parse_input()
    ga_scheduler = packetSchedulingTask(slices, port_bw)  # 创建遗传算法调度器
    best_schedule, best_obj = ga_scheduler.optimizeSchedule()  # 运行调度算法，获取最优的调度方案

    # 输出结果
    ga_scheduler.outputResult()


