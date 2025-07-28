import random
import numpy as np
import time

from nbformat.v4 import new_raw_cell

start_time = time.time()

class GA_KP():
    #参数初始化，包括种群数量，每个染色体上基因的数量，重量限制，导入数据
    def __init__(self, num_pop, num_gene,WEIGHT_LIMIT,p_c,p_m,data):
        self.num_pop=num_pop
        self.num_gene=num_gene
        self.weight_limit=WEIGHT_LIMIT
        self.p_c=p_c
        self.p_m=p_m
        self.data=data
        self.pop = self.init_pop()
        self.fitness=self.fitness_func()
    #检验一个解是否可行，若不可行则对其处理。
    def feasible_test(self,solution):
        one_index=np.where(solution== 1)[0]#得到数组中所有值为1的数值对应的索引
        weight=0
        for index in one_index:
            weight+=self.data[index+ 1][0]
        if weight <=self.weight_limit:
            return solution
        else:
            new_solution=solution.copy()
            sorted_one_index=sorted(one_index, key=lambda i: self.data[i+ 1][1]/self.data[i+ 1][0])#将已装包的物品的价值除以重量得到性价比，再按性价比从低到高排序
            i=0
            while weight>self.weight_limit:#按性价比从低到高取出，while循环结束时便得到一个可行解、
                weight=weight-self.data[sorted_one_index[i]+ 1][0]
                new_solution[sorted_one_index[i]]=0#取出物品，故对应基因为0.
                i=i+1
            zero_index=np.where(new_solution==0)[0]  # 得到新解中所有值为0的数值对应的索引
            sorted_zero_index=sorted(zero_index, key=lambda i: self.data[i+ 1][1]/self.data[i+ 1][0],reverse=True)#求出未装包的物品的性价比并从高到低进行排序。
            for index in sorted_zero_index:#按照性价比从高到低将那些未装包的物品装入背包。
                if weight+self.data[index+ 1][0]<=self.weight_limit:#如果满足重量约束，那就装进去，否则跳过这个物品
                    weight = weight + self.data[index + 1][0]
                    new_solution[index]=1
            return new_solution
    #种群初始化,随机生成每个解并经过可行性检验。
    def init_pop(self):
        pop=[]
        for _ in range(self.num_pop):
            chromosome=np.random.randint(0, 2, size=self.num_gene)#先随机生成一个染色体，此时可能是不可行解
            pop.append(self.feasible_test(chromosome))#对刚生成的染色体进行可行性检验并添加到种群中。
        return pop
    #适应度函数，即所有装入背包物品的总价值
    def fitness_func(self):
        values=[]
        for chromosome in self.pop:
            one_index = np.where(chromosome== 1)[0]
            sum=0
            for index in one_index:
                sum+=self.data[index+ 1][1]
            values.append(sum)
        return np.array(values)
    #轮盘赌选择3个父代
    def select_wheel(self):
        sum_value=sum(self.fitness)
        p=[i/sum_value for i in self.fitness]
        three_index=[]
        for _ in range(3):
            rand=np.random.rand()
            for i, sub in enumerate(p):
                if rand>= 0:
                    rand-= sub
                    if rand< 0:
                        three_index.append(i)
        return np.array(self.pop[three_index[0]]),np.array(self.pop[three_index[1]]),np.array(self.pop[three_index[2]])
    #用轮盘赌得到的父代进行交叉生成子代
    def crossover(self,p1,p2):
        cx_point=np.random.randint(1,7)#表示前cx_point个基因互换,取值为[1,7)的一个整数
        new_p1=np.hstack((p2[:cx_point], p1[cx_point:]))
        new_p2=np.hstack((p1[:cx_point], p2[cx_point:]))
        return self.feasible_test(new_p1),self.feasible_test(new_p2)
    #变异算子
    def mutate(self,p):
        if np.random.random()<self.p_m:
            start = random.randint(0, 4)  # 得到一个[0,4]的随机整数
            end = random.randint(start + 2, 6)
            new_p = p.copy()
            new_p[start:end] = new_p[start:end][::-1]
        else:
            new_p = p.copy()
        return self.feasible_test(new_p)
    #运行主函数
    def main(self):
        max_value=max(self.fitness)
        max_solution=self.pop[np.argmax(self.fitness)].copy()
        print('初代最大价值和最优解为:',max_value,max_solution)
        for _ in range(100):
            new_pop=[self.pop[np.argmax(self.fitness)]]
            p1,p2,p3=self.select_wheel()
            if np.random.random()<self.p_c:
                new_p1, new_p2 = self.crossover(p1, p2)
            else:
                new_p1,new_p2=p1,p2
            new_p1,new_p2,new_p3=self.mutate(new_p1),self.mutate(new_p2),self.mutate(p3)
            new_pop.append(new_p1)
            new_pop.append(new_p2)
            new_pop.append(new_p3)
            self.pop=new_pop.copy()
            self.fitness=self.fitness_func()
            if max_value<max(self.fitness):
                max_value=max(self.fitness)
                max_solution=self.pop[np.argmax(self.fitness)].copy()
        print('最终的最大价值和最优解为:',max_value,max_solution)


#导入初始数据，6个物品和它们的重量、价格
data= {
    1: [10, 15],
    2: [15, 25],
    3: [20, 35],
    4: [25, 45],
    5: [30, 55],
    6: [35, 70]}
model=GA_KP(num_pop=4,num_gene=len(data),WEIGHT_LIMIT=80,p_c=0.9,p_m=0.01,data=data)
model.main()
end_time = time.time()
print("代码运行时间：", end_time - start_time, "秒")