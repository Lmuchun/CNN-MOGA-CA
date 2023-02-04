# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 22:18:20 2022

@author: Lenovo
"""

# -*- coding: utf-8 -*-
import pickle
import geatpy as ea
import numpy as np
from moga_ca import ca
import time
import datetime
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):
        name = 'MyProblem'  # name
        M = 2  # 初始化M（目标维数）
        maxormins = [-1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 108  # 初始化Dim（决策变量维数）——（72+6）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [-50] * Dim  # 决策变量下界
        ub = [50] * Dim  # 决策变量上界

        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.PoolType = PoolType
        num_cores = int(mp.cpu_count())  # 获得计算机的核心数
        self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):
        Vars = pop.Phen
        weight_list = []
        for i in range(Vars.shape[0]):
            weight = Vars[i]
            weight_list.append(weight)

        start = time.time()
        result = self.pool.map_async(ca, weight_list)
        result.wait()
        end = time.time() - start
        print(str(round(end, 3)) + '秒')
        f = result.get()
        f = np.stack(f, axis=0)
        pop.ObjV = f
        pop.CV = 0.15-f[:, 1].reshape(-1, 1)


if __name__ == '__main__':

    NIND = 108
    problem = MyProblem(PoolType='Process')
    refer_ = np.array([1, 1])
    problem.ReferObjV = np.tile(refer_, (NIND, 1))

    def outFunc(alg, pop):  # alg 和 pop为outFunc的固定输入参数，分别为算法对象和每次迭代的种群对象。
        print('第 %d 代' % alg.currentGen)
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_archive_templet(problem, ea.Population(Encoding='RI', NIND=108),
                                                MAXGEN=250,  # 最大进化代数。
                                                logTras=1,
                                                outFunc=outFunc)

    res = ea.optimize(myAlgorithm, verbose=True, drawing=2,
                      outputMsg=True, drawLog=True, saveFlag=True)

    with open("Diction/myDictionary{}.pkl".format(
            datetime.datetime.now().strftime('%d%H%M')), "wb") as tf:
        pickle.dump(res, tf)
    """===========================调用算法模板进行种群进化======================="""
