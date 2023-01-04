import random

from environment2.Task import Task

N_task = 10
storage_list = [5000,5000,5000,5000,500,50,50000,1000,3000,10000]
compute_list = [5,500,5000,50000,50000,50000,50000,1000,3000,10000]


def get_random_task():
    """等概率随机返回一个task"""
    n = random.randint(0, N_task - 1)
    return Task(storage_list[n], compute_list[n])


