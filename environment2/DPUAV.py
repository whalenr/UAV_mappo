import math

import numpy as np


from environment2.Constant import bs_computing_capacity, DPUAV_speed
from environment2.Position import Position
from environment2.Task import Task
from environment2.UAV import UAV, calcul_channel_gain, calcul_SNR
# from environment2.UE import UE

max_compute = 4
"""DP-UAV每个时刻最多能并行计算的用户数量"""


class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self, position: Position):
        super().__init__(position, DPUAV_speed)
        self.B_ue = 1 * (10 ** 6)
        """与ue之间的传输带宽(Hz)"""

        self.transmission_energy = 1 * (10 ** (-3))
        """无人机传输信号发射能耗(j)"""

        self.computing_capacity = 5 * (10 ** 7)
        """DPUAV的计算能力，单位为cpu cycle/s"""

        self.link_range = 172
        """DPUAV和UE之间连接距离的限制，在此范围内才可以连接,单位为m"""

        self.rate_BS = 4 * (10 ** (6))
        """与BS之间的通信速率(bit/s)"""
        self.e = 1*(10**(-20))
        """功耗系数"""

    def get_transmission_time_with_BS(self, ue) -> float:
        """传输单个ue任务到BS的时间(s)"""
        return ue.task.storage / self.rate_BS

    def get_transmission_energy_with_BS(self) -> float:
        """传输单个UE任务到BS消耗的功耗(J)"""
        return self.transmission_energy

    def get_compute_time(self, task: Task) -> float:
        """任务所需要的计算时间(s)"""
        return task.compute / self.computing_capacity

    def get_compute_energy(self, task: Task) -> float:
        """计算任务所需要的能耗(j)"""
        return self.e*(self.computing_capacity**2)*task.compute

    def calcul_single_compute_and_offloading_aoi(self, ue, decisions: int):
        """计算不同卸载策略下的AOI，没有进行卸载则返回None"""
        if decisions == 0:  # 不卸载
            return None
        elif decisions == 1:  # 卸载到UAV
            waiting_time = ue.task.wating_time
            transmission_time = ue.get_transmission_time(self)
            compute_time = self.get_compute_time(ue.task)
            return waiting_time + transmission_time + compute_time
        else:  # 卸载到BS
            waiting_time = ue.task.wating_time
            transmission_time_1 = ue.get_transmission_time(self)
            transmission_time_2 = self.get_transmission_time_with_BS(ue)
            compute_time = ue.task.compute / bs_computing_capacity
            return waiting_time + transmission_time_1 + transmission_time_2 + compute_time

    def calcul_single_compute_and_offloading_energy(self, ue, decision: int):
        """计算不同卸载策略下的UAV的能量消耗，没有卸载则没有能量消耗"""
        if decision == 0:
            return 0
        elif decision == 1:
            compute_energy = self.get_compute_energy(ue.task)
            return compute_energy
        else:
            transmission_energy = self.get_transmission_energy_with_BS()
            return transmission_energy

    # def execute_compute_and_offloading(self, ues: [UE], descisions: [int]):
    #     if len(ues) == len(descisions):
    #         print('error,len of ues != len of decisions')
    #         return None
    #     if descisions.count(1) > max_compute:
    #         print('error, exceed max_compute')
    #
    #     return []
