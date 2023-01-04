import random

import numpy as np
import math

from environment2.Library import get_random_task
from environment2.Position import Position
from environment2.Task import Task
from environment2.UAV import calcul_SNR, calcul_channel_gain
from environment2.DPUAV import DPUAV


class UE:
    def __init__(self, position, speed_limit=0):
        self.position = position
        """UE所在位置"""

        self.high_probability = 1
        """高电量时每个时间间隔产生数据的概率，待定"""
        self.low_probability = 1
        """低电量时每个时间间隔产生数据的概率，待定"""

        self.energy = 1 * (10 ** (-5))
                      # * random.random()
        """用户的电量(j)"""
        self.energy_max = 1 * (10 ** (-5))
        """电量的最大值"""
        self.energy_threshold = 2 * (10 ** (-6))
        """电量阈值，低于阈值，进入低功耗状态"""
        self.energy_state = 1
        """电量状态，1为高电量，0为低电量"""
        self.energy_conversion_efficiency = 0.1
        """无线充电时能量收集效率"""

        self.task = None
        """生成好的任务"""

        self.speed_limit = speed_limit
        """速度限制"""
        self.time_slice = 1
        self.move_limit = self.speed_limit * self.time_slice
        """每个时间间隔移动距离的限制，反应了用户的移动速度"""

        self.transmission_power = 1 * (10 ** (-5))
        """UE的发射功率(w)"""
        self.collect_energy = 5 * (10 ** (-7))
        """UE采集一个数据需要的能量(j)"""

    # 距离相关函数
    def distance_DPUAV(self, dpuav: DPUAV) -> float:
        """与DPUAV的距离"""
        return self.position.distance(dpuav.position)

    def if_link_DPUAV(self, dpuav: DPUAV) -> bool:
        """是否与DPUAV相连"""
        return self.position.if_connect(dpuav.position, dpuav.link_range)

    # 移动相关函数
    # def move_by_radian(self, radian: float, distance: float):
    #     """用户水平移动，弧度形式"""
    #     if not 0 <= distance <= self.move_limit:
    #         print("移动距离超出限制")
    #         return False
    #     self.position.move_by_radian(radian, distance)
    #
    # def move_by_radian_rate(self, radian: float, rate: float):
    #     """用户水平移动，rate参数为0到1之间的数"""
    #     self.move_by_radian(radian, self.move_limit * rate)

    # 电量相关函数
    def update_energy_state(self):
        """更新电量状态"""
        if self.energy > self.energy_threshold:
            self.energy_state = 1
        else:
            self.energy_state = 0

    def charge(self, energy: float):
        """给UE充电，单位为J"""
        temp_energy = energy * self.energy_conversion_efficiency
        self.energy = min(self.energy_max, self.energy + temp_energy)
        self.update_energy_state()  # 更新电量状态

    def discharge(self, energy: float):
        """UE耗电，电量足够则扣除电量，返回True，否则不扣除电量并返回False"""
        if energy <= self.energy:
            self.energy -= energy
            self.update_energy_state()  # 更新电量状态
            return True
        return False

    def get_energy(self):
        """返回当前电量"""
        return self.energy

    def get_energy_state(self):
        """返回电量状态"""
        return self.energy_state

    # 传输相关函数
    def get_transmission_rate_with_UAV(self, uav: DPUAV) -> float:
        """DPUAV和UE之间实际的传输速率,单位为bit/s"""
        SNR = calcul_SNR(self.transmission_power)
        gain = calcul_channel_gain(uav.position, self.position)
        return uav.B_ue * math.log2(1 + gain * SNR)

    def get_transmission_time(self, uav: DPUAV) -> float:
        """UE传输单个任务到无人机的时间(s)"""
        rate = self.get_transmission_rate_with_UAV(uav)
        return self.task.storage / rate

    def get_transmission_energy(self, uav: DPUAV) -> float:
        """传输单个ue任务到无人机的能耗(J)"""
        energy = self.transmission_power * self.get_transmission_time(uav)
        return energy

    # 生成数据相关函数

    def generate_task(self):
        """每个时隙的开始执行，按照电量产生数据并消耗能量，如果不生成数据，则waiting_time+1"""
        generate = None  # 是否产生新数据
        if self.energy_state == 1:
            # 高电量
            generate = random.random() < self.high_probability
        else:
            # 低电量
            generate = random.random() < self.low_probability
        if generate and self.discharge(self.collect_energy):  # 如果要生成新数据和如果电量足够并扣除电量
            self.task = get_random_task()  # 生成新任务
        else:
            if self.task is not None:
                self.task.step()  # waiting_time + 1

    def get_lambda(self) -> float:
        """返回目前UE产生数据的概率"""
        if self.energy_state == 1:
            return self.high_probability
        else:
            return self.low_probability

    def offload_task(self):
        """UE卸载掉任务"""
        if self.task is None:
            print("the ue don't have a task")
            return False
        self.task = None
