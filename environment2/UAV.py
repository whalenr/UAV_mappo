import math

import numpy as np
from environment2.Position import Position
import environment2.Constant


def calcul_Prob_LoS(radian: float) -> float:
    """计算给定弧度下LoS的概率"""
    theta = math.degrees(radian)
    C, D = 10, 0.6
    Prob_Los = 1 / (1 + C * math.exp(-D * (theta - C)))  # 城市环境中视距传输概率
    return Prob_Los


def calcul_Prob_hat(radian: float) -> float:
    """给定弧度下考虑到LOS信道的等效衰减系数"""
    chi = 0.2
    Prob_Los = calcul_Prob_LoS(radian)
    Prob_hat = Prob_Los + (1 - Prob_Los) * chi
    return Prob_hat


def calcul_channel_gain(position1: Position, position2: Position) -> float:
    """计算平均信道增益"""
    beta_0 = 1 * (10 ** (-3))
    alpha = 2
    radian = position1.downcast(position2)
    channel_gain = calcul_Prob_hat(radian) * (position1.distance(position2) ** (-alpha)) * beta_0
    return channel_gain


def calcul_SNR(p: float) -> float:
    sigma_2 = 1 * (10 ** (-13))  # W
    """计算信噪比"""
    return p / sigma_2


def power_by_speed(v: float) -> float:
    """不同速度下的功率,速度的单位为m/s,功率单位为W"""
    P0 = 0.012 / 8 * 1.225 * 0.05 * 0.503 * (300 ** 3) * (0.4 ** 3)
    U_tip = 120
    P1 = (1 + 0.1) * 20 ** (3 / 2) / math.sqrt(2 * 1.225 * 0.503)
    v0 = 4.03
    d0 = 0.6
    rho = 1.225
    s = 0.05
    A = 0.503
    P = P0 * (1 + (3 * v ** 2) / (U_tip ** 2)) + \
        P1 * (math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) ** 0.5 + \
        0.5 * d0 * rho * s * A * v ** 3
    return P


def energy_by_speed(speed: float) -> float:
    """一个时隙内在恒定速度(m/s)下的能耗，单位为J"""
    return power_by_speed(speed) * environment2.Constant.time_slice


class UAV:
    """UAV的基类"""

    def __init__(self, position: Position, speed_limit: float):
        self.position = position
        """UAV所在位置"""

        self.speed_limit = speed_limit
        """飞行最大速度，单位m/s"""

    def get_tail(self):
        """得到历史轨迹"""
        return self.position.tail

    def move_by_radian_rate(self, radian: float, rate: float):
        """无人机水平移动，rate参数为0到1之间的数,更新位置并返回功耗"""
        if not 0 <= rate <= 1:
            print("移动速度超出限制")
            return False
        # 更新位置
        self.position.move_by_radian(radian, rate * self.speed_limit * environment2.Constant.time_slice)
        return energy_by_speed(rate * self.speed_limit)

    def move_by_radian_rate_2(self, radian: float, rate: float):
        """无人机水平运动，radian和rate输入范围为-1到1"""
        new_radian = (radian + 1.0) / 2.0 * 2 * math.pi
        new_rate = (rate + 1.0) / 2.0
        return self.move_by_radian_rate(new_radian, new_rate)
    # def add_temp_energy(self, energy:float):
    #     """一个时隙的能耗先暂时存放在这，时隙结束后再执行update_energy将能耗加入历史总能耗中"""
    #     self.temp_energy += energy
    #
    # def update_energy(self):
    #     """在每个时隙的末尾执行，将temp_energy内的能量加入energy_comsumption中，并清0"""
    #     self.energy_consumption += self.temp_energy
    #     self.temp_energy = 0
    #
    # def get_temp_energy(self)->float:
    #     """返回这个时隙目前消耗的能量"""
    #     return self.temp_energy
