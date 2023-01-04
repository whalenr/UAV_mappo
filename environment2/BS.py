import math

import numpy as np
import Position
from environment2.DPUAV import DPUAV
from environment2.UAV import calcul_SNR


class BS:
    def __init__(self, position: Position):
        self.position = position
        """BS所在位置"""

        # self.B_UAV = 1.0
        # """与UAV之间的传输带宽"""
        # self.self.computing_capacity = 25*(10**7)
        # """BS的计算能力，单位为cpu cycle/时间间隔"""

