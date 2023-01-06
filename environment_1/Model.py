import numpy as np

from environment2.Position import Position
from environment2.UE import UE
from environment2.BS import BS
from environment2.UAV import UAV


class Model:
    """连续仿真的UAV场景下的AOI模型"""

    def __init__(self, num_UE:int, num_BS:int, num_UAV:int):
        self.num_UE = num_UE
        """用户(UE)数量"""
        self.num_BS = num_BS
        """基站(BS)数量"""
        self.num_UAV = num_UAV
        """UAV数量"""
        self.UE_set = [UE(Position(0, 0), 0) for i in range(self.num_UE)]
        self.BS_set = [BS(Position(0, 0), 5) for i in range(self.num_BS)]
        self.UAV_set = [UAV() for i in range(self.num_UAV)]
