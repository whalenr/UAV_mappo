from environment2.Constant import time_slice, ETUAV_height, ETUAV_speed
from environment2.Position import Position
from environment2.UAV import UAV, calcul_channel_gain
from environment2.UE import UE


class ETUAV(UAV):
    """给UE进行无线充电的UAV，energy transmission"""

    def __init__(self, position: Position):
        super().__init__(position, ETUAV_speed)
        self.charge_power = 100.0
        """无人机无线充电的功率(W)"""

    def charge_ue(self, ue: UE):
        """给单个UE充电(J)"""
        gain = calcul_channel_gain(self.position, ue.position)
        ue.charge(gain * self.charge_power * time_slice)

    def charge_all_ues(self, ues:[UE]):
        """"给所有UE充电"""
        for ue in ues:
            self.charge_ue(ue)

    def charge_energy(self, ue:UE):
        """返回充电量"""
        gain = calcul_channel_gain(self.position, ue.position)
        return gain * self.charge_power * time_slice



if __name__ == '__main__':
    etuav = ETUAV(Position(0,0,ETUAV_height))
    ue = UE(Position(0,50,0))
    print(etuav.charge_energy(ue))