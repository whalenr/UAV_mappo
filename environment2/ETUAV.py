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
        """给单个UE充电(J),返回充电电量(j)"""
        gain = calcul_channel_gain(self.position, ue.position)
        return ue.charge(gain * self.charge_power * time_slice)

    def charge_all_ues(self, ues: [UE]):
        """"给所有UE充电，并返回总充电量(j)"""
        sum_charge_energy = 0
        for ue in ues:
            sum_charge_energy += self.charge_ue(ue)

    def get_charge_energy(self, ue: UE):
        """返回充电量，并不实际充电"""
        gain = calcul_channel_gain(self.position, ue.position)
        energy = gain * self.charge_power * time_slice * ue.energy_conversion_efficiency
        """可以冲入的电量"""
        empty_energy = ue.get_energy_max() - ue.get_energy()
        """UE可以充入的电量空间"""
        return min(energy, empty_energy)


if __name__ == '__main__':
    etuav = ETUAV(Position(0, 0, ETUAV_height))
    ue1 = UE(Position(0, 50, 0))
    print(etuav.get_charge_energy(ue1))
