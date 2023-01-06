import math

import numpy as np


class Position:
    """在地图上的位置"""

    def __init__(self, x: float, y: float, z: float):
        self.data = np.empty((1, 3), np.float32)
        """位置的行向量"""
        self.data[0, 0] = x
        self.data[0, 1] = y
        self.data[0, 2] = z
        self.tail = self.data.copy()
        """历史位置组成的数组，尺寸为n*3"""

    def relative_position(self, other_position: 'Position'):
        return self.data - other_position.data

    def relative_horizontal_position(self, other_position: 'Position'):
        """返回list格式的水平相对位置"""
        return [self.data[0, 0] - other_position.data[0, 0], self.data[0, 1] - other_position.data[0, 1]]

    def distance(self, other_position: 'Position') -> float:
        """两个位置之间的距离"""
        return np.linalg.norm(self.data - other_position.data)

    def horizontal_distance(self, other_position: 'Position') -> float:
        """两个位置之间水平的距离"""
        return np.linalg.norm(self.data[0, 0:2] - other_position.data[0, 0:2])

    def vertical_distance(self, other_position: 'Position') -> float:
        """两个位置之间垂直的距离"""
        return np.linalg.norm(self.data[0, 2] - other_position.data[0, 2])

    def downcast(self, other_position: 'Position') -> float:
        """自身对other的下倾角弧度值"""
        dx = self.horizontal_distance(other_position)
        dy = self.vertical_distance(other_position)
        return math.atan2(dy, dx)

    def print(self):
        """打印位置"""
        print(self.data)

    def if_connect(self, other_position: 'Position', threshold: float) -> bool:
        """两个位置之间的距离是否超过阈值"""
        return self.distance(other_position) <= threshold

    def move(self, x_move: float, y_move: float, z_move=0):
        """位置的移动,dx,dx形式"""
        # 保存新的位置
        self.data[0, 0] += x_move
        self.data[0, 1] += y_move
        self.data[0, 2] += z_move
        # 将位置记录到历史位置中
        self.tail = np.vstack((self.tail, self.data))

    def move_by_radian(self, radian: float, distance: float):
        """位置的水平移动，弧度和距离形式"""
        self.move(math.cos(radian) * distance, math.sin(radian) * distance)


if __name__ == "__main__":
    point1 = Position(3, 4, 1.732 * 5)
    point1.print()
    point2 = Position(0, 0, 0)
    print(point1.distance(point2))
    print(point1.if_connect(point2, 1))
    print(point1.if_connect(point2, 5))
    print(point1.if_connect(point2, 8))
    print(point1.horizontal_distance(point2))
    print(point1.vertical_distance(point2))
    print(math.degrees(point1.downcast(point2)))
    point1.move(1, 2, 3)
    point1.move(1, 2, 3)
    print(point1.tail)
