
class Task:
    """任务"""
    def __init__(self, storage: float=5*(10**4), compute: float=5*(10**4)*200):
        self.storage = storage
        """任务的大小，bit数"""
        self.compute = compute
        """计算任务需要的CPU周期"""
        self.wating_time = 0
        """任务的等待时间"""

    def step(self):
        self.wating_time += 1
