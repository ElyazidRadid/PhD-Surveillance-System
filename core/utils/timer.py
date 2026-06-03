# utils/timer.py

import time


class Timer:
    def __init__(self):
        self.times = []

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self):
        self.times.append(time.perf_counter() - self.t0)

    def average(self):
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    def median(self):
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]

    def max(self):
        if not self.times:
            return 0.0
        return max(self.times)

    def min(self):
        if not self.times:
            return 0.0
        return min(self.times)