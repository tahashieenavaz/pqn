class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target: float = 0.001):
        self.top = 1.0
        self.target = target
        self.ratio = ratio

    def get(self, frames: int, total_frames: int) -> float:
        decay_duration = total_frames * self.ratio
        if decay_duration == 0:
            return self.top
        return max(
            self.target, self.top - (frames / decay_duration) * (self.top - self.target)
        )
