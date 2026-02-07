class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, monitor="val/f1"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = None
        self.counter = 0

    def __call__(self, value):
        if self.monitor == "val/loss":
            if self.best is None or value < self.best - self.min_delta:
                self.best = value
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if self.best is None or value > self.best + self.min_delta:
                self.best = value
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
