class EarlyStopping:
    """
    Stops training when monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of evaluations without improvement before stopping.
    min_delta : float
        Minimum improvement required to reset patience.
    mode : str
        "max" (higher is better) or "min" (lower is better).
    """

    def __init__(self, patience=5, min_delta=0.01, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        improvement = (
            score - self.best_score
            if self.mode == "max"
            else self.best_score - score
        )

        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop
