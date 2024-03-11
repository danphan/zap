from pathlib import Path

class Logger:
    def __init__(self, log_file : Path):
        self.log_file = Path(log_file)

    def log(self, metrics : dict):
        if not self.log_file.is_file():
            with open(self.log_file, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
        with open(self.log_file, "a") as f:
            f.write(",".join([f"{v:<4.4g}" for v in metrics.values()]) + "\n")
