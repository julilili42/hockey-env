import os
from utils.logger import Logger


class ModelManager:
    def __init__(self, model_dir, metric_name="winrate", min_delta=0.01):
        self.model_dir = model_dir
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.best_score = float("-inf")

        self.logger = Logger.get_logger()
        os.makedirs(self.model_dir, exist_ok=True)

    def update(self, agent, score, episode):
        """
        Check if score improved and save model if necessary.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score

            save_path = os.path.join(self.model_dir, "td3_best.pt")
            agent.save(save_path)


            self.logger.info(
                f"New best {self.metric_name}: "
                f"{score:.3f} at episode {episode}"
            )

            return True

        return False
