import logging
import os

class Logger:
    _logger = None

    @staticmethod
    def get_logger(log_path="logs/run.log"):
        if Logger._logger is not None:
            return Logger._logger

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logger = logging.getLogger("RL")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()   

        fh = logging.FileHandler(log_path, mode="w")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        Logger._logger = logger
        return logger
