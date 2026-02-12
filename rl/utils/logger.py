import logging
import os

class Logger:
    _logger = None

    @staticmethod
    def get_logger(log_path=None):
        if Logger._logger is not None:
            return Logger._logger

        logger = logging.getLogger("RL")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        )

        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path, mode="w")
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        Logger._logger = logger
        return logger
