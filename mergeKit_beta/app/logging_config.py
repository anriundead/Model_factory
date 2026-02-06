import logging
import os


def app_log_dir(config):
    d = os.path.join(config.PROJECT_ROOT, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def setup_app_logger(config):
    log_dir = app_log_dir(config)
    log_file = os.path.join(log_dir, "app.log")
    logger = logging.getLogger("mergeKit_beta")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.info("应用日志文件: %s", os.path.abspath(log_file))
    return logger


def setup_logging(app, config):
    app.logger = setup_app_logger(config)
    return app.logger
