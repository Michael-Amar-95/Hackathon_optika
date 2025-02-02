import logging
import sys


# TODO: Logs are duplicated in jupyter notebooks
def setup_logging() -> None:
    root_logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s:%(levelname)-5s: %(message)s',
        stream=sys.stdout,
    )
    stderr_handler = root_logger.handlers[0]  # get default handler
    stderr_handler.setLevel(logging.WARNING)  # only print WARNING+
    # add another StreamHandler to stdout which only emits below WARNING
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda rec: rec.levelno < logging.WARNING)
    stdout_handler.setFormatter(stderr_handler.formatter)  # reuse the stderr formatter
    root_logger.addHandler(stdout_handler)
