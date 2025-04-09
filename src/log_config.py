import logging


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s]-[%(name)s]-[%(levelname)s]-%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("train.log", mode="w"), logging.StreamHandler()],
    )


setup_logging()
