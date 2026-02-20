from configuration.config_parser import Config, get_args
from configuration.logger_config import setup_logger
import jax

from dataloader.data_imports import import_data


def main():
    config_path = get_args()
    config = Config(config_path)

    # Logger Setup
    logger = setup_logger(config)
    logger.info(jax.devices())

    # Importing the data
    train, test, valid = import_data(config)




if __name__ == '__main__':
    main()
