import optax

from configuration.config_parser import Config, get_args
from configuration.logger_config import setup_logger
import jax
import flax as nnx

from data_imports import  get_data
from models import ModelFactory


def main():
    config_path = get_args()
    config = Config(config_path)

    # Logger Setup
    logger = setup_logger(config)
    logger.info(jax.devices())

    # Importing the data
    train, test, valid = get_data(config)

    rngs = nnx.Rngs(42)
    model_type = "DETR"  # Toggle to "EfficientDet" for the other side of the duel
    model = ModelFactory.create(model_type, num_classes=1000, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4))



if __name__ == '__main__':
    main()
