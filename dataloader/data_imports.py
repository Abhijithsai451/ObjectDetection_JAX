import grain
import numpy as np
import tensorflow_datasets as tfds

from configuration.config_parser import Config


def import_data(config):
    data_config = config.data
    train = (
            grain.MapDataset.source(tfds.data_source(data_config.get('dataset_name'), split= 'train[:80%]'))
            .seed(data_config.get('seed'))
            .shuffle()
            .to_iter_dataset()
        )
    test = (
            grain.MapDataset.source(tfds.data_source(data_config.get('dataset_name'), split= 'train[80%:90%]'))
            .seed(data_config.get('seed'))
            .shuffle()
            .to_iter_dataset()
        )
    val = (
            grain.MapDataset.source(tfds.data_source(data_config.get('dataset_name'), split= 'train[90%:]'))
            .seed(data_config.get('seed'))
            .shuffle()
            .to_iter_dataset()
        )
    steps = 4
    train_iter = iter(train)
    val_iter = iter(val)
    test_iter = iter(test)

    return train_iter, val_iter, test_iter


