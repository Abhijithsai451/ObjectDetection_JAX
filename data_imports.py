from random import seed

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from configuration.config_parser import Config


def get_data(config: Config):
    data_config = config.data
    seed = data_config.get('seed')
    batch_size = data_config.get('batch_size')
    image_shape = (data_config.get('image_height'), data_config.get('image_width'), 3)
    ds_kwargs = {"name": data_config.get('dataset_name'), "data_dir": data_config.get('data_dir')}
    num_classes = data_config.get('num_classes')

    builder = tfds.builder(**ds_kwargs)
    builder.download_and_prepare(file_format="array_record")

    train_source = tfds.data_source(**ds_kwargs, split= 'test[:80%]')
    test_source = tfds.data_source(**ds_kwargs ,split= 'test[80%:90%]')
    valid_source = tfds.data_source(**ds_kwargs,split= 'test[90%:]')

    def transformation_fn(features):
        image = features['image']
        image = jax.image.resize(image, image_shape, method='bilinear')
        if jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5):
            image = jnp.fliplr(image)
        return {
            'image': image.astype("float32")/255.0,
            'label': jax.nn.one_hot(features['label'], 10)
        }

    def apply_mixup(batch):
        images, labels = batch["image"], batch["label"]
        n = images.shape[0]
        # Sample lambda from Beta(0.2, 0.2)
        lam = jax.random.beta(jax.random.PRNGKey(seed), 0.2, 0.2)

        # Shuffle the batch to pick "Partners" for mixing
        indices = jax.random.permutation(jax.random.PRNGKey(seed), n)

        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]

        return {"image": mixed_images, "label": mixed_labels}



    def build_loader(source, is_training:bool):
        sampler = grain.IndexSampler(
            num_records = len(source),
            num_epochs = None if is_training else 1,
            shard_options = grain.ShardOptions(
                shard_index = jax.process_index(),
                shard_count = jax.process_count(),
                drop_remainder = is_training
            ),
            shuffle = is_training,
            seed = seed
        )
        operations = [
                grain.MapOperation(transformation_fn),
                grain.BatchOperation(batch_size = batch_size, drop_remainder = is_training)
            ]
        if is_training:
            operations.append(grain.MapOperation(apply_mixup))

        data = grain.DataLoader(
            data_source = source,
            sampler = sampler,
            operations =operations ,
            worker_count= 8
        )
        return data

    return {
        'train': build_loader(train_source, True),
        'valid': build_loader(valid_source, False),
        'test': build_loader(test_source, False)
    }