"""shogi_data dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for shogi_data dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(shogi_data): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'hcp': tfds.features.Tensor(shape=[32], dtype='uint8'),
            'eval': tfds.features.Scalar(dtype='int16'),
            'best_move16': tfds.features.Scalar(dtype='uint16'),
            'game_result': tfds.features.Scalar(dtype='uint8'),
            'dummy': tfds.features.Scalar(dtype='uint8')
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('hcp', 'eval', 'best_move16', 'game_result'),
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(shogi_data): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(shogi_data): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(shogi_data): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
