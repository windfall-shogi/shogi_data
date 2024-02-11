"""shogi_data dataset."""
from pathlib import Path

import cshogi
import tensorflow_datasets as tfds
import numpy as np
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
from cshogi import HuffmanCodedPosAndEval


class HCPEDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for shogi_data dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://example.org/login to get the data. Place the `data.zip`
    file in the `manual_dir/`.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(shogi_data): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'hcp': tfds.features.Tensor(shape=[32], dtype=tf.uint8),
                'eval': tfds.features.Scalar(dtype=tf.int16),
                'best_move16': tfds.features.Scalar(dtype=tf.uint8),
                'game_result': tfds.features.Scalar(dtype=tf.uint8)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('hcp', ('eval', 'best_move16', 'game_result')),
            homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(shogi_data): Downloads the data and defines the splits
        path = dl_manager.manual_dir

        # TODO(shogi_data): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'V7-1300hcpe': self._generate_examples(path / 'V7-1300hcpe')
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(shogi_data): Yields (key, example) tuples from the dataset
        count = 1000
        for f in path.glob('*.hcpe'):
            tmp = Path(f)
            size_in_bytes = tmp.stat().st_size
            num_elements = size_in_bytes // HuffmanCodedPosAndEval.itemsize
            for i in range(0, num_elements, count):
                data = np.fromfile(f, dtype=cshogi.HuffmanCodedPosAndEval, count=count, offset=i)
                for j, value in enumerate(data, start=i):
                    # noinspection PyTypeChecker
                    yield f'{tmp.stem}-{j:06d}', {
                        'hcp': value['hcp'],
                        'eval': value['eval'],
                        'best_move16': value['bestMove16'],
                        'game_result': value['gameResult']
                    }
