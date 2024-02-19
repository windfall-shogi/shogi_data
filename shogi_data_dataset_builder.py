"""shogi_data dataset."""
from pathlib import Path
from multiprocessing import Pool

import cshogi
import tensorflow_datasets as tfds
import numpy as np
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
from cshogi import HuffmanCodedPosAndEval, PackedSfenValue


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
            disable_shuffling=False
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(shogi_data): Downloads the data and defines the splits
        path = dl_manager.manual_dir

        # TODO(shogi_data): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'V7-1300hcpe-sub': self._generate_examples(path / 'V7-1300hcpe-sub'),
            # 'suisho4t_2m': self._generate_examples(path / 'suisho4t_2m'),
            # 'floodgate4200-validation': self._generate_examples(path / 'floodgate4200-validation'),
            'shogi_suisho5_depth9-validation': self._generate_examples_packed(path / 'shogi_suisho5_depth9-validation'),
            # 'shogi_suisho5_depth9': self._generate_examples_packed(path / 'shogi_suisho5_depth9')
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(shogi_data): Yields (key, example) tuples from the dataset
        for v in self._generate_example_common(path, HuffmanCodedPosAndEval):
            yield v

    def _generate_examples_packed(self, path):
        for v in self._generate_example_common(path, PackedSfenValue):
            yield v

    @staticmethod
    def _generate_example_common(path, data_type):
        if data_type == cshogi.PackedSfenValue:
            hcp, score, best_move16, game_result = 'sfen', 'score', 'move', 'game_result'
            ext = '.bin'
        else:
            hcp, score, best_move16, game_result = 'hcp', 'eval', 'bestMove16', 'gameResult'
            ext = '.hcpe'

        count = 1000000
        for f in path.glob(f'*{ext}'):
            tmp = Path(f)
            size_in_bytes = tmp.stat().st_size
            num_elements = size_in_bytes // data_type.itemsize

            # args = [(tmp, i, count, data_type) for i in range(0, num_elements, count)]
            # with Pool(8) as p:
            #     for result in p.imap_unordered(convert, args):
            #         for r in result:
            #             yield r
            for i in range(0, num_elements, count):
                data = np.fromfile(f, dtype=data_type, count=count, offset=i)
                for j, value in enumerate(data, start=i):
                    # noinspection PyTypeChecker
                    yield f'{tmp.stem}-{j:08x}', {
                        'hcp': value[hcp],
                        'eval': value[score],
                        'best_move16': value[best_move16],
                        'game_result': value[game_result]
                    }


def convert(args):
    path, i, count, data_type = args
    if data_type == cshogi.PackedSfenValue:
        hcp, score, best_move16, game_result = 'sfen', 'score', 'move', 'game_result'
    else:
        hcp, score, best_move16, game_result = 'hcp', 'eval', 'bestMove16', 'gameResult'

    data = np.fromfile(str(path), dtype=data_type, count=count, offset=i)
    return [(f'{path.stem}-{j:08x}', {
                'hcp': value[hcp],
                'eval': value[score],
                'best_move16': value[best_move16],
                'game_result': value[game_result]
            }) for j, value in enumerate(data, start=i)]