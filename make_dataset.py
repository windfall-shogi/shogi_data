import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cshogi
import click
import sklearn.utils
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter


def convert_packed_sfen(begin, end, path, q):
    data = np.fromfile(path, dtype=cshogi.PackedSfenValue, count=end - begin, offset=begin)
    data = sklearn.utils.shuffle(data, random_state=begin)

    board = cshogi.Board()
    hcps = np.emath((), dtype=cshogi.HuffmanCodedPosAndEval)
    hcps['dummy'] = 0
    for d in data:
        board.set_psfen(d['sfen'])
        hcps['hcp'] = board.to_hcp()
        hcps['eval'] = d['score']
        hcps['bestMove16'] = d['move']
        hcps['gameResult'] = d['game_result']

        q.push(hcps.tobytes())


@click.command()
@click.option('--input-path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path))
@click.option('--index', type=int)
@click.option('--num-threads', type=int, default=8)
def cmd(input_path, output_dir, index, num_threads):
    with mp.Manager() as manager:
        q = manager.Queue(maxsize=10000)

        section_count = 1024000000
        count_per_section = 160000
        range_list = []
        for i in range(64):
            begin = section_count * i + index * count_per_section
            end = begin + count_per_section
            range_list.append((begin, end))
        range_list = sklearn.utils.shuffle(range_list, random_state=index)

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            features = [executor.submit(convert_packed_sfen, begin, end, input_path, q) 
                        for begin, end in range_list]
            writer = ArrayRecordWriter(str(output_dir / 'data.array_record'), 'group_size:1')
            while True:
                while len(q):
                    data = q.get()
                    writer.write(data)
                if all([f.done() for f in features]):
                    while len(q):
                        data = q.get()
                        writer.write(data)
                    break
            writer.close()
    pass


if __name__ == '__main__':
    cmd()
