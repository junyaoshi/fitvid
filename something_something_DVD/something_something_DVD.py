"""something_something_DVD dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import json
import cv2
import multiprocessing as mp
import math
from tqdm import tqdm
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def join(path1, path2):
    return path1 + '/' + path2


_DESCRIPTION = """
20BN-something-something Dataset V2.

Input is the first frame of the video and the task label.
Label is the last frame of the video.
"""

_CITATION = """
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense}, 
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DATA_DIR = '/home/junyao/Datasets/something_something'
_VIDEO_DIR = join(_DATA_DIR, 'something_something')
_IMAGE_DIR = join(_DATA_DIR, 'first_last_frames_64')
_TEMPLATES = {
    "Closing [something]": 0,
    "Moving [something] away from the camera": 1,
    "Moving [something] towards the camera": 2,
    "Opening [something]": 3,
    "Pushing [something] from left to right": 4,
    "Pushing [something] from right to left": 5,
    "Poking [something] so lightly that it doesn't or almost doesn't move": 6,
    "Moving [something] down": 7,
    "Moving [something] up": 8,
    "Pulling [something] from left to right": 9,
    "Pulling [something] from right to left": 10,
    "Pushing [something] with [something]": 11,
    "Moving [something] closer to [something]": 12,
    "Plugging [something] into [something]": 13,
    "Pushing [something] so that it slightly moves": 14
}


class SomethingSomethingDvd(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for something_something_DVD dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'first_frame': tfds.features.Image(shape=(64, 64, 3)),
                'label': tfds.features.Tensor(shape=(15,), dtype=tf.dtypes.float32),
                'last_frame': tfds.features.Image(shape=(64, 64, 3)),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('first_frame', 'last_frame'),  # Set to `None` to disable
            homepage='https://20bn.com/datasets/something-something',
            disable_shuffling=False,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # load json
        train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
        valid_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-validation.json')))

        train_dict, valid_dict = {}, {}
        for k in _TEMPLATES.keys():
            train_dict[k] = []
            valid_dict[k] = []
        for train_data in tqdm(train_list, desc='Parsing training json'):
            if train_data['template'] in _TEMPLATES.keys():
                train_dict[train_data['template']].append(train_data['id'])
        for valid_data in tqdm(valid_list, desc='Parsing validation json'):
            if valid_data['template'] in _TEMPLATES.keys():
                valid_dict[valid_data['template']].append(valid_data['id'])

        # print and visualize
        for k, v in train_dict.items():
            print(f'Train | {k} : {len(v)}')
        for k, v in valid_dict.items():
            print(f'Valid | {k} : {len(v)}')

        return {
            'train': self._generate_examples(train_dict),
            'valid': self._generate_examples(valid_dict),
        }

    def _generate_examples(self, dict):
        """Yields examples."""
        for k, v in dict.items():
            for id in v:
                first_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_first.jpg'))
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                last_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_last.jpg'))
                last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                label = tf.one_hot(_TEMPLATES[k], len(_TEMPLATES)).numpy()
                yield id, {
                    'first_frame': first_frame,
                    'label': label,
                    'last_frame': last_frame,
                }


# debugging
if __name__ == '__main__':
    # load json
    train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
    valid_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-validation.json')))

    # split generator
    train_dict, valid_dict = {}, {}
    for k in _TEMPLATES.keys():
        train_dict[k] = []
        valid_dict[k] = []
    for train_data in tqdm(train_list, desc='Parsing training json'):
        if train_data['template'] in _TEMPLATES.keys():
            train_dict[train_data['template']].append(train_data['id'])
    for valid_data in tqdm(valid_list, desc='Parsing validation json'):
        if valid_data['template'] in _TEMPLATES.keys():
            valid_dict[valid_data['template']].append(valid_data['id'])

    # print and visualize
    for k, v in train_dict.items():
        print(f'Train | {k} : {len(v)}')
    for k, v in valid_dict.items():
        print(f'Valid | {k} : {len(v)}')

    # generate examples
    for k, v in train_dict.items():
        for id in v:
            first_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_first.jpg'))
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            last_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_last.jpg'))
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            label = tf.one_hot(_TEMPLATES[k], len(_TEMPLATES)).numpy()
            break
        break
    print('Done')
