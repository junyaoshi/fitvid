"""something_something_DVD_subgoal dataset."""

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

Input is the kth frame of the video and the task label.
Label is the mth frame and nth frame of the video.
k = (m + n)/2
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
_SUBSAMPLING_RATE = 4
_IMAGE_DIR = join(_DATA_DIR, f'subgoal_frames_subsample{_SUBSAMPLING_RATE}_64')


def single_process_conversion(video_names):
    n_skip_frames = _SUBSAMPLING_RATE - 1
    for video_name in (video_names):
        vc = cv2.VideoCapture(join(_VIDEO_DIR, video_name))
        frames = {}
        skips = n_skip_frames
        frame_idx = 0  # count number of frames
        while True:
            ret, frame = vc.read()
            if ret:
                if skips == n_skip_frames:
                    frames[frame_idx] = frame
                    skips = 0
                else:
                    skips += 1
                frame_idx += 1
            else:
                break
        vc.release()
        video_frames_dir = join(_IMAGE_DIR, video_name[:-5])
        tf.io.gfile.makedirs(video_frames_dir)
        for idx, frame in frames.items():
            frame_resized = cv2.resize(frame, (64, 64),
                                       interpolation=cv2.INTER_CUBIC)
            idx_0000_format = str(10000 + idx)[-4:]
            cv2.imwrite(join(video_frames_dir, f'{video_name[:-5]}_{idx_0000_format}.jpg'),
                        frame_resized)


def convert_videos_to_images(all_video_names):
    if tf.io.gfile.exists(_IMAGE_DIR):
        print(f'Image directory already exists. Skip video to image conversion...')
        return

    tf.io.gfile.mkdir(_IMAGE_DIR)
    # all_video_names = tf.io.gfile.listdir(_VIDEO_DIR)
    num_videos = len(all_video_names)
    num_cpus = mp.cpu_count()

    # split into multiple jobs
    splits = list(range(0, num_videos, math.ceil(num_videos / num_cpus)))
    splits.append(num_videos)
    args_list = [all_video_names[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

    # # debug
    # single_process_conversion(args_list[0])

    # multiprocessing (num_cpus processes)
    pool = mp.Pool(num_cpus)
    with pool as p:
        r = list(tqdm(p.imap(single_process_conversion, args_list), total=num_cpus))


class SomethingSomethingDvdSubgoal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for something_something_DVD_subgoal dataset."""

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
                'current_frame': tfds.features.Image(shape=(64, 64, 3)),
                'label': tfds.features.Tensor(shape=(15,), dtype=tf.dtypes.float32),
                'goal_frame': tfds.features.Image(shape=(64, 64, 3)),
                'subgoal_frame': tfds.features.Image(shape=(64, 64, 3)),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            disable_shuffling=False,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # load json
        train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
        valid_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-validation.json')))

        # split generator
        train_dict, valid_dict = {}, {}
        all_video_names = []
        for k in _TEMPLATES.keys():
            train_dict[k] = []
            valid_dict[k] = []
        for train_data in tqdm(train_list, desc='Parsing training json'):
            if train_data['template'] in _TEMPLATES.keys():
                train_dict[train_data['template']].append(train_data['id'])
                all_video_names.append(train_data['id'] + '.webm')
        for valid_data in tqdm(valid_list, desc='Parsing validation json'):
            if valid_data['template'] in _TEMPLATES.keys():
                valid_dict[valid_data['template']].append(valid_data['id'])
                all_video_names.append(valid_data['id'] + '.webm')

        # print and visualize
        for k, v in train_dict.items():
            print(f'Train | {k} : {len(v)}')
        for k, v in valid_dict.items():
            print(f'Valid | {k} : {len(v)}')
        print(f'Total number of videos: {len(all_video_names)}')

        # convert videos to images
        convert_videos_to_images(all_video_names)
        return {
            'train': self._generate_examples(train_dict),
            'valid': self._generate_examples(valid_dict),
        }

    def _generate_examples(self, dict):
        """Yields examples."""
        # generate examples
        for k, v in tqdm(dict.items()):
            for id in v:
                video_frames_dir = join(_IMAGE_DIR, id)
                frame_list = tf.io.gfile.listdir(video_frames_dir)
                frame_list = sorted(frame_list)
                video_len = len(frame_list)
                for current_idx, current_fname in enumerate(frame_list):
                    for subgoal_idx in range(current_idx + 1, video_len):
                        goal_idx = subgoal_idx + (subgoal_idx - current_idx)
                        if goal_idx >= video_len:
                            break
                        subgoal_fname = frame_list[subgoal_idx]
                        goal_fname = frame_list[goal_idx]
                        current_frame = cv2.imread(join(video_frames_dir, current_fname))
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        subgoal_frame = cv2.imread(join(video_frames_dir, subgoal_fname))
                        subgoal_frame = cv2.cvtColor(subgoal_frame, cv2.COLOR_BGR2RGB)
                        goal_frame = cv2.imread(join(video_frames_dir, goal_fname))
                        goal_frame = cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB)
                        label = tf.one_hot(_TEMPLATES[k], len(_TEMPLATES)).numpy()
                        sample_id = f'{id}_{current_idx}_{subgoal_idx}_{goal_idx}'
                        yield sample_id, {
                            'current_frame': current_frame,
                            'goal_frame': goal_frame,
                            'label': label,
                            'subgoal_frame': subgoal_frame
                        }


# debugging
if __name__ == '__main__':
    # load json
    train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
    valid_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-validation.json')))

    # split generator
    train_dict, valid_dict = {}, {}
    all_video_names = []
    for k in _TEMPLATES.keys():
        train_dict[k] = []
        valid_dict[k] = []
    for train_data in tqdm(train_list, desc='Parsing training json'):
        if train_data['template'] in _TEMPLATES.keys():
            train_dict[train_data['template']].append(train_data['id'])
            all_video_names.append(train_data['id'] + '.webm')
    for valid_data in tqdm(valid_list, desc='Parsing validation json'):
        if valid_data['template'] in _TEMPLATES.keys():
            valid_dict[valid_data['template']].append(valid_data['id'])
            all_video_names.append(valid_data['id'] + '.webm')

    # print and visualize
    for k, v in train_dict.items():
        print(f'Train | {k} : {len(v)}')
    for k, v in valid_dict.items():
        print(f'Valid | {k} : {len(v)}')
    print(f'Total number of videos: {len(all_video_names)}')

    # convert videos to images
    convert_videos_to_images(all_video_names)

    # generate examples
    n_data = 0
    for k, v in train_dict.items():
        for id in v:
            video_frames_dir = join(_IMAGE_DIR, id)
            frame_list = tf.io.gfile.listdir(video_frames_dir)
            frame_list = sorted(frame_list)
            video_len = len(frame_list)
            for current_idx, current_fname in enumerate(frame_list):
                for subgoal_idx in range(current_idx + 1, video_len):
                    goal_idx = subgoal_idx + (subgoal_idx - current_idx)
                    if goal_idx >= video_len:
                        break
                    subgoal_fname = frame_list[subgoal_idx]
                    goal_fname = frame_list[goal_idx]
                    current_frame = cv2.imread(join(video_frames_dir, current_fname))
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    subgoal_frame = cv2.imread(join(video_frames_dir, subgoal_fname))
                    subgoal_frame = cv2.cvtColor(subgoal_frame, cv2.COLOR_BGR2RGB)
                    goal_frame = cv2.imread(join(video_frames_dir, goal_fname))
                    goal_frame = cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB)
                    label = tf.one_hot(_TEMPLATES[k], len(_TEMPLATES)).numpy()
                    n_data += 1
            break
        break
    print(f'generated {n_data} samples')
    print('Done')
