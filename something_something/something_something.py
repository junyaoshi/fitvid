"""something_something dataset."""

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

Input is the first frame of the video.
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


def single_process_conversion(video_names):
    for video_name in (video_names):
        vc = cv2.VideoCapture(join(_VIDEO_DIR, video_name))
        first_frame, last_frame = None, None
        i = 0
        while True:
            ret, frame = vc.read()
            if first_frame is None:
                first_frame = frame
            if ret:
                i += 1
                last_frame = frame
            else:
                break
        vc.release()
        first_frame_resized = cv2.resize(first_frame, (64, 64),
                                         interpolation=cv2.INTER_CUBIC)
        last_frame_resized = cv2.resize(last_frame, (64, 64),
                                        interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(join(_IMAGE_DIR, f'{video_name[:-5]}_first.jpg'),
                    first_frame_resized)
        cv2.imwrite(join(_IMAGE_DIR, f'{video_name[:-5]}_last.jpg'),
                    last_frame_resized)


def convert_videos_to_images():
    if tf.io.gfile.exists(_IMAGE_DIR):
        print(f'Image directory already exists. Skip video to image conversion...')
        return

    tf.io.gfile.mkdir(_IMAGE_DIR)
    all_video_names = tf.io.gfile.listdir(_VIDEO_DIR)
    num_videos = len(all_video_names)
    num_cpus = mp.cpu_count()

    # split into multiple jobs
    splits = list(range(0, num_videos, math.ceil(num_videos / num_cpus)))
    splits.append(num_videos)
    args_list = [all_video_names[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

    # multiprocessing (num_cpus processes)
    pool = mp.Pool(num_cpus)
    with pool as p:
        r = list(tqdm(p.imap(single_process_conversion, args_list), total=num_cpus))


class SomethingSomething(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for something_something dataset."""

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
        convert_videos_to_images()
        videos = tf.io.gfile.listdir(join(_DATA_DIR, 'something_something'))
        train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
        train_videos = [d["id"] + '.webm' for d in train_list]
        valid_videos = list(set(videos) - set(train_videos))
        train_ids = [int(vid_name[:-5]) for vid_name in train_videos]
        valid_ids = [int(vid_name[:-5]) for vid_name in valid_videos]

        return {
            'train': self._generate_examples(train_ids),
            'valid': self._generate_examples(valid_ids)
        }

    def _generate_examples(self, ids):
        """Yields examples."""
        for id in ids:
            first_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_first.jpg'))
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            last_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_last.jpg'))
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            yield id, {
                'first_frame': first_frame,
                'last_frame': last_frame,
            }


# debugging
if __name__ == '__main__':
    # vid2image
    convert_videos_to_images()

    # split generator
    videos = tf.io.gfile.listdir(join(_DATA_DIR, 'something_something'))
    train_list = json.load(tf.io.gfile.GFile(join(_DATA_DIR, 'something-something-v2-train.json')))
    train_videos = [d["id"] + '.webm' for d in train_list]
    valid_videos = list(set(videos) - set(train_videos))
    train_ids = [int(vid_name[:-5]) for vid_name in train_videos]
    valid_ids = [int(vid_name[:-5]) for vid_name in valid_videos]

    # generate examples
    id = train_ids[4396]
    first_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_first.jpg'))
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    last_frame = cv2.imread(join(_IMAGE_DIR, f'{id}_last.jpg'))
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    print('Done')
