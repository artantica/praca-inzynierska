import sys
import os
import shutil
import json
import logging
import argparse
import time
import tqdm
import numpy
import cv2
import re

from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
sys.path.append('faceswap')

from lib.cli import args
from tools.alignments.cli import AlignmentsArgs
from lib.config import generate_configs
from lib.utils import get_backend, get_folder
from plugins.plugin_loader import PluginLoader
from lib.cli.actions import FilesFullPaths
from lib.cli.launcher import ScriptExecutor
from tools.alignments.media import AlignmentData
from tools.alignments.jobs import Merge
from lib.convert import Converter
from lib.face_filter import FaceFilter

sources = dict()
script = None

_DATA_DIRECTORY = "data"
_MODEL_DIRECTORY = 'models'
_FILTER_IMAGES = 'filter_images'
_PROCESSED_DIRECTORY = 'processed'
_VIDEOS = 'videos'
_IMAGES = 'images'
_FRAMES = 'frames'
_FACES = 'faces'
_ALIGNMENTS_FILE = "alignments.fsa"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_sources_info():
    get_people()

def get_people():
    rootdir = get_folder(_DATA_DIRECTORY)
    sources['people'] = ([name for name in os.listdir(rootdir) if not name.startswith(".")])

def count_number_of_matching_files(directory, regex):
    files = os.listdir(directory)
    count = len([file for file in files if re.match(regex, file)])
    return count

get_sources_info()


class FaceSwapInterface:
    def __init__(self):
        self._parser = args.FullHelpArgumentParser()
        generate_configs()
        self._subparser = self._parser.add_subparsers()

    def extract(self, input_dir, output_dir, alignments, arguments):
        """
        Run extraction from faceswap.py.

        :param input_dir: str
        Input directory with images to extract.

        :param output_dir: str
        Output directory, where results will be saved.

        :param alignments: str
        Path for alignments file.

        :param arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments

        """
        args.ExtractArgs(self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = f"extract --input-dir {input_dir} --output-dir {output_dir} --detector {arguments.detector} " \
                   f"--aligner {arguments.aligner}"
        if alignments:
            args_str += f" -al {alignments}"
        self._run_script(args_str)

    def extract_for_conversion(self, input_dir, output_dir, alignments, arguments):
        """
        Run extraction before conversion from faceswap.py.

        :param input_dir: str
        Input directory with images to extract.

        :param output_dir: str
        Output directory, where results will be saved.

        :param alignments: str
        Path for alignments file.

        :param arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments.
        """
        args.ExtractArgs(self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = f"extract --input-dir {input_dir} --output-dir {output_dir} --detector {arguments.detector} " \
                   f"--aligner {arguments.aligner} -al {alignments} -ssf"
        self._run_script(args_str)

    def train(self, input_a_dir, input_b_dir, arguments):
        """
        Run training from faceswap.py.

        :param input_a_dir: str
        Input directory with faces from person A.

        :param input_b_dir: str
        nput directory with faces from person B.

        :param arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments.
        """
        args.TrainArgs(
            self._subparser, "train", "This command trains the model for the two faces A and B.")
        args_str = f"train --input-A {input_a_dir} --input-B {input_b_dir} --model-dir {arguments.model_dir} " \
                   f"--trainer {arguments.trainer} --batch-size {arguments.batch_size} --snapshot-interval {arguments.snapshot_interval}"
        self._run_script(args_str)

    def convert(self, arguments, alignments):
        """
        Run conversion from faceswap.py.

        :param arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments.
        """
        args.ConvertArgs(
            self._subparser, "convert", "This command trains the model for the two faces A and B.")

        args_str = f"convert --input-dir {arguments.input_path} --output-dir {arguments.output_path}" \
                   f" --model-dir {arguments.model_dir} --trainer {arguments.trainer} --alignments {alignments} " \
                   f"-otf -w ffmpeg"
        self._run_script(args_str)

    def convert_live(self, arguments):
        """
        Run live conversion from faceswap.py.

        :param arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments.
        """
        args.ConvertLiveArgs(
            self._subparser, "convert_live", "This command convert live stream and swap faces A for B.")
        args_str = f"convert_live --model-dir {arguments.model_dir} --detector {arguments.detector} " \
                   f"--aligner {arguments.aligner}"
        if arguments.output_path:
            args_str += f" --output-dir {arguments.output_path}"
        self._run_script(args_str)

    def _run_script(self, args_str):
        """
        Run proper script

        :param args_str: str
        Arguments for function to run.
        """
        args = self._parser.parse_args(args_str.split(' '))
        args.func(args)

_faceswap = FaceSwapInterface()

class Video:
    def __init__(self, video_name, extension,  video_path, frames_directory, faces_directory):
        self._video_name = video_name
        self._extension = extension
        self._video_path = video_path
        self._frames_folder = frames_directory
        self._faces_folder = faces_directory

        self._alignments = os.path.join(self._faces_folder, self._video_name + "_" + _ALIGNMENTS_FILE)

    @property
    def video_name(self):
        return self._video_name

    @property
    def video_path(self):
        return self._video_path

    @property
    def alignments_path(self):
        return self._alignments

    @property
    def faces_folder(self):
        return self._faces_folder

    @property
    def frames_folder(self):
        return self._frames_folder


class Alignments:
    def __init__(self, faces_dir, output, alignments_files):
        self.faces_dir = faces_dir
        self.output = output
        self.alignments_files = alignments_files


class Person:
    def __init__(self, name):
        # self._arguments = arguments
        self.name = name

        self.videos = []
        self.logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
        self._check_if_exist()
        self._get_person_directories()
        self._get_videos()
        self.number_of_videos = len(self.videos)
        self.number_of_faces = count_number_of_matching_files(directory=self.faces_directory, regex=r"([a-zA-Z0-9])+\_frame\_[0-9]{4}\_[0-9]{1}\.jpg")

    def _get_person_directories(self):
        path_person = os.path.join(_DATA_DIRECTORY, self.name)
        self.person_directory = get_folder(path=path_person)

        path_videos = os.path.join(_DATA_DIRECTORY, self.name, _VIDEOS)
        self.videos_directory = get_folder(path=path_videos)

        path_filter_images = os.path.join(_DATA_DIRECTORY, self.name, _FILTER_IMAGES)
        self.filter_images_directory = get_folder(path=path_filter_images)

        path_frames = os.path.join(_DATA_DIRECTORY, self.name, _FRAMES)
        self.frames_directory = get_folder(path=path_frames)

        path_faces = os.path.join(_DATA_DIRECTORY, self.name, _FACES)
        self.faces_directory = get_folder(path=path_faces)

    def _check_if_exist(self):
        if self.name in sources['people']:
            self.logger.debug(f"Person {self.name} exist in your library.")
        else:
            self.logger.debug(f"Person {self.name} does not exist in your library.")

    def print_info(self):
        print(self.name)
        print(f"Number of videos: {self.number_of_videos}")
        print(f"Number of faces: {self.number_of_faces}")
        print()

    def _get_videos(self):
        for video_filename in os.listdir(self.videos_directory):
            if os.path.splitext(video_filename)[1] in [".mp4", ".mov"]:
                video_filepath = os.path.join(self.videos_directory, video_filename)
                video_name, extension = os.path.splitext(video_filename)
                video = Video(video_name=video_name, extension=extension, video_path=video_filepath, frames_directory=self.frames_directory, faces_directory=self.faces_directory)
                self.videos.append(video)

    def add_video(self, filepath):
        extension = os.path.splitext(filepath)[1]
        video_name = self.name + str(self.number_of_videos)
        new_video_path = os.path.join(self.videos_directory, video_name + extension)
        shutil.copy(filepath, new_video_path)

        self.number_of_videos += 1

        if video_name not in [video.video_name for video in self.videos]:
            video = Video(video_name=video_name, extension=extension, video_path=new_video_path, frames_directory=self.frames_directory, faces_directory=self.faces_directory)
            self.videos.append(video)
            self._extract_frames(video)
            self._extract_faces(video)

    def add_filter_image(self, filepath):
        shutil.copy(filepath, self.filter_images_directory)

    def _extract_frames(self, video):
        video_clip = VideoFileClip(video.video_path)

        start_time = time.time()
        logger.info('About to extract_frames for {}, fps {}, length {}s'.format(video.video_name,
                                                                                           video_clip.fps,
                                                                                           video_clip.duration))
        frame_number = 0
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video_clip.fps), total=video_clip.fps * video_clip.duration):
            video_frame_file = os.path.join(video.frames_folder, f'{video.video_name}_frame_{frame_number:04d}.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_number += 1

        logger.info(f'Finished extract_frames for {video.frames_folder}, total frames {frame_number - 1}, time taken {time.time() - start_time:.0f}s')

    def _extract_faces(self, video):
        _faceswap.extract(input_dir=self.frames_directory, output_dir=self.faces_directory,
                          alignments=video.alignments_path, arguments=arguments)

    def merge_alignments(self):
        alignments_files = [video.alignments_path for video in self.videos]
        alignment_file_path = os.path.join(self.faces_directory, _ALIGNMENTS_FILE)

        if len(alignments_files) == 1:
            shutil.copy(alignments_files[0], alignment_file_path)
            return
        elif len(alignments_files) < 1:
            logger.error(f"Missing alignment file. Extract again.")
            exit()
        alignments = self._load_alignments(alignments_files)

        args = Alignments(faces_dir=self.faces_directory, output=alignment_file_path, alignments_files=alignments_files)
        m = Merge(alignments, args)
        m.process()

    def _load_alignments(self, alignments_file):
        """ Loads the given alignments file(s) prior to running the selected job.

        Returns
        -------
        :class:`~tools.alignments.media.AlignmentData` or list
            The alignments data formatted for use by the alignments tool. If multiple alignments
            files have been selected, then this will be a list of
            :class:`~tools.alignments.media.AlignmentData` objects
        """
        if len(alignments_file) <= 1:
            self.logger.error("More than one alignments file required for merging")
            return

        retval = [AlignmentData(a_file) for a_file in alignments_file]
        self.logger.debug("Alignments: %s", retval)
        return retval


def get_all_info():
    get_sources_info()
    # print(sources)
    for person_name in sources['people']:
        person = Person(person_name)
        person.print_info()

def _conv_extract_suparser(subparser):
    if get_backend() == "cpu":
        default_detector = default_aligner = "cv2-dnn"
    else:
        default_detector = "s3fd"
        default_aligner = "fan"

    subparser.add_argument('-D', '--detector', dest='detector', type=str.lower, default=default_detector,
                           choices=PluginLoader.get_available_extractors("detect"),
                           help="R|Detector to use. Some of these have configurable settings in "
                                "'/config/extract.ini' or 'Settings > Configure Extract 'Plugins':"
                                "\nL|cv2-dnn: A CPU only extractor which is the least reliable and least "
                                "resource intensive. Use this if not using a GPU and time is important."
                                "\nL|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources "
                                "than other GPU detectors but can often return more false positives."
                                "\nL|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and "
                                "fewer false positives than other GPU detectors, but is a lot more resource "
                                "intensive.")
    subparser.add_argument("-A", "--aligner", type=str.lower, default=default_aligner,
                           choices=PluginLoader.get_available_extractors("align"),
                           help="R|Aligner to use."
                                "\nL|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, "
                                "but less accurate. Only use this if not using a GPU and time is important."
                                "\nL|fan: Best aligner. Fast on GPU, slow on CPU.")
    return subparser


def _set_extract_subparser(subparser):
    subparser.add_argument('-i', '--input-path', dest='input_path',
                           help="Input directory or video. Either a directory containing the image files you " +
                                "wish to process or path to a video file. NB: This should be the source video/" +
                                "frames NOT the source faces')")
    subparser.add_argument('-p', '--person', type=str, dest='name',
                           help='Name of a person, you wish to process. There will be folder with this name created in data/ '
                                'to store all important files within it.')

    return _conv_extract_suparser(subparser)


def _set_train_subparser(subparser):
    subparser.add_argument('-A', '--person-A', type=str, dest='person_A', required=True,
                           help='Name of a person A. This is the original face, i.e. the face that you want to remove '
                                'and replace with face B. This person must have extracted faces to proceed.')
    subparser.add_argument('-B', '--person-B', type=str, dest='person_B', required=True,
                           help='Name of a person B. This is the swap face, i.e. the face that you want to place'
                                'onto the head of person A. This person must have extracted faces to proceed.')
    subparser.add_argument('-m', '--model', dest="model_dir", required=True,
                           help="Model directory. This is where the training data will be stored. You should "
                                "always specify a new folder for new models. If starting a new model, select "
                                "either an empty folder, or a folder which does not exist (which will be "
                                "created). If continuing to train an existing model, specify the location of "
                                "the existing model."
                           )
    subparser.add_argument('-t', '--trainer', type=str.lower, default='original', choices=['original', 'realface', 'villain', 'dfl-sae'],
                           help="R|Select which trainer to use. Trainers can be configured from the Settings "
                                "menu or the config folder."
                                "\nL|original: The original model created by /u/deepfakes."
                                "\nL|dfl-sae: Adaptable model from deepfacelab"
                                "\nL|realface: A high detail, dual density model based on DFaker, with "
                                "customizable in/out resolution. The autoencoders are unbalanced so B>A swaps "
                                "won't work so well. By andenixa et al. Very configurable."
                                "\nL|villain: 128px in/out model from villainguy. Very resource hungry (You "
                                "will require a GPU with a fair amount of VRAM). Good for details, but more "
                                "susceptible to color differences."
                           )
    subparser.add_argument("-bs", "--batch-size", type=int, dest="batch_size", default=16,
                           help="Batch size. This is the number of images processed through the model for each "
                                "side per iteration. NB: As the model is fed 2 sides at a time, the actual "
                                "number of images within the model at any one time is double the number that you "
                                "set here. Larger batches require more GPU RAM."
                           )
    subparser.add_argument("-it", "--iterations", type=int, default=1000000,
                           help="Length of training in iterations. This is only really used for automation. "
                                "There is no 'correct' number of iterations a model should be trained for. "
                                "You should stop training when you are happy with the previews. However, if "
                                "you want the model to stop automatically at a set number of iterations, you "
                                "can set that value here."
                           )
    subparser.add_argument("-ss", "--snapshot-interval", type=int, dest="snapshot_interval", default=25000,
                           help="Sets the number of iterations before saving a backup snapshot of the model "
                                "in it's current state. Set to 0 for off."
                           )
    return subparser

def _set_convert_subparser(subparser):
    subparser.add_argument('-i', '--input-path', dest='input_path',
                           help="Input directory or video. Either a directory containing the image files you " +
                                "wish to process or path to a video file"
                                ". NB: This should be the source video/" +
                                "frames NOT the source faces')")
    subparser.add_argument('-m', '--model', dest="model_dir", required=True,
                           help="Model directory. This is where the training data will be stored. You should "
                                "always specify a new folder for new models. If starting a new model, select "
                                "either an empty folder, or a folder which does not exist (which will be "
                                "created). If continuing to train an existing model, specify the location of "
                                "the existing model."
                           )
    subparser.add_argument('-t', '--trainer', type=str.lower, default='original',
                           choices=['original', 'realface', 'villain', 'dfl-sae'],
                           help="R|Select which trainer to use. Trainers can be configured from the Settings "
                                "menu or the config folder."
                                "\nL|original: The original model created by /u/deepfakes."
                                "\nL|dfl-sae: Adaptable model from deepfacelab"
                                "\nL|realface: A high detail, dual density model based on DFaker, with "
                                "customizable in/out resolution. The autoencoders are unbalanced so B>A swaps "
                                "won't work so well. By andenixa et al. Very configurable."
                                "\nL|villain: 128px in/out model from villainguy. Very resource hungry (You "
                                "will require a GPU with a fair amount of VRAM). Good for details, but more "
                                "susceptible to color differences."
                           )
    subparser.add_argument('-o', '--output-path', dest='output_path', required=True,
                           help="Output directory. This is where the converted file will be saved.")

    return _conv_extract_suparser(subparser)


def _set_convert_live_subparser(subparser):
    subparser.add_argument('-m', '--model', dest="model_dir", required=True,
                           help="Model directory. This is where the training data will be stored. You should "
                                "always specify a new folder for new models. If starting a new model, select "
                                "either an empty folder, or a folder which does not exist (which will be "
                                "created). If continuing to train an existing model, specify the location of "
                                "the existing model."
                           )
    subparser.add_argument('-o', '--output-path', dest='output_path', default=None,
                           help="Optional output directory. This is where the converted files will be saved.")

    return _conv_extract_suparser(subparser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # EXTRACT
    extract = subparsers.add_parser('extract', help="Extract the faces from pictures")
    _set_extract_subparser(extract)

    # TRAIN
    train = subparsers.add_parser('train', help="This command trains the model for the two faces A and B")
    _set_train_subparser(train)

    # CONVERT
    convert = subparsers.add_parser('convert', help="Convert a source image to a new one with the face swapped")
    _set_convert_subparser(convert)

    # CONVERT LIVE
    convert_live = subparsers.add_parser('convert_live', help="Convert a live stream to a new one with the face swapped")
    _set_convert_live_subparser(convert_live)

    arguments = parser.parse_args()
    if hasattr(arguments, "command") and arguments.command:
        script = ScriptExecutor(arguments.command)

    if arguments.command == 'extract':
        person = Person(arguments.name)
        person.add_video(arguments.input_path)
    elif arguments.command == 'train':
        person_A = Person(arguments.person_A)
        person_A.merge_alignments()
        person_B = Person(arguments.person_B)
        person_B.merge_alignments()

        _faceswap.train(input_a_dir=person_A.faces_directory, input_b_dir=person_B.faces_directory, arguments=arguments)
    elif arguments.command == 'convert':
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        alignments_file = os.path.join(temp_dir.name, _ALIGNMENTS_FILE)
        _faceswap.extract_for_conversion(input_dir=arguments.input_path, output_dir=temp_dir.name,
                                         alignments=alignments_file, arguments=arguments)
        _faceswap.convert(arguments=arguments, alignments=alignments_file)

        temp_dir.cleanup()
    elif arguments.command == 'convert_live':
        _faceswap.convert_live(arguments=arguments)