import os
from argparse import Namespace
import argparse
import cv2
import time
import tqdm
import numpy

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx.all import crop
from moviepy.editor import AudioFileClip, clips_array, TextClip, CompositeVideoClip
import shutil

from pathlib import Path

import sys
sys.path.append('faceswap')

from lib.cli import args
from tools.alignments.cli import AlignmentsArgs
from lib.config import generate_configs

import json


class FaceSwapInterface:
    def __init__(self):
        self._parser = args.FullHelpArgumentParser()
        generate_configs()
        self._subparser = self._parser.add_subparsers()

    def extract(self, input_dir, output_dir, filter_path):
        extract = args.ExtractArgs(
            self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = "extract --input-dir {} --output-dir {} --detector cv2-dnn" #--filter {}
        args_str = args_str.format(input_dir, output_dir) #filter_path
        self._run_script(args_str)

    def train(self, input_a_dir, input_b_dir, model_dir):
        model_type = "Original"

        train = args.TrainArgs(
            self._subparser, "train", "This command trains the model for the two faces A and B.")
        args_str = "train --input-A {} --input-B {} --model-dir {} --trainer {} --batch-size {} --write-image"
        args_str = args_str.format(input_a_dir, input_b_dir, model_dir, model_type, 20)
        self._run_script(args_str)

    def _run_script(self, args_str):
        args = self._parser.parse_args(args_str.split(' '))
        args.func(args)

class FaceSwapToolsInterface:
    def __init__(self):
        self._parser = args.FullHelpArgumentParser()
        generate_configs()
        self._subparser = self._parser.add_subparsers()
        self._parser.set_defaults(func=self._bad_args)

    def _bad_args(self, *args):
        """ Print help on bad arguments """
        self._parser.print_help()
        sys.exit(0)

    def align_merge(self, aligments_path, faces_dir):
        align = AlignmentsArgs(
            self._subparser, "alignments", "This command lets you perform various tasks pertaining to an alignments file.")
        args_str = "alignments --job merge --alignments_file {} -faces_folder {}"
        args_str = args_str.format(" ".join(aligments_path), faces_dir)
        self._run_script(args_str)

    def _run_script(self, args_str):
        self._set_tools()
        args = self._parser.parse_args(args_str.split(' '))
        args.func(args)
        self._unset_tools()

    def _set_tools(self):
        original = {"original": False}
        data = None
        with open("faceswap/config/.faceswap", "r") as cnf:
            data = json.load(cnf)
            data.update(original)
        with open("faceswap/config/.faceswap", "w") as cnf:
            json.dump(data, cnf)

    def _unset_tools(self):
        original = {"original": True} # TODO: change na tools
        data = None
        with open("faceswap/config/.faceswap", "r") as cnf:
            data = json.load(cnf)
            data.update(original)
        with open("faceswap/config/.faceswap", "w") as cnf:
            json.dump(data, cnf)

class MyFaceSwap:
    VIDEO_PATH = 'data/videos'
    PERSON_PATH = 'data/persons'
    PROCESSED_PATH = 'data/processed'
    OUTPUT_PATH = 'data/output'
    MODEL_PATH = 'models'
    MODELS = {}

    @classmethod
    def add_model(cls, model):
        MyFaceSwap.MODELS[model._name] = model

    def __init__(self, name, person_a, person_b):
        def _create_person_data(person):
            return {
                'name': person,
                'videos': [],
                'faces': os.path.join(MyFaceSwap.PERSON_PATH, person + '.jpg'),
                'photos': [],
                'alignments': []
            }

        self._name = name

        self._people = {
            person_a: _create_person_data(person_a),
            person_b: _create_person_data(person_b),
        }
        self._person_a = person_a
        self._person_b = person_b

        self._faceswap = FaceSwapInterface()

        self._faceswap_tool = FaceSwapToolsInterface()

        if not os.path.exists(os.path.join(MyFaceSwap.VIDEO_PATH)):
            os.makedirs(MyFaceSwap.VIDEO_PATH)

    def add_photos(self, person, photo_dir):
        self._people[person]['photos'].append(photo_dir)

    def add_video(self, person, name, url=None, fps=20):
        self._people[person]['videos'].append({
            'name': name,
            'url': url,
            'fps': fps
        })

    def fetch(self):
        self._process_media(self._fetch_video)

    def extract_frames(self):
        self._process_media(self._extract_frames)

    def extract_faces(self):
        self._process_media(self._extract_faces)
        self._process_media(self._extract_faces_from_photos, 'photos')

    def merge_alignments(self):
        self._process_media(self._get_alignments)
        self._process_person(self._merge_alignments)
        self._process_person(self._move_alignment_file)

    def all_videos(self):
        return self._people[self._person_a]['videos'] + self._people[self._person_b]['videos']

    def _process_media(self, func, media_type='videos'):
        for person in self._people:
            for video in self._people[person][media_type]:
                func(person, video)

    def _process_person(self, func):
        for person in self._people:
            func(person)

    def _video_path(self, video):
        return os.path.join(MyFaceSwap.VIDEO_PATH, video['name'])

    def _video_frames_path(self, video):
        return os.path.join(MyFaceSwap.PROCESSED_PATH, video['name'] + '_frames')

    def _video_faces_path(self, video):
        return os.path.join(MyFaceSwap.PROCESSED_PATH, video['name'] + '_faces')

    def _model_path(self):
        path = MyFaceSwap.MODEL_PATH
        return os.path.join(path, self._name)

    def _model_data_path(self):
        return os.path.join(MyFaceSwap.PROCESSED_PATH, "model_data_" + self._name)

    def _model_person_data_path(self, person):
        return os.path.join(self._model_data_path(), person)

    def _extract_frames(self, person, video):
        video_frames_dir = self._video_frames_path(video)
        video_clip = VideoFileClip(self._video_path(video))

        start_time = time.time()
        print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video_frames_dir,
                                                                                           video_clip.fps,
                                                                                           video_clip.duration))

        if os.path.exists(video_frames_dir):
            print('[extract-frames] frames already exist, skipping extraction: {}'.format(video_frames_dir))
            return

        os.makedirs(video_frames_dir)
        frame_num = 0
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video['fps']), total=video_clip.fps * video_clip.duration):
            video_frame_file = os.path.join(video_frames_dir, 'frame_{:03d}.jpg'.format(frame_num))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_num += 1

        print('[extract] finished extract_frames for {}, total frames {}, time taken {:.0f}s'.format(
            video_frames_dir, frame_num - 1, time.time() - start_time))

    def _extract_faces(self, person, video):
        video_faces_dir = self._video_faces_path(video)

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(video_faces_dir))

        if os.path.exists(video_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(video_faces_dir))
            return

        os.makedirs(video_faces_dir)
        self._faceswap.extract(self._video_frames_path(video), video_faces_dir, self._people[person]['faces'])

    def _get_alignments(self, person, video):
        align_file_path = os.path.join(os.getcwd(), self._video_frames_path(video), 'alignments.fsa')
        print(align_file_path)
        self._people[person]['alignments'].append(align_file_path)

    def _merge_alignments(self, person):
        self._faceswap_tool.align_merge(self._people[person]['alignments'], self._model_person_data_path(person))

    def _move_alignment_file(self, person):
        merged_alignments = None
        first = self._people[person]['alignments'][0]
        directory = os.path.dirname(first)

        for file in os.listdir(directory):
            if file.endswith(".fsa") and file != 'alignments.fsa':
                merged_alignments = file

        if merged_alignments:
            merged_alignments_path = os.path.join(directory, merged_alignments)
            new_merged_alignments_path = os.path.join(self._model_person_data_path(person), "alignments.fsa")
            print(f"Current: {merged_alignments_path}, new: {new_merged_alignments_path}")
            os.replace(merged_alignments_path, new_merged_alignments_path)

    def _extract_faces_from_photos(self, person, photo_dir):
        photo_faces_dir = self._video_faces_path({'name': photo_dir})

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(photo_faces_dir))

        if os.path.exists(photo_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(photo_faces_dir))
            return

        os.makedirs(photo_faces_dir)
        self._faceswap.extract(self._video_path({'name': photo_dir}), photo_faces_dir, self._people[person]['faces'])

    def preprocess(self):
        self.extract_frames()
        self.extract_faces()
        # self.merge_faces()

    def _symlink_faces_for_model(self, person, video):
        if isinstance(video, str):
            video = {'name': video}
        for face_file in os.listdir(self._video_faces_path(video)):
            target_file = os.path.join(self._model_person_data_path(person), video['name'] + "_" + face_file)
            face_file_path = os.path.join(os.getcwd(), self._video_faces_path(video), face_file)
            os.symlink(face_file_path, target_file)

    def train(self):
        # Setup directory structure for model, and create one director for person_a faces, and
        # another for person_b faces containing symlinks to all faces.
        if not os.path.exists(self._model_path()):
            os.makedirs(self._model_path())

        if os.path.exists(self._model_data_path()):
            shutil.rmtree(self._model_data_path())

        for person in self._people:
            os.makedirs(self._model_person_data_path(person))
        self._process_media(self._symlink_faces_for_model)

        #merge align files
        self.merge_alignments()

        self._faceswap.train(self._model_person_data_path(self._person_a), self._model_person_data_path(self._person_b),
                             self._model_path())

    def convert(self, video_file, swap_model=False, duration=None, start_time=None, use_gan=False, face_filter=False,
                photos=True, crop_x=None, width=None, side_by_side=False):
        # Magic incantation to not have tensorflow blow up with an out of memory error.
        import tensorflow as tf
        import keras.backend.tensorflow_backend as K
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        K.set_session(tf.Session(config=config))

        # Load model
        model_name = "Original"
        converter_name = "Masked"
        if use_gan:
            model_name = "GAN"
            converter_name = "GAN"
        model = PluginLoader.get_model(model_name)(Path(self._model_path(use_gan)))
        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        # Load converter
        converter = PluginLoader.get_converter(converter_name)
        converter = converter(model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        # Load face filter
        filter_person = self._person_a
        if swap_model:
            filter_person = self._person_b
        filter = FaceFilter(self._people[filter_person]['faces'])

        # Define conversion method per frame
        def _convert_frame(frame, convert_colors=True):
            if convert_colors:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Swap RGB to BGR to work with OpenCV
            for face in detect_faces(frame, "cnn"):
                if (not face_filter) or (face_filter and filter.check(face)):
                    frame = converter.patch_image(frame, face)
                    frame = frame.astype(numpy.float32)
            if convert_colors:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Swap RGB to BGR to work with OpenCV
            return frame

        def _convert_helper(get_frame, t):
            return _convert_frame(get_frame(t))

        media_path = self._video_path({'name': video_file})
        if not photos:
            # Process video; start loading the video clip
            video = VideoFileClip(media_path)

            # If a duration is set, trim clip
            if duration:
                video = video.subclip(start_time, start_time + duration)

            # Resize clip before processing
            if width:
                video = video.resize(width=width)

            # Crop clip if desired
            if crop_x:
                video = video.fx(crop, x2=video.w / 2)

            # Kick off convert frames for each frame
            new_video = video.fl(_convert_helper)

            # Stack clips side by side
            if side_by_side:
                def add_caption(caption, clip):
                    text = (TextClip(caption, font='Amiri-regular', color='white', fontsize=80).
                            margin(40).
                            set_duration(clip.duration).
                            on_color(color=(0, 0, 0), col_opacity=0.6))
                    return CompositeVideoClip([clip, text])

                video = add_caption("Original", video)
                new_video = add_caption("Swapped", new_video)
                final_video = clips_array([[video], [new_video]])
            else:
                final_video = new_video

            # Resize clip after processing
            # final_video = final_video.resize(width = (480 * 2))

            # Write video
            output_path = os.path.join(self.OUTPUT_PATH, video_file)
            final_video.write_videofile(output_path, rewrite_audio=True)

            # Clean up
            del video
            del new_video
            del final_video
        else:
            # Process a directory of photos
            for face_file in os.listdir(media_path):
                face_path = os.path.join(media_path, face_file)
                image = cv2.imread(face_path)
                image = _convert_frame(image, convert_colors=False)
                cv2.imwrite(os.path.join(self.OUTPUT_PATH, face_file), image)


if __name__ == '__main__':
    faceit = MyFaceSwap('emilia_to_jeniffer', 'Emilia', "Jeniffer")

    faceit.add_video('Emilia', 'EmiliaClarke_1.mp4')
    faceit.add_video('Emilia', 'EmiliaClarke_2.mp4')
    faceit.add_video('Emilia', 'EmiliaClarke_3.mp4')
    faceit.add_video("Jeniffer", 'JenifferAniston_1.mp4')
    faceit.add_video("Jeniffer", 'JenifferAniston_2.mp4')
    faceit.add_video("Jeniffer", 'JenifferAniston_3.mp4')
    MyFaceSwap.add_model(faceit)

    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['preprocess', 'train', 'convert'])
    parser.add_argument('model', choices=MyFaceSwap.MODELS.keys())
    parser.add_argument('video', nargs='?')
    parser.add_argument('--duration', type=int, default=None)
    parser.add_argument('--photos', action='store_true', default=False)
    parser.add_argument('--swap-model', action='store_true', default=False)
    parser.add_argument('--face-filter', action='store_true', default=False)
    parser.add_argument('--start-time', type=int, default=0)
    parser.add_argument('--crop-x', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--side-by-side', action='store_true', default=False)
    arguments = parser.parse_args()

    faceit = MyFaceSwap.MODELS[arguments.model]

    if arguments.task == 'preprocess':
        faceit.preprocess()
    elif arguments.task == 'train':
        faceit.train()
    elif arguments.task == 'convert':
        if not arguments.video:
            print('Need a video to convert. Some ideas: {}'.format(
                ", ".join([video['name'] for video in faceit.all_videos()])))
        else:
            faceit.convert(arguments.video, duration=arguments.duration, swap_model=arguments.swap_model, face_filter=arguments.face_filter,
                           start_time=arguments.start_time, photos=arguments.photos, crop_x=arguments.crop_x, width=arguments.width,
                           side_by_side=arguments.side_by_side)
