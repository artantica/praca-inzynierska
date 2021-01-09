"""
Source: https://github.com/deepfakes/faceswap
"""

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

from .utils import get_folder, _video_extensions
from .sources import (sources, _DATA_DIRECTORY, _PROCESSED_DIRECTORY, _VIDEOS, _IMAGES,
                      _FRAMES, _FACES, _ALIGNMENTS_FILE)


def count_number_of_matching_files(directory, regex):
    files = os.listdir(directory)
    count = len([file for file in files if re.match(regex, file)])
    return count

class Person:
    def __init__(self, name):
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

        # path_filter_images = os.path.join(_DATA_DIRECTORY, self.name, _FILTER_IMAGES)
        # self.filter_images_directory = get_folder(path=path_filter_images)

        path_images = os.path.join(_DATA_DIRECTORY, self.name, _IMAGES)
        self.images_directory = get_folder(path=path_images)

        path_frames = os.path.join(_DATA_DIRECTORY, self.name, _FRAMES)
        self.frames_directory = get_folder(path=path_frames)

        path_faces = os.path.join(_DATA_DIRECTORY, self.name, _FACES)
        self.faces_directory = get_folder(path=path_faces)

        path_processed = os.path.join(_DATA_DIRECTORY, self.name, _PROCESSED_DIRECTORY)
        self.processed_directory = get_folder(path=path_processed)

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

    # def _get_videos(self):
    #     for video_filename in os.listdir(self.videos_directory):
    #         if os.path.splitext(video_filename)[1] in _video_extensions:
    #             video_filepath = os.path.join(self.videos_directory, video_filename)
    #             video_name = os.path.splitext(video_filename)[0]
    #             video = Video(video_name=video_name, filepath=video_filepath, frames_directory=self.frames_directory, faces_directory=self.faces_directory)
    #             self.videos.append(video)
    #
    # def add_video(self, filepath):
    #     extension = os.path.splitext(filepath)[1]
    #     new_name = self.name + str(self.number_of_videos) + extension
    #     shutil.copy(filepath, os.path.join(self.videos_directory, new_name))
    #
    #     self.number_of_videos += 1
    #     video_name = os.path.basename(new_name)
    #     print(video_name)
    #     if video_name not in [video.video_name for video in self.videos]:
    #         video = Video(video_name=video_name, filepath=os.path.join(self.videos_directory, video_name), frames_directory=self.frames_directory, faces_directory=self.faces_directory)
    #         self.videos.append(video)
    #
    # def add_filter_image(self, filepath):
    #     shutil.copy(filepath, self.filter_images_directory)
    #
    # def extract_all_videos(self):
    #     if self.videos:
    #         for video in self.videos:
    #             print(video.video_name)
    #             self._extract_frames(video)
    #
    #         self._extract_faces()

            # self._merge_alignments()

    # def _extract_frames(self, video):
    #     video_clip = VideoFileClip(video.filepath)
    #
    #     start_time = time.time()
    #     print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video.video_name,
    #                                                                                        video_clip.fps,
    #                                                                                        video_clip.duration))
    #     frame_number = 0
    #     for frame in tqdm.tqdm(video_clip.iter_frames(fps=video_clip.fps), total=video_clip.fps * video_clip.duration):
    #         video_frame_file = os.path.join(video.frames_directory, f'{video.video_name}_frame_{frame_number:04d}.jpg')
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Swap RGB to BGR to work with OpenCV
    #         cv2.imwrite(video_frame_file, frame)
    #         frame_number += 1
    #
    #     video.set_frames_number(frame_number-1)
    #
    #     print(f'[extract] finished extract_frames for {video.frames_directory}, total frames {frame_number - 1}, time taken {time.time() - start_time:.0f}s')
    #
    # def _extract_faces_of_video(self, video):
    #     start_time = time.time()
    #     print(f'[extract-faces] about to extract faces for {video.faces_directory}')
    #
    #     _faceswap.extract(input_dir=video.frames_directory, output_dir=video.faces_directory, aligments=video.alignments)
    #     video.count_number_of_faces()
    #
    # def _extract_faces(self):
    #     _faceswap.extract(input_dir=self.frames_directory, output_dir=self.faces_directory,
    #                       aligments=os.path.join(self.faces_directory, _ALIGNMENTS_FILE))

    # def _get_alignments(self, video):
    #     self._people[person]['alignments'].append(align_file_path)

    # def _merge_alignments(self):
    #     alignments_files = [video.alignments for video in self.videos]
    #     _faceswap_tool.align_merge(alignments_files, self.faces_directory)
    #
    # def _move_alignment_file(self, person):
    #     merged_alignments = None
    #     first = self._people[person]['alignments'][0]
    #     directory = os.path.dirname(first)
    #
    #     for file in os.listdir(directory):
    #         if file.endswith(".fsa") and file != 'alignments.fsa':
    #             merged_alignments = file
    #
    #     if merged_alignments:
    #         merged_alignments_path = os.path.join(directory, merged_alignments)
    #         new_merged_alignments_path = os.path.join(self._model_person_data_path(person), "alignments.fsa")
    #         print(f"Current: {merged_alignments_path}, new: {new_merged_alignments_path}")
    #         os.replace(merged_alignments_path, new_merged_alignments_path)
