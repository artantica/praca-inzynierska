"""
Source: https://github.com/deepfakes/faceswap
"""
import os

from .utils import get_folder

_DATA_DIRECTORY = "data"
_MODEL_DIRECTORY = 'models'
_FILTER_IMAGES = 'filter_images'
_PROCESSED_DIRECTORY = 'processed'
_VIDEOS = 'videos'
_IMAGES = 'images'
_FRAMES = 'frames'
_FACES = 'faces'
_ALIGNMENTS_FILE = "alignments.fsa"

sources = dict()

def get_sources_info():
    get_people()
    get_models()

def get_people():
    rootdir = get_folder(_DATA_DIRECTORY)
    sources['people'] = ([name for name in os.listdir(rootdir) if not name.startswith(".")])

def get_models():
    rootdir = get_folder(_MODEL_DIRECTORY)
    sources['models'] = ([name for name in os.listdir(rootdir) if not name.startswith(".")])
    pass

get_sources_info()