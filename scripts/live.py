import logging
import re
import os
import sys
from threading import Event
from time import sleep

import cv2

logger = logging.getLogger(__name__)

class Convert_Live:
    def __init__(self, arguments):
        self.arguments = arguments
        self._process()

    def _process(self):
        pass
