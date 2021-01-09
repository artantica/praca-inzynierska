#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import logging
import re
import os
import sys
from threading import Event
from time import sleep

import cv2
import numpy as np
from tqdm import tqdm

from scripts.fsmedia import Alignments, PostProcess, finalize
from lib.serializer import get_serializer
from lib.live_convert import LiveConverter
from lib.gpu_stats import GPUStats
from lib.image import read_image_hash, ImagesLoader
from lib.multithreading import MultiThread, total_cpus
from lib.keypress import KBHit
from lib.queue_manager import queue_manager
from lib.utils import FaceswapError, get_backend, get_folder, get_image_paths
from plugins.extract.pipeline import Extractor, ExtractMedia
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert_Live:  # pylint:disable=too-few-public-methods
    """ The Faceswap Face Conversion Process.

    The conversion process is responsible for swapping the faces on source frames with the optional
    output from a trained model.

    It leverages a series of user selected post-processing plugins, executed from
    :class:`lib.convert.Converter`.

    The convert process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self._args = arguments
        self.results_path = self.get_output(arguments=arguments)

        self._patch_threads = None

        self._webcam_io = WebcamIO(self._args)
        self._predictor = Predict(self._args)

        self.model_name = self._args.model_dir.split("/")[-1]

        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        self._converter = LiveConverter(self._predictor.output_size,
                                    self._predictor.coverage_ratio,
                                    self._webcam_io.draw_transparent,
                                    self._webcam_io.pre_encode,
                                    arguments,
                                    configfile=configfile)

        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def get_output(arguments):
        """ If output argument is specified, set output path
            otherwise return None """
        if not hasattr(arguments, "output_dir") or not arguments.output_dir:
            return None
        return get_folder(arguments.output_dir)

    @property
    def _queue_size(self):
        """ int: Size of the converter queues. 16 for single process otherwise 32 """
        retval = 16 #TODO: add prepoer args
        # if self._args.singleprocess:
        #     retval = 16
        # else:
        #     retval = 32
        logger.debug(retval)
        return retval

    @property
    def _pool_processes(self):
        """ int: The number of threads to run in parallel. Based on user options and number of
        available processors. """
        retval = total_cpus()
        retval = 1 if retval == 0 else retval
        logger.debug(retval)
        return retval

    def process(self):
        """ The entry point for triggering the Live Conversion Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.debug("Starting Live Conversion")
        print("Staring live mode. Capturing video from web camera!")
        print("Press Enter to Quit")

        video_capture = cv2.VideoCapture(0)
        import time
        time.sleep(1)
        counter = 0

        images_to_save = []
        keypress = KBHit(is_gui=False)

        while True:
            status, frame = video_capture.read()
            if status:
                # frame = cv2.resize(frame, (640, 480))
                # flip image, because webcam inverts it and we trained the model the other way!
                frame = cv2.flip(frame, 1)
                # cv2.imshow('Real', frame)
                image = self.convert_frame(frame, convert_colors=False)

                cv2.imshow('Live video', image)

                if self.results_path:
                    image_save = image * 255.0
                    image_save = np.rint(image_save,
                                       out=np.empty(image_save.shape, dtype="uint8"),
                                       casting='unsafe')
                    images_to_save.append((counter, image_save))
                # if self._writer_pre_encode is not None:
                #     patched_face = self._writer_pre_encode(patched_face)
                #     print("After _writer_pre_encode")

                # swapped/filename = "{}_swapped_{:04d}.png".format(self.model_name, counter)
                # swapped_path = os.path.join(swapped_frame_path, swapped_filename)
                # args = (cv2.IMWRITE_PNG_COMPRESSION, 3)
                # image = cv2.imencode(".png", image, args)[1]
                # with open(swapped_path, "wb") as outfile:
                #     outfile.write(image)
                # cv2.imwrite(swapped_path, image_save)

            # Hit 'enter' on the keyboard to quit!
            if cv2.waitKey(1) and keypress.kbhit():
                console_key = keypress.getch()
                if console_key in ("\n", "\r"):
                    logger.debug("Exit requested")
                    video_capture.release()
                    break
            counter += 1

        cv2.destroyAllWindows()
        # TODO: zapis pliku
        # self._webcam_io.save(images_to_save)
        import moviepy.video.io.ImageSequenceClip
        if self.results_path:
            import tempfile
            temp_dir = tempfile.TemporaryDirectory()
            image_files = []
            for item in images_to_save:
                path = os.path.join(temp_dir.name, f"{item[0]}.png")
                cv2.imwrite(path, item[1])
                image_files.append(path)

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=6)
            clip.write_videofile('my_video.mp4')
            temp_dir.cleanup()

        # exit()

    def convert_frame(self, frame, convert_colors=True):
        if convert_colors:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV

        item = dict(filename="filename", image=frame, detected_faces=self._webcam_io.get_detected_faces(frame))
        item = self._predictor.load_item(item)

        frame = self._converter.process(item)

        if convert_colors:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
        return frame

class WebcamIO:
    """ Webcam Input/Output for the converter process.

    Background threads to:
        * Load image from webcam and get the detected faces
        * Send images back to screen or save to disc

    Parameters
    ----------
    images: :class:`lib.image.ImagesLoader`
        The input images
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: %s)",
                     self.__class__.__name__, arguments)
        self._args = arguments
        # self._pre_process = PostProcess(arguments)
        # self._completion_event = Event()

        # For frame skipping
        self._imageidxre = re.compile(r"(\d+)(?!.*\d\.)(?=\.\w+$)")
        # self._writer = self._get_writer()

        # Extractor for on the fly detection
        self._extractor = self._load_extractor()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def draw_transparent(self):
        """ bool: ``True`` if the selected writer's Draw_transparent configuration item is set
        otherwise ``False`` """
        return False #self._writer.config.get("draw_transparent", False)

    @property
    def pre_encode(self):
        """ python function: Selected writer's pre-encode function, if it has one,
        otherwise ``None`` """
        # dummy = np.zeros((20, 20, 3), dtype="uint8")
        # test = self._writer.pre_encode(dummy)
        # retval = None if test is None else self._writer.pre_encode
        # logger.debug("Writer pre_encode function: %s", retval)
        return None #retval

    def get_detected_faces(self, image):
        return self._get_detected_faces(image)

    # Initialization
    def _get_writer(self):
        """ Load the selected writer plugin.

        Returns
        -------
        :mod:`plugins.convert.writer` plugin
            The requested writer plugin
        """
        args = [self._args.output_dir]
        if self._args.writer in ("ffmpeg", "gif"):
            args.extend([self._total_count, self._frame_ranges])
        if self._args.writer == "ffmpeg":
            if self._images.is_video:
                args.append(self._args.input_dir)
            else:
                args.append(self._args.reference_video)
        logger.debug("Writer args: %s", args)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        return PluginLoader.get_converter("writer", self._args.writer)(*args,
                                                                       configfile=configfile)

    def _load_extractor(self):
        """ Load the CV2-DNN Face Extractor Chain.

        For On-The-Fly conversion we use a CPU based extractor to avoid stacking the GPU.
        Results are poor.

        Returns
        -------
        :class:`plugins.extract.Pipeline.Extractor`
            The face extraction chain to be used for on-the-fly conversion
        """
        # if not self._alignments.have_alignments_file and not self._args.on_the_fly:
        #     logger.error("No alignments file found. Please provide an alignments file for your "
        #                  "destination video (recommended) or enable on-the-fly conversion (not "
        #                  "recommended).")
        #     sys.exit(1)
        # if self._alignments.have_alignments_file:
        #     if self._args.on_the_fly:
        #         logger.info("On-The-Fly conversion selected, but an alignments file was found. "
        #                     "Using pre-existing alignments file: '%s'", self._alignments.file)
        #     else:
        #         logger.debug("Alignments file found: '%s'", self._alignments.file)
        #     return None

        logger.debug("Loading extractor")
        logger.warning("On-The-Fly conversion selected. This will use the inferior cv2-dnn for "
                       "extraction and will produce poor results.")
        logger.warning("It is recommended to generate an alignments file for your destination "
                       "video with Extract first for superior results.")
        maskers = ["components", "extended"]
        extractor = Extractor(detector=self._args.detector, # "mtcnn", #"cv2-dnn", mtcnn
                              aligner=self._args.aligner, #"cv2-dnn",
                              masker=self._args.mask_type, #maskers, #"extended",
                              multiprocess=True,
                              rotate_images=None,
                              min_size=20)
        extractor.launch()
        logger.debug("Loaded extractor")
        return extractor

        normalization = None if self._args.normalization == "none" else self._args.normalization

        maskers = ["components", "extended"]
        maskers += self._args.masker if self._args.masker else []
        self._extractor = Extractor(self._args.detector,
                                    self._args.aligner,
                                    maskers,
                                    configfile=configfile,
                                    multiprocess=not self._args.singleprocess,
                                    exclude_gpus=self._args.exclude_gpus,
                                    rotate_images=self._args.rotate_images,
                                    min_size=self._args.min_size,
                                    normalize_method=normalization)

    def _get_detected_faces(self, image):
        """ Return the detected faces for the given image.

        If we have an alignments file, then the detected faces are created from that file. If
        we're running On-The-Fly then they will be extracted from the extractor.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.faces_detect.DetectedFace` objects
        """
        # logger.trace("Getting faces for: '%s'", filename)
        detected_faces = self._detect_faces(image)
        logger.trace("Got %s faces.", len(detected_faces))
        return detected_faces


    def _detect_faces(self, image):
        """ Extract the face from a frame for On-The-Fly conversion.

        Pulls detected faces out of the Extraction pipeline.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.faces_detect.DetectedFace` objects
         """

        self._extractor.input_queue.put(ExtractMedia("", image))
        # self._extractor.input_queue.put('EOF')
        faces = next(self._extractor.detected_faces())

        return faces.detected_faces

    # Saving tasks
    def _save(self):
        """ Save the converted images.

        Puts the selected writer into a background thread and feeds it from the output of the
        patch queue.

        Parameters
        ----------
        completion_event: :class:`event.Event`
            An even that this process triggers when it has finished saving
        """
        logger.debug("Save Images: Start")
        write_preview = self._args.redirect_gui and self._writer.is_stream
        preview_image = os.path.join(self._writer.output_folder, ".gui_preview.jpg")
        logger.debug("Write preview for gui: %s", write_preview)
        for idx in tqdm(range(self._total_count), desc="Converting", file=sys.stdout):
            if self._queues["save"].shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = self._queues["save"].get()
            if item == "EOF":
                logger.debug("EOF Received")
                break
            filename, image = item
            # Write out preview image for the GUI every 10 frames if writing to stream
            if write_preview and idx % 10 == 0 and not os.path.exists(preview_image):
                logger.debug("Writing GUI Preview image: '%s'", preview_image)
                cv2.imwrite(preview_image, image)
            self._writer.write(filename, image)
        self._writer.close()
        # completion_event.set()
        logger.debug("Save Faces: Complete")

class Predict():
    """ Obtains the output from the Faceswap model.

    Parameters
    ----------
    in_queue: :class:`queue.Queue`
        The queue that contains images and detected faces for feeding the model
    queue_size: int
        The maximum size of the input queue
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments):
        self._args = arguments

        self._serializer = get_serializer("json")
        self._faces_count = 0
        self._verify_output = False

        self._model = self._load_model()
        # self._batchsize = self._get_batchsize(queue_size)
        self._sizes = self._get_io_sizes()
        self._coverage_ratio = self._model.coverage_ratio

        logger.debug("Initialized %s:", self.__class__.__name__)

    def load_item(self, item):
        batch = self._predict_faces(item)
        return batch

    @property
    def faces_count(self):
        """ int: The total number of faces seen by the Predictor. """
        return self._faces_count

    @property
    def verify_output(self):
        """ bool: ``True`` if multiple faces have been found in frames, otherwise ``False``. """
        return self._verify_output

    @property
    def coverage_ratio(self):
        """ float: The coverage ratio that the model was trained at. """
        return self._coverage_ratio

    @property
    def has_predicted_mask(self):
        """ bool: ``True`` if the model was trained to learn a mask, otherwise ``False``. """
        return bool(self._model.config.get("learn_mask", False))

    @property
    def output_size(self):
        """ int: The size in pixels of the Faceswap model output. """
        return self._sizes["output"]

    def _get_io_sizes(self):
        """ Obtain the input size and output size of the model.

        Returns
        -------
        dict
            input_size in pixels and output_size in pixels
        """
        input_shape = self._model.model.input_shape
        input_shape = [input_shape] if not isinstance(input_shape, list) else input_shape
        output_shape = self._model.model.output_shape
        output_shape = [output_shape] if not isinstance(output_shape, list) else output_shape
        retval = dict(input=input_shape[0][1], output=output_shape[-1][1])
        logger.debug(retval)
        return retval

    def _load_model(self):
        """ Load the Faceswap model.

        Returns
        -------
        :mod:`plugins.train.model` plugin
            The trained model in the specified model folder
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir, make_folder=False)
        if not model_dir:
            raise FaceswapError("{} does not exist.".format(self._args.model_dir))
        trainer = self._get_model_name(model_dir)
        model = PluginLoader.get_model(trainer)(model_dir, self._args, predict=True)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _get_batchsize(self, queue_size):
        """ Get the batch size for feeding the model.

        Sets the batch size to 1 if inference is being run on CPU, otherwise the minimum of the
        input queue size and the model's `convert_batchsize` configuration option.

        Parameters
        ----------
        queue_size: int
            The queue size that is feeding the predictor

        Returns
        -------
        int
            The batch size that the model is to be fed at.
        """
        logger.debug("Getting batchsize")
        is_cpu = GPUStats().device_count == 0
        batchsize = 1 if is_cpu else self._model.config["convert_batchsize"]
        batchsize = min(queue_size, batchsize)
        logger.debug("Batchsize: %s", batchsize)
        logger.debug("Got batchsize: %s", batchsize)
        return batchsize

    def _get_model_name(self, model_dir):
        """ Return the name of the Faceswap model used.

        If a "trainer" option has been selected in the command line arguments, use that value,
        otherwise retrieve the name of the model from the model's state file.

        Parameters
        ----------
        model_dir: str
            The folder that contains the trained Faceswap model

        Returns
        -------
        str
            The name of the Faceswap model being used.

        """
        if hasattr(self._args, "trainer") and self._args.trainer:
            logger.debug("Trainer name provided: '%s'", self._args.trainer)
            return self._args.trainer

        statefile = [fname for fname in os.listdir(str(model_dir))
                     if fname.endswith("_state.json")]
        if len(statefile) != 1:
            raise FaceswapError("There should be 1 state file in your model folder. {} were "
                                "found. Specify a trainer with the '-t', '--trainer' "
                                "option.".format(len(statefile)))
        statefile = os.path.join(str(model_dir), statefile[0])

        state = self._serializer.load(statefile)
        trainer = state.get("name", None)

        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file. "
                                "Specify a trainer with the '-t', '--trainer' option.")
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    def _predict_faces(self, item):
        """ Run Prediction on the Faceswap model in a background thread.

        Reads from the :attr:`self._in_queue`, prepares images for prediction
        then puts the predictions back to the :attr:`self.out_queue`
        """
        faces_seen = 0
        consecutive_no_faces = 0
        batch = list()
        is_amd = get_backend() == "amd"

        logger.trace("Got from queue: '%s'", item["filename"])
        faces_count = len(item["detected_faces"])

        self._faces_count += faces_count
        if faces_count > 1:
            self._verify_output = True
            logger.verbose("Found more than one face in an image! '%s'",
                           os.path.basename(item["filename"]))

        self.load_aligned(item)

        faces_seen += faces_count
        batch.append(item)

        if batch:
            logger.trace("Batching to predictor. Frames: %s, Faces: %s",
                         len(batch), faces_seen)
            detected_batch = [detected_face for item in batch
                              for detected_face in item["detected_faces"]]
            if faces_seen != 0:
                feed_faces = self._compile_feed_faces(detected_batch)
                batch_size = None
                # if is_amd and feed_faces.shape[0] != self._batchsize:
                #     logger.verbose("Fallback to BS=1")
                #     batch_size = 1
                predicted = self._predict(feed_faces, batch_size)
            else:
                predicted = list()

            batch = self._queue_out_frames(batch, predicted)
        return batch

    def load_aligned(self, item):
        """ Load the model's feed faces and the reference output faces.

        For each detected face in the incoming item, load the feed face and reference face
        images, correctly sized for input and output respectively.

        Parameters
        ----------
        item: dict
            The incoming image and list of :class:`~lib.faces_detect.DetectedFace` objects

        """
        logger.trace("Loading aligned faces: '%s'", item["filename"])
        for detected_face in item["detected_faces"]:
            detected_face.load_feed_face(item["image"],
                                         size=self._sizes["input"],
                                         coverage_ratio=self._coverage_ratio,
                                         dtype="float32")
            if self._sizes["input"] == self._sizes["output"]:
                detected_face.reference = detected_face.feed
            else:
                detected_face.load_reference_face(item["image"],
                                                  size=self._sizes["output"],
                                                  coverage_ratio=self._coverage_ratio,
                                                  dtype="float32")
        logger.trace("Loaded aligned faces: '%s'", item["filename"])

    @staticmethod
    def _compile_feed_faces(detected_faces):
        """ Compile a batch of faces for feeding into the Predictor.

        Parameters
        ----------
        detected_faces: list
            List of `~lib.faces_detect.DetectedFace` objects

        Returns
        -------
        :class:`numpy.ndarray`
            A batch of faces ready for feeding into the Faceswap model.
        """
        logger.trace("Compiling feed face. Batchsize: %s", len(detected_faces))
        feed_faces = np.stack([detected_face.feed_face[..., :3]
                               for detected_face in detected_faces]) / 255.0
        logger.trace("Compiled Feed faces. Shape: %s", feed_faces.shape)
        return feed_faces

    def _predict(self, feed_faces, batch_size=None):
        """ Run the Faceswap models' prediction function.

        Parameters
        ----------
        feed_faces: :class:`numpy.ndarray`
            The batch to be fed into the model
        batch_size: int, optional
            Used for plaidml only. Indicates to the model what batch size is being processed.
            Default: ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped faces for the given batch
        """
        logger.trace("Predicting: Batchsize: %s", len(feed_faces))
        feed = [feed_faces]
        logger.trace("Input shape(s): %s", [item.shape for item in feed])

        predicted = self._model.model.predict(feed, batch_size=batch_size)
        predicted = predicted if isinstance(predicted, list) else [predicted]
        logger.trace("Output shape(s): %s", [predict.shape for predict in predicted])

        # Only take last output(s)
        if predicted[-1].shape[-1] == 1:  # Merge mask to alpha channel
            predicted = np.concatenate(predicted[-2:], axis=-1).astype("float32")
        else:
            predicted = predicted[-1].astype("float32")

        logger.trace("Final shape: %s", predicted.shape)
        return predicted

    def _queue_out_frames(self, batch, swapped_faces):
        """ Compile the batch back to original frames and put to the Out Queue.

        For batching, faces are split away from their frames. This compiles all detected faces
        back to their parent frame before putting each frame to the out queue in batches.

        Parameters
        ----------
        batch: dict
            The batch that was used as the input for the model predict function
        swapped_faces: :class:`numpy.ndarray`
            The predictions returned from the model's predict function
        """
        logger.trace("Queueing out batch. Batchsize: %s", len(batch))
        pointer = 0
        for item in batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["swapped_faces"] = np.array(list())
            else:
                item["swapped_faces"] = swapped_faces[pointer:pointer + num_faces]

            logger.trace("Putting to queue. ('%s', detected_faces: %s, swapped_faces: %s)",
                         item["filename"], len(item["detected_faces"]),
                         item["swapped_faces"].shape[0])
            pointer += num_faces

        logger.trace("Queued out batch. Batchsize: %s", len(batch))
        return batch


class OptionalActions():  # pylint:disable=too-few-public-methods
    """ Process specific optional actions for Convert.

    Currently only handles skip faces. This class should probably be (re)moved.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    input_images: list
        List of input image files
    alignments: :class:`lib.alignments.Alignments`
        The alignments file for this conversion
    """

    def __init__(self, arguments, input_images, alignments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._args = arguments
        self._input_images = input_images
        self._alignments = alignments

        self._remove_skipped_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # SKIP FACES #
    def _remove_skipped_faces(self):
        """ If the user has specified an input aligned directory, remove any non-matching faces
        from the alignments file. """
        logger.debug("Filtering Faces")
        face_hashes = self._get_face_hashes()
        if not face_hashes:
            logger.debug("No face hashes. Not skipping any faces")
            return
        pre_face_count = self._alignments.faces_count
        self._alignments.filter_hashes(face_hashes, filter_out=False)
        logger.info("Faces filtered out: %s", pre_face_count - self._alignments.faces_count)

    def _get_face_hashes(self):
        """ Check for the existence of an aligned directory for identifying which faces in the
        target frames should be swapped.

        Returns
        -------
        list
            A list of face hashes that exist in the given input aligned directory.
        """
        face_hashes = list()
        input_aligned_dir = self._args.input_aligned_dir

        if input_aligned_dir is None:
            logger.verbose("Aligned directory not specified. All faces listed in the "
                           "alignments file will be converted")
        elif not os.path.isdir(input_aligned_dir):
            logger.warning("Aligned directory not found. All faces listed in the "
                           "alignments file will be converted")
        else:
            file_list = get_image_paths(input_aligned_dir)
            logger.info("Getting Face Hashes for selected Aligned Images")
            for face in tqdm(file_list, desc="Hashing Faces"):
                face_hashes.append(read_image_hash(face))
            logger.debug("Face Hashes: %s", (len(face_hashes)))
            if not face_hashes:
                raise FaceswapError("Aligned directory is empty, no faces will be converted!")
            if len(face_hashes) <= len(self._input_images) / 3:
                logger.warning("Aligned directory contains far fewer images than the input "
                               "directory, are you sure this is the right folder?")
        return face_hashes
