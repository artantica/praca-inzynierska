# praca-inzynierska Informatyka na WPPT rocznik 2017

Repozytorium z plikami dotyczącymi pracy inżynierskiej p.t. `Aplikacja podmieniająca twarz na nagraniach wideo` (ang. `Face-swapping application for video recordings`)

Wszelkie notatki i przemyślenia znajdują się w pliku [Notatki.md](Notatki.md)

# DYSK Z MATERIAŁAMI
- [Link do dysku zawierającego wytrenowane modele](https://drive.google.com/drive/folders/1S20eGUh6uvBmnVMiTDVRQWlbVa3ITbOX?usp=sharing)
- [Link do dysku zawierającego zdjęcia osób i wydobyte twarze, na których były wytrenowane modele](https://drive.google.com/drive/folders/1Wv7QKvLnM8RLZvUaEA3JEoUs072wcoZg?usp=sharing)

# REPOZYTORIUM
[Link do repozytorium GitHub zawierajacy kod źródłowy](https://github.com/artantica/praca-inzynierska.git)
# LATEX
- [Link z uprawnieniami do wyświetlania](https://www.overleaf.com/read/gbgjwrqzpmbx)

# APLIKACJA

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
[![Python](https://img.shields.io/badge/pip-20.3.3-blue)](https://img.shields.io/badge/pip-20.3.3-blue)


## INSTALACJA
Opcjonalne (zalecane), zależności programowe mogą być zarządzane używając `pipenv`, który może zostać zainstalowany za pomocą`pip`. Po pomyślnej instalacji, odpalenie powłoki `pipenv` za pomocą komendy w głównym folderze projektu, po której następuję komenda instalacji `pipenv` skutkuje pomyślną instalacją wszystkich koniecznych modułów.

Aby w łatwy sposób przeprowadzić instalację potrzebnych modułów należy uruchomić komendę 
``` python setup.py  ```
i postępować zgodnie z instrukcją. 

## WYDOBYCIE TWARZY
```bash
usage: main.py extract [-h] [-i INPUT_PATH] [-p NAME] [-D {cv2-dnn,mtcnn,s3fd}] [-A {cv2-dnn,fan}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        Input directory or video. Either a directory containing the image files you wish to process or path to a video file. NB: This should be the source video/frames NOT the source faces')
  -p NAME, --person NAME
                        Name of a person, you wish to process. There will be folder with this name created in data/ to store all important files within it.
  -D {cv2-dnn,mtcnn,s3fd}, --detector {cv2-dnn,mtcnn,s3fd}
                        R|Detector to use. Some of these have configurable settings in '/config/extract.ini' or 'Settings > Configure Extract 'Plugins': L|cv2-dnn: A CPU only extractor which is the least reliable
                        and least resource intensive. Use this if not using a GPU and time is important. L|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources than other GPU detectors but can
                        often return more false positives. L|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and fewer false positives than other GPU detectors, but is a lot more resource
                        intensive.
  -A {cv2-dnn,fan}, --aligner {cv2-dnn,fan}
                        R|Aligner to use. L|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, but less accurate. Only use this if not using a GPU and time is important. L|fan: Best aligner.
                        Fast on GPU, slow on CPU.
```

## TRENING
```bash
usage: main.py train [-h] -A PERSON_A -B PERSON_B -m MODEL_DIR [-t {original,realface,villain,dfl-sae}] [-bs BATCH_SIZE] [-it ITERATIONS] [-ss SNAPSHOT_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  -A PERSON_A, --person-A PERSON_A
                        Name of a person A. This is the original face, i.e. the face that you want to remove and replace with face B. This person must have extracted faces to proceed.
  -B PERSON_B, --person-B PERSON_B
                        Name of a person B. This is the swap face, i.e. the face that you want to placeonto the head of person A. This person must have extracted faces to proceed.
  -m MODEL_DIR, --model MODEL_DIR
                        Model directory. This is where the training data will be stored. You should always specify a new folder for new models. If starting a new model, select either an empty folder, or a folder
                        which does not exist (which will be created). If continuing to train an existing model, specify the location of the existing model.
  -t {original,realface,villain,dfl-sae}, --trainer {original,realface,villain,dfl-sae}
                        R|Select which trainer to use. Trainers can be configured from the Settings menu or the config folder. L|original: The original model created by /u/deepfakes. L|dfl-sae: Adaptable model from
                        deepfacelab L|realface: A high detail, dual density model based on DFaker, with customizable in/out resolution. The autoencoders are unbalanced so B>A swaps won't work so well. By andenixa
                        et al. Very configurable. L|villain: 128px in/out model from villainguy. Very resource hungry (You will require a GPU with a fair amount of VRAM). Good for details, but more susceptible to
                        color differences.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. This is the number of images processed through the model for each side per iteration. NB: As the model is fed 2 sides at a time, the actual number of images within the model at
                        any one time is double the number that you set here. Larger batches require more GPU RAM.
  -it ITERATIONS, --iterations ITERATIONS
                        Length of training in iterations. This is only really used for automation. There is no 'correct' number of iterations a model should be trained for. You should stop training when you are
                        happy with the previews. However, if you want the model to stop automatically at a set number of iterations, you can set that value here.
  -ss SNAPSHOT_INTERVAL, --snapshot-interval SNAPSHOT_INTERVAL
                        Sets the number of iterations before saving a backup snapshot of the model in it's current state. Set to 0 for off.
```

## KONWERSJA
```bash
usage: main.py convert [-h] [-i INPUT_PATH] -m MODEL_DIR [-t {original,realface,villain,dfl-sae}] -o OUTPUT_PATH [-D {cv2-dnn,mtcnn,s3fd}] [-A {cv2-dnn,fan}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        Input directory or video. Either a directory containing the image files you wish to process or path to a video file. NB: This should be the source video/frames NOT the source faces')
  -m MODEL_DIR, --model MODEL_DIR
                        Model directory. This is where the training data will be stored. You should always specify a new folder for new models. If starting a new model, select either an empty folder, or a folder
                        which does not exist (which will be created). If continuing to train an existing model, specify the location of the existing model.
  -t {original,realface,villain,dfl-sae}, --trainer {original,realface,villain,dfl-sae}
                        R|Select which trainer to use. Trainers can be configured from the Settings menu or the config folder. L|original: The original model created by /u/deepfakes. L|dfl-sae: Adaptable model from
                        deepfacelab L|realface: A high detail, dual density model based on DFaker, with customizable in/out resolution. The autoencoders are unbalanced so B>A swaps won't work so well. By andenixa
                        et al. Very configurable. L|villain: 128px in/out model from villainguy. Very resource hungry (You will require a GPU with a fair amount of VRAM). Good for details, but more susceptible to
                        color differences.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Output directory. This is where the converted file will be saved.
  -D {cv2-dnn,mtcnn,s3fd}, --detector {cv2-dnn,mtcnn,s3fd}
                        R|Detector to use. Some of these have configurable settings in '/config/extract.ini' or 'Settings > Configure Extract 'Plugins': L|cv2-dnn: A CPU only extractor which is the least reliable
                        and least resource intensive. Use this if not using a GPU and time is important. L|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources than other GPU detectors but can
                        often return more false positives. L|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and fewer false positives than other GPU detectors, but is a lot more resource
                        intensive.
  -A {cv2-dnn,fan}, --aligner {cv2-dnn,fan}
                        R|Aligner to use. L|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, but less accurate. Only use this if not using a GPU and time is important. L|fan: Best aligner.
                        Fast on GPU, slow on CPU.

```

## KONWESJA NA ŻYWO
```bash
usage: main.py convert_live [-h] -m MODEL_DIR [-o OUTPUT_PATH] [-D {cv2-dnn,mtcnn,s3fd}] [-A {cv2-dnn,fan}]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model MODEL_DIR
                        Model directory. This is where the training data will be stored. You should always specify a new folder for new models. If starting a new model, select either an empty folder, or a folder
                        which does not exist (which will be created). If continuing to train an existing model, specify the location of the existing model.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Optional output directory. This is where the converted files will be saved.
  -D {cv2-dnn,mtcnn,s3fd}, --detector {cv2-dnn,mtcnn,s3fd}
                        R|Detector to use. Some of these have configurable settings in '/config/extract.ini' or 'Settings > Configure Extract 'Plugins': L|cv2-dnn: A CPU only extractor which is the least reliable
                        and least resource intensive. Use this if not using a GPU and time is important. L|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources than other GPU detectors but can
                        often return more false positives. L|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and fewer false positives than other GPU detectors, but is a lot more resource
                        intensive.
  -A {cv2-dnn,fan}, --aligner {cv2-dnn,fan}
                        R|Aligner to use. L|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, but less accurate. Only use this if not using a GPU and time is important. L|fan: Best aligner.
                        Fast on GPU, slow on CPU.
```