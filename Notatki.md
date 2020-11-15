# Praca inżynierska

We wstępie zawrzeć opis dlaczego jest to ciekawe i czemu takie ważne i jakie zagrożenia niosą za sobą deepfaki. Dodać ogólny opis aplikacji, uwzględnić, że to proof of concept istniejących zagrożeń, zwłąszcza w dobie zdalnej komunikacji czy weryfikacji tożsamości.
Dodać o istniejących rozwiązaniach.
Zawrzeć podstawy sztucznej inteligencji i deep learningu.

## Dostępne narzędzia

| Narzędzie | Link | Główne cechy |
| --- | --- | --- |
| Faceswap | https://github.com/deepfakes/faceswap | Używa dwóch par koder-dekoder. Parametry kodera są wspólne. |
| Faceswap-GAN | https://github.com/shaoanlu/faceswap-GAN | Adversial loss i perceptual loss (VGGface) są dodane do architektury autoencoderowej. |
| Few-Shot Face Translation GAN | https://github.com/shaoanlu/fewshot-face-translation-GAN | Używa wytrenowanego modelu rozpoznawania twarzy do wydobycia latent embeddings z GAN processing.   |
| DeepFaceLab | https://github.com/iperov/DeepFaceLab | Rozszerze metody z Faceswap nowymi modelami, np. H64, H128, LIAEFI128, SAE. Zapewnia tryb wykrycia wielu twarzy np. S3FD, MTCNN, dlib lub manulanie  |
| DFaker | https://github.com/dfaker/df | Funkcja strat  DSSIM jest użyta do rekonstrukcji twarzy. Implementacja oparta na bibliotece Keras. |
| Deepfakes web β | https://deepfakesweb.com/ | Komercyjna strona do podmiany twarzy używająca algorytmów uczenia głębokiego. |
