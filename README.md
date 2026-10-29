# vision

Nauka OpenCV i YOLO: kamerka z laptopa wykrywa telefon i twarz na żywo.

## Pliki

- `01_kamera.py`: podgląd z kamerki (`q` wychodzi)
- `02_telefon.py`: YOLO (`yolo11n.pt`) rysuje ramki na obrazie z kamerki
- `03_twarz.py`: wykrywanie twarzy Haar cascade (OpenCV)
- `02_and_03.py`: telefon (YOLO) i twarz (Haar) naraz, z licznikiem FPS i pomiarem czasu etapów
- `04_wideo.py`: to samo co wyżej, ale na pliku wideo zamiast kamerki. Zapisuje nagranie z ramkami i pomiarami czasu (`python 04_wideo.py wejscie.mp4 wyjscie.mp4`)
- `Dockerfile`, `.dockerignore`: uruchamianie `04_wideo.py` w kontenerze Dockera

## Jak uruchomić

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python 02_and_03.py
```

- Linia z `torch` to PyTorch z GPU (CUDA 12.8, RTX 5060). Bez karty NVIDIA: zwykłe `pip install torch`.
- OpenCV musi być 4.x (`opencv-python<5` w `requirements.txt`). Wersja 5.0 nie ma `CascadeClassifier`.
- Wagi `yolo11n.pt` pobierają się same przy pierwszym uruchomieniu.

## Docker

Dockeryzacja `04_wideo.py` pomogła mi nauczyć się Dockera: obrazów (image), warstw i cache, `.dockerignore`, przekazywania GPU do kontenera i montowania folderów (`-v`).

```
docker build -t vision .
docker run --rm --gpus all -v ${PWD}:/data vision /data/wejscie.mp4 /data/wyjscie.mp4
```

- `--gpus all` jest potrzebne, bo kod używa `device=0`. Wymaga Docker Desktop z WSL2 i sterownika NVIDIA.
- `-v ${PWD}:/data` montuje bieżący folder do kontenera. Dzięki temu kontener widzi plik wejściowy i zapisuje wynik na dysku.
- Plików `.mp4` nie ma w repo (`.gitignore`), trzeba dodać własne.
- Obraz jest duży (kilka GB), bo zawiera PyTorch z CUDA. Na Jetsonie (ARM) ten Dockerfile się nie zbuduje, bo potrzebne są tam inne paczki.
- Obserwacja: FPS nagrania zrobionego w Dockerze był około 2 razy niższy niż bez Dockera. Przyczyny jeszcze sprawdzam (podejrzewam odczyt i zapis plików przez montowany folder).

## Wyniki (2026-10-26)

RTX 5060 Laptop, obraz 640x480, czas jednego etapu na klatkę:

| Etap | Przed | Po  dwukrotnym zmniejszeniu obrazu dla Haara |
|---|---|---|
| YOLO (tylko telefon) | 13,8 ms | 10-20 ms |
| Haar (twarz) | 54,6 ms | ~9 ms |
| Cała pętla | ~13 FPS | 22-32 FPS |

- Haar przyspieszył około 6 razy po dwóch zmianach naraz: obraz zmniejszony 2x w każdą stronę (`cv2.resize`, `fx=0.5`) i `minSize=(40,40)`.
- Teraz limitem jest kamerka (około 30 FPS), bo `cap.read()` czeka na następną klatkę.
- Wyniki YOLO skaczą między próbami (10 i 20 ms), przyczyny nie sprawdzałem.

## Jakość

- **Telefon:** działa, ale `conf` spada, gdy telefon jest bardzo blisko kamery. Najlepiej z 30-60 cm.
- **Twarz:** Haar wykrywa słabo (tylko twarz z przodu, wrażliwy na światło).

