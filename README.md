# vision

Nauka OpenCV i YOLO: kamerka z laptopa wykrywa telefon i twarz na żywo. Cel końcowy: ramka na własnej ręce.

## Pliki

- `01_kamera.py`: podgląd z kamerki (`q` wychodzi)
- `02_telefon.py`: YOLO (`yolo11n.pt`) rysuje ramki na obrazie z kamerki
- `03_twarz.py`: wykrywanie twarzy Haar cascade (OpenCV)
- `02_and_03.py`: telefon (YOLO) i twarz (Haar) naraz, z licznikiem FPS i pomiarem czasu etapów

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
- **Twarz:** Haar wykrywa słabo (tylko twarz z przodu, wrażliwy na światło). mozna sprawdzic rozne `minNeighbors` (3-5), niższe `scaleFactor` (1.05).

