from ultralytics import YOLO
import cv2
import time
import sys

model = YOLO("yolo11n.pt")
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if len(sys.argv) != 3:
    raise SystemExit("użycie: python 04_wideo.py wideo.mp4 wyjscie.mp4")

wejscie = sys.argv[1]
wyjscie = sys.argv[2]

cap = cv2.VideoCapture(wejscie)

if not cap.isOpened():
    raise SystemExit("nie laduje pliku")
prev = time.perf_counter()

fpsw = cap.get(propId=cv2.CAP_PROP_FPS)
szer = int(cap.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
wys = int(cap.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(wyjscie, cv2.VideoWriter_fourcc(*"mp4v"), fpsw, (szer, wys))

print(f"{fpsw},   {wys},   {szer} \n")
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    t1 = time.perf_counter()
    fps1 = -(t0-t1)*1000
    
    if not ok:
        break
    cv2.putText(img=frame, org=(10,60), text=f"ms read: {fps1:.2f}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)
    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    result = model(frame, device=0, verbose=False, classes=[67])
    t1 = time.perf_counter()
    fps2 = -(t0-t1)*1000
    cv2.putText(img=frame, org=(10,80), text=f"ms model: {fps2:.2f}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)
    
    for box in result[0].boxes:
        xy = box.xyxy[0].cpu().numpy().astype(int)
        con = float(box.conf[0])
        cv2.rectangle(frame, (xy[0],xy[1]), (xy[2], xy[3]), (255,0,0), 2)
        cv2.putText(img=frame, org=(xy[0], xy[1] - 10), text=f"telefon {con:.2f}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)

    t0 = time.perf_counter()
    gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40,40))
    t1 = time.perf_counter()
    fps3 = -(t0-t1)*1000
    cv2.putText(img=frame, org=(10,100), text=f"ms faces: {fps3:.4f}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (2*x,2*y), (2*x+2*w, 2*y+2*h), (0,255,0), 2)
        cv2.putText(img=frame, org=(2*x,2*y-10), text="twarz",fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
    now = time.perf_counter()
    fps = 1/(now-prev)
    prev = now
    cv2.putText(img=frame, org=(10,20), text=f"fps: {int(fps)}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)
    t0 = time.perf_counter()
    writer.write(frame)
    t1 = time.perf_counter()
    ms = (t1 - t0)*1000
    print(f"czas zapisu {ms}")

cap.release()
writer.release()