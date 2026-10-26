from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Nie mogę otworzyć kamerki - sprawdź uprawnienia w Windows.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    result = model(frame, device=0, verbose=False)
    annotated = result[0].plot()
    cv2.imshow("Kamera", annotated)
    print(result[0].boxes) 

    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cap.release()
cv2.destroyAllWindows()
