import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("nie dziala kamerka")

while True:
    ok, frame = cap.read()  
    if not ok:
        break

    frame = cv2.flip(frame, 1) 
    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()
