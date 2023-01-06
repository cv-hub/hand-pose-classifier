from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

handnet = load_model("handpose_classifier.model")

cap = cv2.VideoCapture(0)

SCREEN_SIZE = (640, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[1])

startX, startY, endX, endY = 0, 0, 250, 250

while cap.isOpened():

    ret, frame = cap.read()
    # frame = cv2.resize(frame, (300, 400))

    if not ret:
        print("No frame")
        break

    hand = frame[startY:endY, startX:endX]
    hands = []
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    hand = cv2.resize(hand, (224, 224))
    hand = preprocess_input(hand)

    hands.append(hand)

    hands = np.array(hands, dtype="float32")
    preds = handnet.predict_on_batch(hands)

    for pred in preds:
        pred_prono, pred_supino = pred
        p = "pronus" if pred_prono > pred_supino else "supine"
        print(pred_prono, pred_supino, p)

    cv2.rectangle(frame, (startX, startY, endX, endY), (0, 200, 0), 5)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(500) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
