import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('mask_detector.model')
model.save('mask_detector.keras')
model = tf.keras.models.load_model('mask_detector.keras')

IMG_SIZE = 100
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face / 255.0, axis=0)
    pred = model.predict(face)[0][0]
    
    label = "Mask" if pred < 0.5 else "No Mask"
    color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 2)
    
    cv2.imshow('Mask Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
