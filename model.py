import cv2
import numpy as np
import os

class FaceAgingModel:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load face detection model
        self.face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(BASE_DIR, "models", "deploy.prototxt"),
            os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        )

        # Load age prediction model
        self.age_net = cv2.dnn.readNetFromCaffe(
            os.path.join(BASE_DIR, "models", "age_deploy.prototxt"),
            os.path.join(BASE_DIR, "models", "age_net.caffemodel")
        )

        self.AGE_BUCKETS = [
            "(0-2)", "(4-6)", "(8-12)", "(15-20)",  "(20-25)", 
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"
        ]

    def detect_faces(self, img):
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Safe bounding box
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                faces.append((startX, startY, endX, endY))

        return faces

    def predict_age(self, img, face_box):
        (startX, startY, endX, endY) = face_box

        face = img[startY:endY, startX:endX]

        if face.size == 0:
            return "Unknown"

        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.426, 87.768, 114.895)
        )

        self.age_net.setInput(face_blob)
        preds = self.age_net.forward()

        age = self.AGE_BUCKETS[preds[0].argmax()]
        return age

    def apply_aging_effect(self, img, face_box):
        (startX, startY, endX, endY) = face_box

        face = img[startY:endY, startX:endX]

        if face.size == 0:
            return img

        # 1 Skin dullness
        face = cv2.convertScaleAbs(face, alpha=0.9, beta=-20)

        # 2 Blur (aging texture)
        face = cv2.GaussianBlur(face, (5, 5), 0)

        # 3 Add noise (wrinkle simulation)
        noise = np.random.normal(0, 10, face.shape).astype("uint8")
        face = cv2.add(face, noise)

        # 4 Slight contrast boost
        face = cv2.convertScaleAbs(face, alpha=1.2, beta=10)

        # Put back into image
        img[startY:endY, startX:endX] = face

        return img

    def predict(self, img):
        # Handle RGBA → RGB
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert RGB → BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = self.detect_faces(img_bgr)

        ages = []
        output_img = img_bgr.copy()

        for face_box in faces:
            # Predict age
            age = self.predict_age(img_bgr, face_box)
            ages.append(age)

            # Apply aging effect
            output_img = self.apply_aging_effect(output_img, face_box)

            # Draw bounding box + label
            (startX, startY, endX, endY) = face_box
            cv2.rectangle(output_img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(output_img, age, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert back BGR → RGB
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        return output_rgb, ages