import cv2
import numpy as np
import os


class FaceAnalyzer:
    """Loads pretrained Caffe models for face detection, age and gender estimation.

    Models used (all pretrained, not trained by the author):
    - Face detector: res10_300x300_ssd_iter_140000 (OpenCV SSD)
    - Age classifier: age_net (Levi & Hassner, 9 buckets)
    - Gender classifier: gender_net (Levi & Hassner, binary)
    """

    # 8 buckets — must match the age_net output layer (8 classes).
    # These are the canonical Levi & Hassner age ranges, in model output order.
    AGE_BUCKETS = [
        "(0-2)", "(4-6)", "(8-12)", "(15-20)",
        "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ]
    GENDER_LIST = ["Male", "Female"]
    MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    MIN_FACE_SIZE = 40        # Skip face crops smaller than this (pixels)
    FACE_PAD_RATIO = 0.40     # Pad crop heavily for context (hair/neck) required by gender model
    FACE_CONF_THRESHOLD = 0.5 # Minimum face-detector confidence

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_dir = base_dir

        self._model_files = {
            "face_proto":  os.path.join(base_dir, "models", "deploy.prototxt"),
            "face_model":  os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel"),
            "age_proto":   os.path.join(base_dir, "models", "age_deploy.prototxt"),
            "age_model":   os.path.join(base_dir, "models", "age_net.caffemodel"),
            "gender_proto": os.path.join(base_dir, "models", "gender_deploy.prototxt"),
            "gender_model": os.path.join(base_dir, "models", "gender_net.caffemodel"),
        }

        # Check all files exist before attempting to load
        missing = self.check_model_files()
        if missing:
            raise FileNotFoundError(
                f"Missing model files in models/: {', '.join(missing)}"
            )

        # Load face detection network (SSD, pretrained)
        self.face_net = cv2.dnn.readNetFromCaffe(
            self._model_files["face_proto"],
            self._model_files["face_model"],
        )

        # Load age prediction network (pretrained, Levi & Hassner)
        self.age_net = cv2.dnn.readNetFromCaffe(
            self._model_files["age_proto"],
            self._model_files["age_model"],
        )

        # Load gender prediction network (pretrained, Levi & Hassner)
        self.gender_net = cv2.dnn.readNetFromCaffe(
            self._model_files["gender_proto"],
            self._model_files["gender_model"],
        )

    def check_model_files(self):
        """Return list of missing model file basenames."""
        missing = []
        for name, path in self._model_files.items():
            if not os.path.isfile(path):
                missing.append(os.path.basename(path))
        return missing

    def detect_faces(self, img_bgr, threshold=None):
        """Detect faces in a BGR image.

        Args:
            img_bgr: image in BGR.
            threshold: optional per-call detector confidence cutoff. Falls back
                to FACE_CONF_THRESHOLD. Passed explicitly (not stored) so the
                live slider never mutates the shared cached analyzer.

        Returns list of dicts with keys:
            box: (x1, y1, x2, y2) — clamped to image bounds
            confidence: float — face-detector score (NOT age/gender confidence)

        Faces smaller than MIN_FACE_SIZE on either axis are skipped.
        """
        thr = self.FACE_CONF_THRESHOLD if threshold is None else threshold
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < thr:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip tiny faces — unreliable predictions
            if (x2 - x1) < self.MIN_FACE_SIZE or (y2 - y1) < self.MIN_FACE_SIZE:
                continue

            faces.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
            })

        return faces

    def _pad_and_crop(self, img_bgr, box):
        """Pad the bounding box by FACE_PAD_RATIO on each side and crop.

        Extra context around the face improves age/gender classifier accuracy.
        Returns the cropped face (BGR) or None if the crop is invalid.
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = box
        fw, fh = x2 - x1, y2 - y1

        pad_x = int(fw * self.FACE_PAD_RATIO)
        pad_y = int(fh * self.FACE_PAD_RATIO)

        # Expand with padding, clamped to image bounds
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        crop = img_bgr[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        return crop

    def predict_face_attributes(self, face_crop_bgr):
        """Run age and gender classifiers on a single face crop (BGR).

        Both nets use the same 227x227 input size and mean values.

        Returns dict:
            gender: str ("Male" or "Female")
            gender_prob: float (softmax max — classifier confidence)
            age_bucket: str (e.g. "(25-32)")
            age_prob: float (softmax max — classifier confidence)
        """
        blob = cv2.dnn.blobFromImage(
            face_crop_bgr, 1.0, (227, 227), self.MEAN_VALUES
        )

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender_idx = int(gender_preds[0].argmax())
        gender = self.GENDER_LIST[gender_idx]
        gender_prob = float(gender_preds[0].max())

        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_idx = int(age_preds[0].argmax())
        age_bucket = self.AGE_BUCKETS[age_idx]
        age_prob = float(age_preds[0].max())

        return {
            "gender": gender,
            "gender_prob": gender_prob,
            "age_bucket": age_bucket,
            "age_prob": age_prob,
        }

    def analyze_image(self, img_rgb, threshold=None):
        """Full pipeline: detect faces, predict attributes for each.

        Args:
            img_rgb: Input image in RGB format (as from PIL / st.camera_input).
            threshold: optional detector confidence cutoff (see detect_faces).

        Returns:
            List of dicts, each with:
                box: (x1, y1, x2, y2)
                confidence: float (face-detector score — separate from age/gender)
                gender: str
                gender_prob: float
                age_bucket: str
                age_prob: float
        """
        # Handle RGBA → RGB
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        detected = self.detect_faces(img_bgr, threshold)

        results = []
        for face in detected:
            crop = self._pad_and_crop(img_bgr, face["box"])
            if crop is None:
                continue

            attrs = self.predict_face_attributes(crop)
            results.append({
                "box": face["box"],
                "confidence": face["confidence"],
                **attrs,
            })

        return results