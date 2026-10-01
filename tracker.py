"""Centroid-based face tracker with EMA box smoothing and label majority-vote.

Designed for live camera mode in the Face Analysis app.
- Persistent IDs by nearest-centroid matching with a distance limit.
- Exponential moving average on bounding boxes to reduce jitter.
- Rolling majority vote on gender/age labels to prevent flickering.
- Throttled age/gender predictions (every N frames per face) for performance.

Not used for upload-photo mode (single image, no tracking needed).
"""

from collections import deque, OrderedDict, Counter
import numpy as np


class FaceTracker:
    """Simple centroid tracker with smoothing for live face analysis."""

    def __init__(self, max_disappeared=15, max_distance=100,
                 ema_alpha=0.3, label_history_size=10, predict_interval=5):
        """
        Args:
            max_disappeared: Drop a track after this many consecutive missed frames.
            max_distance:    Max centroid distance (px) to match a detection to a track.
            ema_alpha:       EMA factor for box smoothing (0 = fully old, 1 = fully new).
            label_history_size: How many recent predictions to keep per track for smoothing.
            predict_interval:   Run age/gender inference every N frames per face.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.ema_alpha = ema_alpha
        self.label_history_size = label_history_size
        self.predict_interval = predict_interval

        self.next_id = 1
        self.tracks = OrderedDict()  # id -> track dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all tracks (e.g. when switching from live to upload mode)."""
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections, frame_bgr, analyzer):
        """Update tracks with new detections and return tracked faces.

        Args:
            detections: list of {box, confidence} from FaceAnalyzer.detect_faces().
            frame_bgr:  current frame in BGR (for cropping + running predictions).
            analyzer:   FaceAnalyzer instance.

        Returns:
            list of dicts, each with:
                id:          int  (stable, persistent across frames)
                box:         (x1, y1, x2, y2) — EMA-smoothed
                confidence:  float (face-detector score)
                gender:      str
                gender_prob: float
                age_bucket:  str
                age_prob:    float
                is_primary:  bool (True for the largest face)
        """
        # --- Edge case: no detections this frame ---
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    self._deregister(tid)
            return self._build_output()

        # --- Edge case: no existing tracks ---
        if len(self.tracks) == 0:
            for det in detections:
                self._register(det["box"], det["confidence"])
            self._run_predictions(frame_bgr, analyzer)
            return self._build_output()

        # --- Match detections to existing tracks by nearest centroid ---
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self._centroid(self.tracks[tid]["box"]) for tid in track_ids]
        )
        det_centroids = np.array(
            [self._centroid(d["box"]) for d in detections]
        )

        # Distance matrix: (num_tracks, num_detections)
        diff = track_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # Greedy matching — smallest distance first
        matched_tracks = set()
        matched_dets = set()

        flat_order = np.argsort(dist_matrix, axis=None)
        num_dets = len(detections)

        for flat_idx in flat_order:
            t_idx = int(flat_idx // num_dets)
            d_idx = int(flat_idx % num_dets)

            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            if dist_matrix[t_idx, d_idx] > self.max_distance:
                continue

            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

            # Update the matched track
            tid = track_ids[t_idx]
            det = detections[d_idx]
            track = self.tracks[tid]

            # EMA smooth the bounding box
            old_box = np.array(track["box"], dtype=np.float64)
            new_box = np.array(det["box"], dtype=np.float64)
            smoothed = self.ema_alpha * new_box + (1.0 - self.ema_alpha) * old_box
            track["box"] = tuple(int(round(v)) for v in smoothed)

            track["confidence"] = det["confidence"]
            track["disappeared"] = 0
            track["frames_since_pred"] += 1

        # Unmatched tracks → increment disappeared, maybe deregister
        for t_idx in range(len(track_ids)):
            if t_idx not in matched_tracks:
                tid = track_ids[t_idx]
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched detections → register as new tracks
        for d_idx in range(len(detections)):
            if d_idx not in matched_dets:
                self._register(detections[d_idx]["box"], detections[d_idx]["confidence"])

        # Run throttled predictions
        self._run_predictions(frame_bgr, analyzer)

        return self._build_output()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(box):
        """Center point (cx, cy) of a bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @staticmethod
    def _area(box):
        """Area (w * h) of a bounding box."""
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _register(self, box, confidence):
        """Create a new track with the next available ID."""
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "box": tuple(box),
            "confidence": confidence,
            "disappeared": 0,
            # Force immediate prediction on first appearance
            "frames_since_pred": self.predict_interval,
            # Label history deques (fixed-size sliding windows)
            "gender_history":      deque(maxlen=self.label_history_size),
            "age_history":         deque(maxlen=self.label_history_size),
            "gender_prob_history": deque(maxlen=self.label_history_size),
            "age_prob_history":    deque(maxlen=self.label_history_size),
            # Current smoothed values (displayed to user)
            "gender": "\u2014",       # em-dash placeholder until first prediction
            "gender_prob": 0.0,
            "age_bucket": "\u2014",
            "age_prob": 0.0,
        }

    def _deregister(self, tid):
        """Remove a track by ID."""
        del self.tracks[tid]

    def _run_predictions(self, frame_bgr, analyzer):
        """Run age/gender inference on tracks that are due (every predict_interval frames)."""
        for tid, track in self.tracks.items():
            # Skip faces not seen this frame — don't run inference on a stale box
            if track["disappeared"] > 0:
                continue
            if track["frames_since_pred"] >= self.predict_interval:
                crop = analyzer._pad_and_crop(frame_bgr, track["box"])
                if crop is not None:
                    attrs = analyzer.predict_face_attributes(crop)
                    track["gender_history"].append(attrs["gender"])
                    track["age_history"].append(attrs["age_bucket"])
                    track["gender_prob_history"].append(attrs["gender_prob"])
                    track["age_prob_history"].append(attrs["age_prob"])
                    self._smooth_labels(track)
                track["frames_since_pred"] = 0

    def _smooth_labels(self, track):
        """Update smoothed gender/age from history: majority vote + mean probability."""
        if track["gender_history"]:
            counter = Counter(track["gender_history"])
            track["gender"] = counter.most_common(1)[0][0]
            track["gender_prob"] = float(np.mean(track["gender_prob_history"]))

        if track["age_history"]:
            counter = Counter(track["age_history"])
            track["age_bucket"] = counter.most_common(1)[0][0]
            track["age_prob"] = float(np.mean(track["age_prob_history"]))

    def _build_output(self):
        """Return list of visible (non-disappeared) tracks with is_primary flag."""
        output = []
        for tid, track in self.tracks.items():
            if track["disappeared"] > 0:
                continue  # Don't show faces that are currently missing
            output.append({
                "id": track["id"],
                "box": track["box"],
                "confidence": track["confidence"],
                "gender": track["gender"],
                "gender_prob": track["gender_prob"],
                "age_bucket": track["age_bucket"],
                "age_prob": track["age_prob"],
            })

        # Mark the largest face as primary (for bottom status bar)
        if output:
            largest_idx = max(range(len(output)),
                              key=lambda i: self._area(output[i]["box"]))
            for i, face in enumerate(output):
                face["is_primary"] = (i == largest_idx)

        return output
