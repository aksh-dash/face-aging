"""Shared drawing + stats for both app modes (upload and live).

draw_overlay() renders the same annotations in both modes: thin corner
brackets around each face, an uppercase top-corner label, and a bottom
status bar for the primary (largest) face. Only the primary face's
brackets use the accent colour; all others are grey.

Colours/sizes live in THEME so styling stays in one place (Phase 4 refines
the fine editorial details: meter lines, letter-spacing, exact insets).
All frames are BGR (OpenCV convention).
"""

import cv2

from theme import PALETTE, SIZES, hex_to_bgr

# Overlay colours derived once from the single theme (BGR for cv2).
C = {k: hex_to_bgr(v) for k, v in PALETTE.items()}

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN   # stand-in for monospace numerals
LT = cv2.LINE_AA

def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def tag_primary(faces):
    """Mark the largest face as primary (upload mode; the tracker already
    sets is_primary in live mode). Mutates and returns the list."""
    if faces:
        largest = max(range(len(faces)), key=lambda i: _area(faces[i]["box"]))
        for i, f in enumerate(faces):
            f["is_primary"] = (i == largest)
    return faces


def _brackets(img, box, color, len_ratio=None, inset=None):
    """Draw four thin 1px corner brackets (~len_ratio of each side), inset."""
    len_ratio = SIZES["bracket_len_ratio"] if len_ratio is None else len_ratio
    inset = SIZES["bracket_inset"] if inset is None else inset
    x1, y1, x2, y2 = box
    x1 += inset; y1 += inset; x2 -= inset; y2 -= inset
    lx, ly = int((x2 - x1) * len_ratio), int((y2 - y1) * len_ratio)
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, 1, LT)   # top-left
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, 1, LT)
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, 1, LT)   # top-right
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, 1, LT)
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, 1, LT)   # bottom-left
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, 1, LT)
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, 1, LT)   # bottom-right
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, 1, LT)


def _put_spaced(img, text, org, scale, color, tracking):
    """Draw uppercase, letter-spaced text char by char. Returns end x."""
    x, y = org
    for ch in text.upper():
        cv2.putText(img, ch, (x, y), FONT, scale, color, 1, LT)
        (cw, _), _ = cv2.getTextSize(ch, FONT, scale, 1)
        x += cw + tracking
    return x


def _label(img, box, text, color):
    x1, y1, _, _ = box
    _put_spaced(img, text, (x1 + 2, max(12, y1 - 6)),
                SIZES["label_scale"], color, SIZES["label_tracking"])


def _status_bar(img, stats):
    """Bottom bar: 1px divider, monospaced text, tiny meter lines for probs."""
    h, w = img.shape[:2]
    scale, thick = SIZES["bar_scale"], SIZES["meter_thick"]
    text_y, div_y, meter_y = h - 12, h - 28, h - 6

    cv2.line(img, (16, div_y), (w - 16, div_y), C["divider"], 1, LT)

    # (text, value-or-None) — value drives a meter under that segment.
    segs = [("FACES %02d" % stats["n_faces"], None)]
    if stats.get("face_conf") is not None:
        segs += [("FACE %.2f" % stats["face_conf"], stats["face_conf"]),
                 ("GENDER %.2f" % stats["gender_prob"], stats["gender_prob"]),
                 ("AGE %.2f" % stats["age_prob"], stats["age_prob"])]
    if stats.get("fps") is not None:
        segs.append(("FPS %.0f" % stats["fps"], None))

    x, gap = 16, 22
    for text, val in segs:
        cv2.putText(img, text, (x, text_y), MONO, scale, C["text"], 1, LT)
        (tw, _), _ = cv2.getTextSize(text, MONO, scale, 1)
        if val is not None:                       # meter: muted track + accent fill
            cv2.line(img, (x, meter_y), (x + tw, meter_y), C["divider"], thick, LT)
            cv2.line(img, (x, meter_y), (x + int(tw * val), meter_y), C["accent"], thick, LT)
        x += tw + gap


def build_stats(faces, fps=None):
    """Bottom-bar stats for the primary (largest) face."""
    stats = {"n_faces": len(faces), "fps": fps,
             "face_conf": None, "gender_prob": None, "age_prob": None}
    primary = next((f for f in faces if f.get("is_primary")), None)
    if primary is None and faces:
        primary = max(faces, key=lambda f: _area(f["box"]))
    if primary is not None:
        stats["face_conf"] = primary.get("confidence")
        stats["gender_prob"] = primary.get("gender_prob")
        stats["age_prob"] = primary.get("age_prob")
    return stats


def draw_overlay(frame, faces, stats):
    """Annotate a writable BGR frame in place and return it. Used by BOTH modes.

    faces: dicts with box, gender, age_bucket, is_primary, optional id/probs.
    stats: dict from build_stats() — n_faces + primary probs + optional fps.
    """
    for f in faces:
        color = C["accent"] if f.get("is_primary") else C["grey"]
        _brackets(frame, f["box"], color)
        label = "%s  %s" % (f.get("gender", "-"), f.get("age_bucket", "-"))
        if f.get("id") is not None:
            label = "%s  %02d" % (label, f["id"])
        _label(frame, f["box"], label, color)
    _status_bar(frame, stats)
    return frame
