"""Single source of truth for colours, fonts, and sizes (Phase 4 design).

Hex is canonical. cv2 draws in BGR, so hex_to_bgr() derives the tuples the
overlay needs; the Streamlit CSS uses the hex values directly. Change a value
here and it updates both the on-image overlay and the page chrome.
"""

# Palette — black, quiet, editorial. One accent, used only on the primary face.
PALETTE = {
    "bg":      "#0A0A0A",   # page background
    "text":    "#EDEDED",   # primary off-white
    "muted":   "#777777",   # secondary / inactive
    "faint":   "#555555",   # disclaimer, hairlines
    "divider": "#2A2A2A",   # 1px dividers, meter track
    "grey":    "#8A8A8A",   # non-primary face brackets/labels
    "accent":  "#4A9EFF",   # primary face brackets + meter fill ONLY
}

FONTS = {
    "ui":   "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "mono": "'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace",
}

SIZES = {
    "max_width": 640,           # narrow centered layout (px)
    "bracket_len_ratio": 0.20,  # corner bracket = 20% of each side
    "bracket_inset": 2,         # px inset from the detected box
    "label_scale": 0.45,        # cv2 top-label font scale
    "label_tracking": 2,        # px letter-spacing for the label
    "bar_scale": 0.9,           # cv2 bottom-bar font scale
    "meter_thick": 2,           # px meter line thickness
}


def hex_to_bgr(h):
    """'#4A9EFF' -> (255, 158, 74) for cv2 (BGR)."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
