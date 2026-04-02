"""
MediaPipe FaceMesh landmark indices for feature extraction.

Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

Total landmarks: 478
- Face mesh: 468 landmarks (indices 0-467)
- Left iris: 5 landmarks (indices 468-472)
- Right iris: 5 landmarks (indices 473-477)
"""

# === EYE LANDMARKS ===

# Left eye contour (6 points for EAR calculation)
LEFT_EYE = {
    "p1": 33,   # Left corner
    "p2": 160,  # Upper lid (outer)
    "p3": 158,  # Upper lid (inner)
    "p4": 133,  # Right corner
    "p5": 153,  # Lower lid (inner)
    "p6": 144,  # Lower lid (outer)
}

# Right eye contour (6 points for EAR calculation)
RIGHT_EYE = {
    "p1": 362,  # Right corner
    "p2": 385,  # Upper lid (outer)
    "p3": 387,  # Upper lid (inner)
    "p4": 263,  # Left corner
    "p5": 373,  # Lower lid (inner)
    "p6": 380,  # Lower lid (outer)
}

# Left eye full contour (for visualization)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Right eye full contour (for visualization)
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


# === IRIS LANDMARKS ===

# Left iris (5 points: center + 4 cardinal points)
LEFT_IRIS = {
    "center": 468,
    "right": 469,
    "top": 470,
    "left": 471,
    "bottom": 472,
}

# Right iris (5 points: center + 4 cardinal points)
RIGHT_IRIS = {
    "center": 473,
    "right": 474,
    "top": 475,
    "left": 476,
    "bottom": 477,
}


# === MOUTH LANDMARKS ===

# Outer mouth contour
MOUTH_OUTER = {
    "top": 13,      # Upper lip center
    "bottom": 14,   # Lower lip center
    "left": 61,     # Left corner
    "right": 291,   # Right corner
}

# Inner mouth (for MAR calculation)
MOUTH_INNER = {
    "top_outer": 13,
    "top_inner": 82,
    "bottom_inner": 87,
    "bottom_outer": 14,
    "left": 78,
    "right": 308,
}

# Vertical mouth landmarks for MAR
MOUTH_VERTICAL = [13, 14, 82, 87, 312, 317, 78, 308]


# === NOSE LANDMARKS ===

NOSE = {
    "tip": 1,
    "bridge": 6,
    "bottom": 2,
    "left_nostril": 129,
    "right_nostril": 358,
}


# === FACE OVAL / HEAD POSE ===

# Key points for head pose estimation
HEAD_POSE_POINTS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Face center approximation (between eyes)
FACE_CENTER = {
    "left_eye_inner": 133,
    "right_eye_inner": 362,
}


# === EYEBROW LANDMARKS ===

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295]


# === HELPER FUNCTIONS ===

def get_eye_landmarks_for_ear():
    """Get landmark indices needed for EAR calculation."""
    left = [LEFT_EYE["p1"], LEFT_EYE["p2"], LEFT_EYE["p3"], 
            LEFT_EYE["p4"], LEFT_EYE["p5"], LEFT_EYE["p6"]]
    right = [RIGHT_EYE["p1"], RIGHT_EYE["p2"], RIGHT_EYE["p3"],
             RIGHT_EYE["p4"], RIGHT_EYE["p5"], RIGHT_EYE["p6"]]
    return left, right


def get_mouth_landmarks_for_mar():
    """Get landmark indices needed for MAR calculation."""
    return [
        MOUTH_OUTER["top"],
        MOUTH_OUTER["bottom"],
        MOUTH_OUTER["left"],
        MOUTH_OUTER["right"],
        MOUTH_INNER["top_inner"],
        MOUTH_INNER["bottom_inner"],
    ]
