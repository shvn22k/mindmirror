"""
Mouth feature extraction: Mouth Aspect Ratio (MAR) and yawn detection.

MAR formula (similar to EAR):
    MAR = (|top - bottom|) / (|left - right|)

Higher MAR = mouth more open (potential yawn)
"""

import numpy as np
from .landmark_indices import MOUTH_OUTER, MOUTH_INNER


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def compute_mar(landmarks):
    """
    Compute Mouth Aspect Ratio.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        MAR value (float)
    """
    # Outer mouth corners
    top = landmarks[MOUTH_OUTER["top"]][:2]
    bottom = landmarks[MOUTH_OUTER["bottom"]][:2]
    left = landmarks[MOUTH_OUTER["left"]][:2]
    right = landmarks[MOUTH_OUTER["right"]][:2]
    
    # Also use inner mouth for more accurate vertical measurement
    top_inner = landmarks[MOUTH_INNER["top_inner"]][:2]
    bottom_inner = landmarks[MOUTH_INNER["bottom_inner"]][:2]
    
    # Vertical distance (mouth opening)
    vertical_outer = euclidean_distance(top, bottom)
    vertical_inner = euclidean_distance(top_inner, bottom_inner)
    vertical = (vertical_outer + vertical_inner) / 2.0
    
    # Horizontal distance (mouth width)
    horizontal = euclidean_distance(left, right)
    
    if horizontal == 0:
        return 0.0
    
    mar = vertical / horizontal
    return mar


def compute_mouth_features(landmarks):
    """
    Compute all mouth-related features for a single frame.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Dict with mouth features
    """
    mar = compute_mar(landmarks)
    
    return {
        "mar": mar,
    }


def is_yawn_frame(mar, threshold=0.6):
    """
    Check if a frame represents a yawn (mouth wide open).
    
    Args:
        mar: Mouth Aspect Ratio value
        threshold: MAR threshold for yawn detection
        
    Returns:
        Boolean indicating if this is a yawn frame
    """
    return mar > threshold


def detect_yawns(mar_sequence, threshold=0.6, min_frames=5):
    """
    Detect yawns from a sequence of MAR values.
    
    A yawn is detected when MAR exceeds threshold for at least min_frames.
    Yawns are typically longer than regular mouth movements.
    
    Args:
        mar_sequence: Array of MAR values (one per frame)
        threshold: MAR threshold for yawn detection
        min_frames: Minimum consecutive frames above threshold
        
    Returns:
        List of yawn events: [(start_frame, end_frame, duration_frames), ...]
    """
    is_above = mar_sequence > threshold
    yawns = []
    
    i = 0
    while i < len(is_above):
        if is_above[i]:
            start = i
            while i < len(is_above) and is_above[i]:
                i += 1
            end = i - 1
            duration = end - start + 1
            
            if duration >= min_frames:
                yawns.append((start, end, duration))
        else:
            i += 1
    
    return yawns


def compute_yawn_features(mar_sequence, fps=25, threshold=0.6):
    """
    Compute yawn statistics for a sequence of frames.
    
    Args:
        mar_sequence: Array of MAR values
        fps: Frames per second
        threshold: MAR threshold for yawn detection
        
    Returns:
        Dict with yawn statistics
    """
    yawns = detect_yawns(mar_sequence, threshold)
    
    duration_seconds = len(mar_sequence) / fps
    
    if len(yawns) == 0:
        return {
            "yawn_count": 0,
            "yawn_rate_per_min": 0.0,
            "avg_yawn_duration_ms": 0.0,
            "total_yawn_time_ms": 0.0,
        }
    
    durations_frames = [y[2] for y in yawns]
    durations_ms = [d * (1000 / fps) for d in durations_frames]
    
    yawn_rate = (len(yawns) / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    return {
        "yawn_count": len(yawns),
        "yawn_rate_per_min": yawn_rate,
        "avg_yawn_duration_ms": np.mean(durations_ms),
        "total_yawn_time_ms": np.sum(durations_ms),
    }
