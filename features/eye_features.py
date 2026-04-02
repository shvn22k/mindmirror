"""
Eye-based feature extraction: Eye Aspect Ratio (EAR) and blink detection.

EAR formula:
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

Where p1-p6 are the 6 eye landmarks:
         p2    p3
          \    /
      p1 ------- p4
          /    \
         p6    p5
"""

import numpy as np
from .landmark_indices import LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def compute_ear_single_eye(landmarks, eye_indices):
    """
    Compute Eye Aspect Ratio for a single eye.
    
    Args:
        landmarks: Array of shape (478, 3) with landmark coordinates
        eye_indices: Dict with keys p1-p6 containing landmark indices
        
    Returns:
        EAR value (float)
    """
    p1 = landmarks[eye_indices["p1"]][:2]  # Use only x, y
    p2 = landmarks[eye_indices["p2"]][:2]
    p3 = landmarks[eye_indices["p3"]][:2]
    p4 = landmarks[eye_indices["p4"]][:2]
    p5 = landmarks[eye_indices["p5"]][:2]
    p6 = landmarks[eye_indices["p6"]][:2]
    
    # Vertical distances
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    
    # Horizontal distance
    horizontal = euclidean_distance(p1, p4)
    
    if horizontal == 0:
        return 0.0
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def compute_ear(landmarks):
    """
    Compute Eye Aspect Ratio for both eyes.
    
    Args:
        landmarks: Array of shape (478, 3) with landmark coordinates
        
    Returns:
        Tuple of (ear_left, ear_right, ear_avg)
    """
    ear_left = compute_ear_single_eye(landmarks, LEFT_EYE)
    ear_right = compute_ear_single_eye(landmarks, RIGHT_EYE)
    ear_avg = (ear_left + ear_right) / 2.0
    
    return ear_left, ear_right, ear_avg


def compute_eye_features(landmarks):
    """
    Compute all eye-related features for a single frame.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Dict with ear_left, ear_right, ear_avg
    """
    ear_left, ear_right, ear_avg = compute_ear(landmarks)
    
    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_avg": ear_avg,
    }


def detect_blinks(ear_sequence, threshold=0.2, min_frames=1):
    """
    Detect blinks from a sequence of EAR values.
    
    A blink is detected when EAR drops below threshold for at least min_frames.
    
    Args:
        ear_sequence: Array of EAR values (one per frame)
        threshold: EAR threshold for blink detection (default 0.2)
        min_frames: Minimum consecutive frames below threshold
        
    Returns:
        List of blink events: [(start_frame, end_frame, duration_frames), ...]
    """
    is_below = ear_sequence < threshold
    blinks = []
    
    i = 0
    while i < len(is_below):
        if is_below[i]:
            start = i
            while i < len(is_below) and is_below[i]:
                i += 1
            end = i - 1
            duration = end - start + 1
            
            if duration >= min_frames:
                blinks.append((start, end, duration))
        else:
            i += 1
    
    return blinks


def compute_blink_features_for_sequence(ear_sequence, fps=25, threshold=0.2):
    """
    Compute blink statistics for a sequence of frames.
    
    Args:
        ear_sequence: Array of EAR values
        fps: Video frames per second (for rate calculation)
        threshold: EAR threshold for blink detection
        
    Returns:
        Dict with blink statistics
    """
    blinks = detect_blinks(ear_sequence, threshold)
    
    duration_seconds = len(ear_sequence) / fps
    
    if len(blinks) == 0:
        return {
            "blink_count": 0,
            "blink_rate_per_min": 0.0,
            "avg_blink_duration_ms": 0.0,
            "max_blink_duration_ms": 0.0,
        }
    
    durations_frames = [b[2] for b in blinks]
    durations_ms = [d * (1000 / fps) for d in durations_frames]
    
    blink_rate = (len(blinks) / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    return {
        "blink_count": len(blinks),
        "blink_rate_per_min": blink_rate,
        "avg_blink_duration_ms": np.mean(durations_ms),
        "max_blink_duration_ms": np.max(durations_ms),
    }


def is_blink_frame(ear_avg, threshold=0.2):
    """
    Check if a single frame represents a blink (eye closed).
    
    Args:
        ear_avg: Average EAR value for the frame
        threshold: EAR threshold
        
    Returns:
        Boolean indicating if this is a blink frame
    """
    return ear_avg < threshold
