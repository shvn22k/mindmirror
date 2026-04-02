"""
Gaze direction and fixation feature extraction.

Uses iris landmarks to estimate gaze direction relative to eye center.
"""

import numpy as np
from .landmark_indices import LEFT_IRIS, RIGHT_IRIS, LEFT_EYE, RIGHT_EYE


def compute_eye_center(landmarks, eye_indices):
    """
    Compute the center of an eye from its corner landmarks.
    
    Args:
        landmarks: Array of shape (478, 3)
        eye_indices: Dict with p1 (left corner) and p4 (right corner)
        
    Returns:
        Array of (x, y) center coordinates
    """
    p1 = landmarks[eye_indices["p1"]][:2]
    p4 = landmarks[eye_indices["p4"]][:2]
    return (p1 + p4) / 2.0


def compute_gaze_direction_single_eye(landmarks, iris_indices, eye_indices):
    """
    Compute gaze direction for a single eye.
    
    Gaze is estimated as the offset of iris center from eye center,
    normalized by eye width.
    
    Args:
        landmarks: Array of shape (478, 3)
        iris_indices: Dict with 'center' key
        eye_indices: Dict with p1, p4 keys (eye corners)
        
    Returns:
        Tuple of (gaze_x, gaze_y) in normalized units
        - Positive x = looking right
        - Positive y = looking down
    """
    iris_center = landmarks[iris_indices["center"]][:2]
    eye_center = compute_eye_center(landmarks, eye_indices)
    
    # Eye width for normalization
    p1 = landmarks[eye_indices["p1"]][:2]
    p4 = landmarks[eye_indices["p4"]][:2]
    eye_width = np.sqrt(np.sum((p4 - p1) ** 2))
    
    if eye_width == 0:
        return 0.0, 0.0
    
    # Gaze offset normalized by eye width
    gaze_x = (iris_center[0] - eye_center[0]) / eye_width
    gaze_y = (iris_center[1] - eye_center[1]) / eye_width
    
    return gaze_x, gaze_y


def compute_gaze_features(landmarks):
    """
    Compute gaze-related features for a single frame.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Dict with gaze features
    """
    left_gaze_x, left_gaze_y = compute_gaze_direction_single_eye(
        landmarks, LEFT_IRIS, LEFT_EYE
    )
    right_gaze_x, right_gaze_y = compute_gaze_direction_single_eye(
        landmarks, RIGHT_IRIS, RIGHT_EYE
    )
    
    # Average gaze direction
    gaze_x = (left_gaze_x + right_gaze_x) / 2.0
    gaze_y = (left_gaze_y + right_gaze_y) / 2.0
    
    return {
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
        "gaze_left_x": left_gaze_x,
        "gaze_left_y": left_gaze_y,
        "gaze_right_x": right_gaze_x,
        "gaze_right_y": right_gaze_y,
    }


def compute_gaze_dispersion(gaze_sequence):
    """
    Compute gaze dispersion (spread) over a sequence of frames.
    
    Higher dispersion = more scattered gaze (exploring)
    Lower dispersion = focused gaze (concentrating)
    
    Args:
        gaze_sequence: Array of shape (n_frames, 2) with gaze_x, gaze_y
        
    Returns:
        Dict with dispersion metrics
    """
    if len(gaze_sequence) < 2:
        return {
            "gaze_dispersion": 0.0,
            "gaze_std_x": 0.0,
            "gaze_std_y": 0.0,
        }
    
    std_x = np.std(gaze_sequence[:, 0])
    std_y = np.std(gaze_sequence[:, 1])
    
    # Overall dispersion (RMS of standard deviations)
    dispersion = np.sqrt(std_x**2 + std_y**2)
    
    return {
        "gaze_dispersion": dispersion,
        "gaze_std_x": std_x,
        "gaze_std_y": std_y,
    }


def detect_fixations(gaze_sequence, threshold=0.05, min_frames=3):
    """
    Detect fixation periods (stable gaze).
    
    A fixation is detected when gaze stays within threshold for min_frames.
    
    Args:
        gaze_sequence: Array of shape (n_frames, 2) with gaze_x, gaze_y
        threshold: Maximum gaze movement to count as fixation
        min_frames: Minimum frames for a fixation
        
    Returns:
        List of fixation events: [(start_frame, end_frame, duration), ...]
    """
    if len(gaze_sequence) < min_frames:
        return []
    
    fixations = []
    i = 0
    
    while i < len(gaze_sequence) - min_frames + 1:
        window = gaze_sequence[i:i+min_frames]
        dispersion = np.std(window, axis=0).max()
        
        if dispersion < threshold:
            # Start of fixation found, extend it
            start = i
            j = i + min_frames
            
            while j < len(gaze_sequence):
                extended = gaze_sequence[start:j+1]
                if np.std(extended, axis=0).max() < threshold:
                    j += 1
                else:
                    break
            
            end = j - 1
            duration = end - start + 1
            fixations.append((start, end, duration))
            i = j
        else:
            i += 1
    
    return fixations


def compute_fixation_features(gaze_sequence, fps=25):
    """
    Compute fixation statistics for a gaze sequence.
    
    Args:
        gaze_sequence: Array of shape (n_frames, 2)
        fps: Frames per second
        
    Returns:
        Dict with fixation metrics
    """
    fixations = detect_fixations(gaze_sequence)
    
    if len(fixations) == 0:
        return {
            "fixation_count": 0,
            "avg_fixation_duration_ms": 0.0,
            "fixation_ratio": 0.0,
        }
    
    durations = [f[2] for f in fixations]
    total_fixation_frames = sum(durations)
    
    return {
        "fixation_count": len(fixations),
        "avg_fixation_duration_ms": np.mean(durations) * (1000 / fps),
        "fixation_ratio": total_fixation_frames / len(gaze_sequence),
    }
