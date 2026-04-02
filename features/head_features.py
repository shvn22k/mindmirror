"""
Head position and movement feature extraction.

Uses nose tip and key facial landmarks to estimate head position and pose.
"""

import numpy as np
from .landmark_indices import NOSE, HEAD_POSE_POINTS, FACE_CENTER


def compute_head_position(landmarks):
    """
    Compute head position using nose tip.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Tuple of (x, y, z) normalized coordinates
    """
    nose_tip = landmarks[NOSE["tip"]]
    return nose_tip[0], nose_tip[1], nose_tip[2]


def compute_face_center(landmarks):
    """
    Compute face center position (midpoint between inner eye corners).
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Tuple of (x, y, z) coordinates
    """
    left_eye = landmarks[FACE_CENTER["left_eye_inner"]]
    right_eye = landmarks[FACE_CENTER["right_eye_inner"]]
    
    center = (left_eye + right_eye) / 2.0
    return center[0], center[1], center[2]


def compute_head_pose(landmarks):
    """
    Estimate head pose (pitch, yaw, roll) from landmarks.
    
    This is a simplified estimation using key landmark positions.
    For more accurate pose, use cv2.solvePnP with 3D model points.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Dict with pitch, yaw, roll estimates (in relative units, not degrees)
    """
    nose_tip = landmarks[HEAD_POSE_POINTS["nose_tip"]]
    chin = landmarks[HEAD_POSE_POINTS["chin"]]
    left_eye = landmarks[HEAD_POSE_POINTS["left_eye_corner"]]
    right_eye = landmarks[HEAD_POSE_POINTS["right_eye_corner"]]
    left_mouth = landmarks[HEAD_POSE_POINTS["left_mouth_corner"]]
    right_mouth = landmarks[HEAD_POSE_POINTS["right_mouth_corner"]]
    
    # Yaw: horizontal rotation (looking left/right)
    # Estimated by asymmetry between left and right eye distances to nose
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    yaw = nose_tip[0] - eye_center_x
    
    # Pitch: vertical rotation (looking up/down)
    # Estimated by vertical distance from nose to eye line vs nose to chin
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    nose_to_eye = eye_center_y - nose_tip[1]
    nose_to_chin = chin[1] - nose_tip[1]
    pitch = nose_to_eye / (nose_to_chin + 1e-6)  # Avoid division by zero
    
    # Roll: tilting head sideways
    # Estimated by angle between eyes
    eye_diff_y = right_eye[1] - left_eye[1]
    eye_diff_x = right_eye[0] - left_eye[0]
    roll = eye_diff_y / (eye_diff_x + 1e-6)
    
    return {
        "head_pitch": pitch,
        "head_yaw": yaw,
        "head_roll": roll,
    }


def compute_head_features(landmarks):
    """
    Compute all head-related features for a single frame.
    
    Args:
        landmarks: Array of shape (478, 3)
        
    Returns:
        Dict with head position and pose features
    """
    head_x, head_y, head_z = compute_head_position(landmarks)
    pose = compute_head_pose(landmarks)
    
    return {
        "head_x": head_x,
        "head_y": head_y,
        "head_z": head_z,
        "head_pitch": pose["head_pitch"],
        "head_yaw": pose["head_yaw"],
        "head_roll": pose["head_roll"],
    }


def compute_head_movement(positions):
    """
    Compute head movement statistics from a sequence of positions.
    
    Args:
        positions: Array of shape (n_frames, 3) with x, y, z positions
        
    Returns:
        Dict with movement statistics
    """
    if len(positions) < 2:
        return {
            "head_movement_total": 0.0,
            "head_movement_variance": 0.0,
            "head_movement_range_x": 0.0,
            "head_movement_range_y": 0.0,
        }
    
    # Frame-to-frame displacement
    displacements = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(displacements[:, :2] ** 2, axis=1))  # x, y only
    
    # Statistics
    total_movement = np.sum(distances)
    variance = np.var(positions[:, :2])
    range_x = np.max(positions[:, 0]) - np.min(positions[:, 0])
    range_y = np.max(positions[:, 1]) - np.min(positions[:, 1])
    
    return {
        "head_movement_total": total_movement,
        "head_movement_variance": variance,
        "head_movement_range_x": range_x,
        "head_movement_range_y": range_y,
    }
