import cv2


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
    }


def read_video_frames(video_path, sample_every=3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % sample_every == 0:
            yield frame_idx, frame

        frame_idx += 1

    cap.release()


def read_all_sampled_frames(video_path, sample_every=3):
    frames = []
    for _, frame in read_video_frames(video_path, sample_every):
        frames.append(frame)
    return frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m cognitive_load.video_utils <video_path>")
        print("\nTesting with first video from dataset...")

        from cognitive_load.dataset import AVCaffeDataset

        dataset = AVCaffeDataset(
            video_base="data/processed/avcaffe/videos",
            label_path="data/processed/avcaffe/labels/mental_demand.txt",
        )
        video_path, _ = dataset[0]
    else:
        video_path = sys.argv[1]

    print(f"Video: {video_path}")

    info = get_video_info(video_path)
    if info:
        print(f"  FPS: {info['fps']}")
        print(f"  Total frames: {info['frame_count']}")
        print(f"  Duration: {info['duration_seconds']:.2f}s")
        print(f"  Resolution: {info['width']}x{info['height']}")

        sampled_count = len(list(read_video_frames(video_path, sample_every=3)))
        print(f"  Sampled frames (every 3rd): {sampled_count}")
    else:
        print("  Could not open video")
