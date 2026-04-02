import os

from cognitive_load.load_labels import load_labels


def get_video_label(video_path, label_map):
    """
    Extract label for a video based on its filename.

    Video naming: aiim001_task_3_clip_012.avi
    Label key:    aiim001_task_3
    """
    filename = os.path.basename(video_path)

    parts = filename.split("_")

    participant = parts[0]
    task = parts[1] + "_" + parts[2]

    key = f"{participant}_{task}"

    return label_map.get(key, None)


def parse_video_info(video_path):
    filename = os.path.basename(video_path)
    name = filename.replace(".avi", "")
    parts = name.split("_")

    return {
        "participant": parts[0],
        "task": f"{parts[1]}_{parts[2]}",
        "clip_num": int(parts[4]),
    }


class AVCaffeDataset:
    """Dataset loader for AVCAffe cognitive load dataset."""

    def __init__(self, video_base, label_path):
        self.video_base = video_base
        self.label_path = label_path
        self.labels = load_labels(label_path)
        self._video_list = None

    def _build_video_list(self):
        if self._video_list is not None:
            return

        self._video_list = []

        for participant in sorted(os.listdir(self.video_base)):
            participant_path = os.path.join(self.video_base, participant)
            if not os.path.isdir(participant_path):
                continue

            for task in sorted(os.listdir(participant_path)):
                task_path = os.path.join(participant_path, task)
                if not os.path.isdir(task_path):
                    continue

                for video_file in sorted(os.listdir(task_path)):
                    if video_file.endswith(".avi"):
                        self._video_list.append(os.path.join(task_path, video_file))

    def __len__(self):
        self._build_video_list()
        return len(self._video_list)

    def __iter__(self):
        self._build_video_list()

        for video_path in self._video_list:
            label = get_video_label(video_path, self.labels)
            if label is not None:
                yield video_path, label

    def __getitem__(self, idx):
        self._build_video_list()
        video_path = self._video_list[idx]
        label = get_video_label(video_path, self.labels)
        return video_path, label

    def get_stats(self):
        self._build_video_list()

        matched = sum(1 for _, label in self if label is not None)

        return {
            "total_videos": len(self._video_list),
            "matched": matched,
            "total_labels": len(self.labels),
        }


if __name__ == "__main__":
    dataset = AVCaffeDataset(
        video_base="data/processed/avcaffe/videos",
        label_path="data/processed/avcaffe/labels/mental_demand.txt",
    )

    print(f"Dataset size: {len(dataset)} videos")
    print(f"Stats: {dataset.get_stats()}")
    print("\nFirst 5 samples:")
    for i, (video, label) in enumerate(dataset):
        print(f"  {os.path.basename(video)} -> {label}")
        if i >= 4:
            break
