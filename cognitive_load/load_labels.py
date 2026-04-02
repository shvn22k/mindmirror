def load_labels(label_file_path):
    """
    Load labels from a file with format: key, value
    Example: aiim001_task_1, 1.0

    Returns: dict mapping key -> float value
    """
    label_map = {}

    with open(label_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," not in line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            value = parts[1].strip()

            try:
                label_map[key] = float(value)
            except ValueError:
                print(f"Warning: Could not parse value '{value}' for key '{key}'")
                continue

    return label_map


if __name__ == "__main__":
    path = "data/processed/avcaffe/labels/mental_demand.txt"
    labels = load_labels(path)

    print("Total labels:", len(labels))
    print("Sample:", list(labels.items())[:5])
