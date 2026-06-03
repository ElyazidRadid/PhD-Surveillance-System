import os
import json


def convert_mot_to_json(gt_path, output_path):
    frames = {}

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = int(parts[6])
            cls = int(parts[7])

            # Keep only valid person detections
            if conf != 1 or cls != 1:
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            obj = {
                "class_name": "person",
                "bbox": [x1, y1, x2, y2]
            }

            if frame_id not in frames:
                frames[frame_id] = []

            frames[frame_id].append(obj)

    # Convert to sorted list
    output = []
    for frame_id in sorted(frames.keys()):
        output.append({
            "frame_id": frame_id,
            "objects": frames[frame_id]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved annotation JSON to: {output_path}")


if __name__ == "__main__":
    gt_path = "C:/Users/msi/Documents/PhD Surveillance System/datasets/MOT20/train/MOT20-05/gt/gt.txt"
    output_path = "benchmark/annotations/MOT20-05.json"

    convert_mot_to_json(gt_path, output_path)