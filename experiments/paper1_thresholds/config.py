# config.py

RUN_MODE = "benchmark"  # "webcam" or "benchmark"

VIDEO_CONFIG = {
    "source": 0,  # webcam index, video path, or RTSP URL
    "target_fps": 30,
    "frame_width": 416,
    "frame_height": 416,
    "use_frame_skip": False,
    "frame_skip_interval": 1
}

PREPROCESS_CONFIG = {
    "resize": False,
    "resize_width": 416,
    "resize_height": 416,

    "gaussian_blur": False,
    "gaussian_kernel": (3, 3),

    "bilateral_filter": False,
    "bilateral_d": 5,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,

    "normalize": False,
    "hist_eq": False
}

DETECTION_CONFIG = {
    "model_type": "yolov8_ultralytics",
    "weights_path": "models/yolov8s.pt",

    "confidence_threshold": 0.05,
    "target_classes": [
        "person"
    ]
}

BENCHMARK_CONFIG = {
    "input_type": "image_folder",   # "video" or "image_folder"
    "input_root": "datasets/MOT20/train/",
    "sequence_names": ["MOT20-02"], #,"MOT20-05"
    "output_folder": "benchmark/results/",
    "model_name": "yolov8m_0.05_GPU_t",
    "show_preview": True
}

TRACKER_CONFIG = {
    "track_activation_threshold": 0.1,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30
}