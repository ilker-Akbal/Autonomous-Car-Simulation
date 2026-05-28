from pathlib import Path
import cv2


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return path


def open_video(video_path):
    video_path = check_file_exists(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    return cap


def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def create_video_writer(output_path, fps, width, height):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Video writer açılamadı: {output_path}")

    return writer


def save_frame(frame, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise RuntimeError(f"Görsel kaydedilemedi: {output_path}")