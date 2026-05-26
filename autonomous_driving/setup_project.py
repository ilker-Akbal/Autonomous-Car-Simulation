from pathlib import Path

PROJECT_ROOT = Path(r"autonomous_driving_project")

folders = [
    "configs",
    "scripts",
    "src",
    "src/data",
    "src/models",
    "src/inference",
    "src/utils",
    "outputs",
    "outputs/prepared_data",
    "outputs/predictions",
    "outputs/logs",
    "outputs/models",
    "notebooks",
]

files = {
    "README.md": "# Autonomous Driving Project\n",
    "requirements.txt": "",
    "main.py": "",
    "configs/data_bdd.yaml": "",
    "configs/train.yaml": "",
    "configs/infer.yaml": "",
    "scripts/prepare_bdd.py": "",
    "scripts/prepare_kitti.py": "",
    "scripts/train_yolo.py": "",
    "scripts/infer_image.py": "",
    "scripts/infer_video.py": "",
    "scripts/evaluate.py": "",
    "src/__init__.py": "",
    "src/data/__init__.py": "",
    "src/data/bdd_loader.py": "",
    "src/data/kitti_loader.py": "",
    "src/data/converter.py": "",
    "src/data/class_map.py": "",
    "src/models/__init__.py": "",
    "src/models/detector.py": "",
    "src/inference/__init__.py": "",
    "src/inference/predictor.py": "",
    "src/inference/video_processor.py": "",
    "src/inference/decision_engine.py": "",
    "src/utils/__init__.py": "",
    "src/utils/paths.py": "",
    "src/utils/io.py": "",
    "src/utils/visualizer.py": "",
    "src/utils/logger.py": "",
}

def main():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

    for relative_path, content in files.items():
        file_path = PROJECT_ROOT / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")

    print(f"Proje yapısı oluşturuldu: {PROJECT_ROOT}")

if __name__ == "__main__":
    main()