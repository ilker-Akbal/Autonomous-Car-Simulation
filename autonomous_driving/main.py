from pathlib import Path

from src.inference.video_processor import run_video


def main():
    video_path = Path(
        r"two-cars-go-straight-through-a-red-traffic-light-driving-dash-cam-uk-dash-cam-ca.mp4"
    )

    print("=" * 60)
    print("ADAS PIPELINE BAŞLIYOR")
    print(f"Video: {video_path}")
    print("=" * 60)

    run_video(str(video_path))

    print("=" * 60)
    print("ADAS PIPELINE TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()