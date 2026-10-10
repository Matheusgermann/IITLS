# training/extract_frames.py

"""
Extrai frames das câmeras reais (ou de datasets de câmera fixa, como o AI City
Track 1) para montar o conjunto de domínio: a mesma altura, ângulo e tipo de
via que a Jetson vai ver em produção.

Pula frames quase idênticos ao último salvo (câmera fixa + trânsito parado gera
muita redundância) para não inflar o dataset com imagens repetidas.

Exemplo:
    python training/extract_frames.py --videos videos_entrada --out data/dominio/raw \
        --every-seconds 1.0 --max-per-video 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def frame_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    every_seconds: float,
    max_per_video: int,
    min_difference: float,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"  ! não abriu: {video_path}")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * every_seconds)))
    last_signature = None
    frame_index = 0
    saved = 0

    while saved < max_per_video:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % stride == 0:
            signature = frame_signature(frame)
            is_new_scene = (
                last_signature is None
                or float(np.mean(np.abs(signature - last_signature))) >= min_difference
            )
            if is_new_scene:
                name = f"{video_path.stem}_{frame_index:07d}.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_signature = signature
                saved += 1

        frame_index += 1

    capture.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, type=Path, help="arquivo de vídeo ou pasta com vídeos")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--every-seconds", type=float, default=1.0)
    parser.add_argument("--max-per-video", type=int, default=600)
    parser.add_argument("--min-difference", type=float, default=2.0,
                        help="diferença média de pixel (0-255) mínima para salvar outro frame")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)

    total = 0
    for video_path in videos:
        saved = extract_from_video(
            video_path, args.out, args.every_seconds, args.max_per_video, args.min_difference
        )
        print(f"{video_path.name}: {saved} frames")
        total += saved

    print(f"\nTotal: {total} frames em {args.out}")


if __name__ == "__main__":
    main()
