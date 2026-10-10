# training/pseudo_label.py

"""
Pré-anota frames do domínio com um modelo professor (o best.pt atual ou um
modelo maior), gerando labels YOLO no esquema IITLS.

Anotar 2.000 frames do zero leva dias; revisar pré-anotações leva horas.
O script separa o que é confiável do que precisa de olho humano:

- caixas com conf >= --conf-keep viram label
- imagens com caixas entre --conf-review e --conf-keep entram em
  review.txt, ordenadas pela quantidade de casos duvidosos (revise essas
  primeiro no CVAT / Label Studio / Roboflow)

Exemplo:
    python training/pseudo_label.py --model pesos/best.pt \
        --images data/dominio/raw --out data/dominio/pseudo --imgsz 1280 --tta
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from classes import IITLS_NAMES, mapping_for_model_names

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="maior que o de treino: o professor pode ser lento, precisa ser preciso")
    parser.add_argument("--conf-keep", type=float, default=0.45)
    parser.add_argument("--conf-review", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--tta", action="store_true", help="test-time augmentation (flip + multiescala)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.model)
    mapping = mapping_for_model_names(model.names)
    print("Mapeamento professor -> IITLS:",
          {model.names[k]: IITLS_NAMES[v] for k, v in mapping.items()})

    images_out = args.out / "images"
    labels_out = args.out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    review: list[tuple[int, str]] = []

    for start in range(0, len(image_paths), args.batch):
        chunk = image_paths[start:start + args.batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=args.imgsz,
            conf=args.conf_review,
            iou=args.iou,
            augment=args.tta,
            device=args.device,
            verbose=False,
        )

        for image_path, result in zip(chunk, results):
            lines: list[str] = []
            doubtful = 0

            if result.boxes is not None:
                for cls, conf, xywhn in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.xywhn.tolist(),
                ):
                    target = mapping.get(int(cls))
                    if target is None:
                        continue
                    if conf >= args.conf_keep:
                        x, y, w, h = xywhn
                        lines.append(f"{target} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    else:
                        doubtful += 1

            shutil.copy2(image_path, images_out / image_path.name)
            (labels_out / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            if doubtful:
                review.append((doubtful, image_path.name))

        print(f"{min(start + args.batch, len(image_paths))}/{len(image_paths)}", end="\r")

    review.sort(reverse=True)
    (args.out / "review.txt").write_text(
        "".join(f"{name}\t{count}\n" for count, name in review), encoding="utf-8"
    )
    print(f"\n{len(image_paths)} imagens pré-anotadas em {args.out}")
    print(f"{len(review)} imagens com detecções duvidosas listadas em {args.out / 'review.txt'}")


if __name__ == "__main__":
    main()
