# training/remap_dataset.py

"""
Converte um dataset YOLO (BDD100K, COCO ou outro) para o esquema de classes IITLS.

- Remove classes que não interessam à contagem
- Corrige/descarta caixas inválidas (fora de [0,1], largura/altura <= 0)
- Descarta caixas minúsculas que só viram ruído no treino
- Imagens são ligadas por symlink (não duplica 5 GB de disco)

Exemplo (Colab, depois de baixar o BDD100K do Kaggle):
    python training/remap_dataset.py \
        --src /content/bdd100k_yolo_kaggle --dst /content/iitls_bdd \
        --source bdd100k --splits train val test
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from classes import IITLS_NAMES, SOURCE_MAPPINGS, write_data_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, x - w / 2)
    y1 = max(0.0, y - h / 2)
    x2 = min(1.0, x + w / 2)
    y2 = min(1.0, y + h / 2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def remap_label_file(
    src_label: Path,
    mapping: dict[int, int],
    min_box_size: float,
    stats: Counter,
) -> list[str]:
    lines: list[str] = []

    if not src_label.exists():
        return lines

    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            stats["linhas_malformadas"] += 1
            continue

        class_id = int(float(parts[0]))
        if class_id not in mapping:
            stats["caixas_classe_removida"] += 1
            continue

        box = clamp_box(*map(float, parts[1:]))
        if box is None:
            stats["caixas_invalidas"] += 1
            continue

        x, y, w, h = box
        if w < min_box_size or h < min_box_size:
            stats["caixas_minusculas"] += 1
            continue

        target = mapping[class_id]
        stats[f"classe_{IITLS_NAMES[target]}"] += 1
        lines.append(f"{target} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return lines


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def remap_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    mapping: dict[int, int],
    min_box_size: float,
    keep_empty: bool,
    copy: bool,
) -> Counter:
    stats: Counter = Counter()
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(src_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stats["imagens_lidas"] += 1
        label_path = src_labels / f"{image_path.stem}.txt"
        lines = remap_label_file(label_path, mapping, min_box_size, stats)

        if not lines and not keep_empty:
            stats["imagens_sem_objeto_descartadas"] += 1
            continue

        link_or_copy(image_path, dst_images / image_path.name, copy)
        (dst_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["imagens_escritas"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="raiz do dataset YOLO de origem")
    parser.add_argument("--dst", required=True, type=Path, help="raiz do dataset convertido")
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_MAPPINGS))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--min-box-size", type=float, default=0.004,
                        help="lado mínimo normalizado (0.004 ~ 4 px numa imagem de 1280)")
    parser.add_argument("--keep-empty", action="store_true",
                        help="mantém imagens sem objetos (negativos). Por padrão mantém só no val/test")
    parser.add_argument("--copy", action="store_true", help="copia imagens em vez de symlink")
    args = parser.parse_args()

    mapping = SOURCE_MAPPINGS[args.source]

    for split in args.splits:
        keep_empty = args.keep_empty or split != "train"
        stats = remap_split(args.src, args.dst, split, mapping, args.min_box_size, keep_empty, args.copy)
        print(f"\n[{split}]")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")

    write_data_yaml(
        args.dst / "data.yaml",
        dataset_root=args.dst.resolve(),
        train="train/images",
        val="val/images",
        test="test/images" if "test" in args.splits else None,
    )
    print(f"\ndata.yaml escrito em {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
