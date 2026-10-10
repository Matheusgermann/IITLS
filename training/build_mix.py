# training/build_mix.py

"""
Monta o dataset final de treino misturando:

- dados genéricos (BDD100K convertido, COCO, UA-DETRAC...) -> variedade
- dados de domínio (frames das câmeras IITLS revisados)     -> o que importa

O domínio é pequeno perto dos 70k do BDD100K, então ele é repetido
(--domain-repeat) na lista de treino para pesar de verdade no gradiente.

A validação usa SÓ dados de domínio, separados por vídeo/câmera: frames do
mesmo vídeo no treino e na validação inflam o mAP e escondem overfitting.

Exemplo:
    python training/build_mix.py --out data/mix \
        --generic /content/iitls_bdd/train \
        --domain data/dominio/revisado --domain-repeat 8 --val-fraction 0.2
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from classes import write_data_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    images_dir = root / "images" if (root / "images").is_dir() else root
    return sorted(
        p.resolve() for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def group_key(image_path: Path) -> str:
    """Frames gerados por extract_frames.py se chamam <video>_<frame>.jpg."""
    stem = image_path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def split_by_group(
    images: list[Path],
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[group_key(image)].append(image)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    if len(keys) < 2:
        print("! só existe 1 vídeo/câmera no domínio: a validação vai compartilhar cena com o treino")
        shuffled = images[:]
        random.Random(seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * val_fraction))
        return shuffled[cut:], shuffled[:cut]

    target_val = len(images) * val_fraction
    val: list[Path] = []
    train: list[Path] = []
    for key in keys:
        if len(val) < target_val:
            val.extend(groups[key])
        else:
            train.extend(groups[key])

    if not train:
        train, val = val[:-len(groups[keys[-1]])], groups[keys[-1]]

    return train, val


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{p}\n" for p in images), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generic", nargs="*", type=Path, default=[],
                        help="pastas de treino genéricas (com images/ e labels/ lado a lado)")
    parser.add_argument("--generic-fraction", type=float, default=1.0,
                        help="usa só uma fração do genérico (ex.: 0.3 para treinos rápidos)")
    parser.add_argument("--domain", nargs="+", type=Path, required=True,
                        help="pastas de domínio revisadas (images/ + labels/)")
    parser.add_argument("--domain-repeat", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    generic: list[Path] = []
    for root in args.generic:
        images = list_images(root)
        if args.generic_fraction < 1.0:
            images = rng.sample(images, int(len(images) * args.generic_fraction))
        generic.extend(images)

    domain: list[Path] = []
    for root in args.domain:
        domain.extend(list_images(root))

    domain_train, domain_val = split_by_group(domain, args.val_fraction, args.seed)

    train = generic + domain_train * args.domain_repeat
    rng.shuffle(train)

    write_list(args.out / "train.txt", train)
    write_list(args.out / "val.txt", domain_val)
    write_data_yaml(
        args.out / "data.yaml",
        dataset_root=args.out.resolve(),
        train="train.txt",
        val="val.txt",
    )

    print(f"genérico: {len(generic)} imagens")
    print(f"domínio treino: {len(domain_train)} imagens x{args.domain_repeat}")
    print(f"domínio val: {len(domain_val)} imagens ({len({group_key(p) for p in domain_val})} vídeos)")
    print(f"data.yaml: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
