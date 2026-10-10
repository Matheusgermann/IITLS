# training/evaluate.py

"""
Compara pesos lado a lado no MESMO conjunto de validação de domínio.

Saída: tabela markdown por classe (AP50 e AP50-95), mAP geral, recall,
latência e GFLOPs. É assim que se decide se um peso novo substitui o best.pt.

Funciona com o best.pt antigo (10 classes BDD100K): as previsões ficam
restritas às 7 classes IITLS, cujos IDs coincidem.

Exemplo:
    python training/evaluate.py --data data/mix/data.yaml --imgsz 640 \
        --weights pesos/best.pt runs/iitls/edge_s640/weights/best.pt \
        --report runs/iitls/comparacao.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from classes import IITLS_NAMES


def evaluate(weights: str, data: str, imgsz: int, device: str | None, batch: int) -> dict:
    model = YOLO(weights)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = get_flops(model.model, imgsz)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        classes=list(IITLS_NAMES),
        plots=False,
        verbose=False,
    )

    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        name = IITLS_NAMES.get(int(class_id))
        if name is None:
            continue
        per_class[name] = (float(metrics.box.ap50[index]), float(metrics.box.ap[index]))

    return {
        "weights": weights,
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_ms": float(metrics.speed["inference"]),
        "params_m": params_m,
        "gflops": gflops,
        "per_class": per_class,
    }


def to_markdown(rows: list[dict], imgsz: int) -> str:
    lines = [
        f"## Comparação de pesos (imgsz={imgsz})",
        "",
        "| pesos | mAP50 | mAP50-95 | P | R | ms/img | params (M) | GFLOPs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {Path(row['weights']).parent.parent.name or row['weights']}/{Path(row['weights']).name} "
            f"| {row['map50']:.3f} | {row['map']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['inference_ms']:.1f} | {row['params_m']:.1f} | {row['gflops']:.1f} |"
        )

    lines += ["", "### AP50 / AP50-95 por classe", ""]
    header = "| classe | " + " | ".join(Path(r["weights"]).parent.parent.name or r["weights"] for r in rows) + " |"
    lines += [header, "|---|" + "---|" * len(rows)]
    for name in IITLS_NAMES.values():
        cells = []
        for row in rows:
            ap50, ap = row["per_class"].get(name, (float("nan"), float("nan")))
            cells.append(f"{ap50:.3f} / {ap:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = [evaluate(w, args.data, args.imgsz, args.device, args.batch) for w in args.weights]
    report = to_markdown(rows, args.imgsz)
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
