# training/train.py

"""
Treino do detector IITLS com presets pensados para o fluxo professor -> aluno.

Presets:
  teacher   modelo grande e preciso (roda na GPU de treino, não na Jetson).
            Usado para pré-anotar frames do domínio e como referência de acurácia.
  edge      modelo pequeno que vai para a Jetson (TensorRT FP16, 640 px).
  finetune  ajuste fino curto de um modelo já treinado sobre o mix com domínio.

Exemplos (Colab L4/A100):
    python training/train.py --preset teacher --data /content/iitls_bdd/data.yaml \
        --weights pesos/best.pt --name teacher_m960
    python training/train.py --preset edge --data data/mix/data.yaml --name edge_s640
    python training/train.py --preset finetune --data data/mix/data.yaml \
        --weights runs/edge_s640/weights/best.pt --name edge_s640_ft
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

COMMON = {
    "optimizer": "AdamW",
    "cos_lr": True,
    "patience": 20,
    "close_mosaic": 10,
    "save_period": 5,
    "plots": True,
    # Câmera fixa e elevada: leve rotação/perspectiva cobre variação de instalação
    "degrees": 3.0,
    "perspective": 0.0005,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    # Chuva, noite, contraluz
    "hsv_h": 0.015,
    "hsv_s": 0.6,
    "hsv_v": 0.5,
}

PRESETS = {
    "teacher": {
        "model": "yolo11l.pt",
        "imgsz": 960,
        "epochs": 60,
        "batch": 10,
        "lr0": 0.001,
        "mixup": 0.1,
        "copy_paste": 0.0,
    },
    "edge": {
        "model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 120,
        "batch": 32,
        "lr0": 0.001,
        "mixup": 0.05,
    },
    "finetune": {
        "model": None,
        "imgsz": 640,
        "epochs": 30,
        "batch": 32,
        "lr0": 0.0002,
        "warmup_epochs": 1.0,
        "mixup": 0.0,
        "close_mosaic": 5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=None,
                        help="checkpoint inicial (ex.: pesos/best.pt). Sem isso usa o preset")
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="runs/iitls")
    parser.add_argument("--hours", type=float, default=None,
                        help="limite de tempo (sobrepõe epochs): útil no Colab")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", nargs="*", default=[],
                        help="sobrescreve hiperparâmetros: --set imgsz=800 batch=8")
    args = parser.parse_args()

    if args.resume:
        if not args.weights:
            parser.error("--resume precisa de --weights apontando para last.pt")
        YOLO(args.weights).train(resume=True)
        return

    config = {**COMMON, **PRESETS[args.preset]}
    model_source = args.weights or config.pop("model")
    config.pop("model", None)

    if model_source is None:
        parser.error(f"o preset {args.preset} precisa de --weights")

    for item in args.set:
        key, value = item.split("=", 1)
        try:
            config[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            config[key] = {"true": True, "false": False}.get(value.lower(), value)

    if args.hours:
        config["time"] = args.hours

    print(f"Modelo inicial: {model_source}")
    print(f"Config: {config}")

    # Quando o checkpoint tem outro número de classes (ex.: best.pt com 10),
    # o Ultralytics reaproveita backbone/neck e reinicia só a cabeça de classes.
    model = YOLO(model_source)
    model.train(data=args.data, project=args.project, name=args.name, exist_ok=True, **config)

    metrics = model.val(data=args.data, imgsz=config["imgsz"], split="val", plots=False)
    print(f"\nmAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
