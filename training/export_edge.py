# training/export_edge.py

"""
Exporta o peso para rodar na Jetson.

O engine TensorRT (.engine) é específico da GPU e da versão do TensorRT,
então ele tem que ser gerado NA PRÓPRIA Jetson. Aqui há dois modos:

  onnx    (na máquina de treino) gera um .onnx estático e simplificado,
          pronto para levar à Jetson.
  engine  (na Jetson) gera o .engine FP16 (ou INT8 em Jetson Orin) e
          mede FPS real com --benchmark.

Exemplos:
    # máquina de treino
    python training/export_edge.py onnx --weights runs/iitls/edge_s640/weights/best.pt --imgsz 640

    # Jetson (dentro do container ultralytics para JetPack)
    python training/export_edge.py engine --weights best.pt --imgsz 640 --half --benchmark video.mp4
    python training/export_edge.py engine --weights best.pt --imgsz 640 --int8 \
        --data data/mix/data.yaml --benchmark video.mp4
"""

from __future__ import annotations

import argparse
import time

from ultralytics import YOLO


def benchmark(model_path: str, source: str, imgsz: int, frames: int) -> None:
    import cv2

    model = YOLO(model_path, task="detect")
    capture = cv2.VideoCapture(source)
    images = []
    while len(images) < frames:
        success, frame = capture.read()
        if not success:
            break
        images.append(frame)
    capture.release()

    if not images:
        raise RuntimeError(f"não foi possível ler frames de {source}")

    for frame in images[:10]:
        model.predict(frame, imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    for frame in images:
        model.predict(frame, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    fps = len(images) / elapsed
    print(f"{model_path}: {fps:.1f} FPS ({1000 / fps:.1f} ms/frame, inclui pré e pós-processamento)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["onnx", "engine"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 (padrão recomendado na Jetson)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8: só compensa em Jetson Orin; Jetson Nano (Maxwell) não acelera INT8")
    parser.add_argument("--data", default=None, help="data.yaml para calibração INT8")
    parser.add_argument("--workspace", type=float, default=2.0, help="GB para o builder do TensorRT")
    parser.add_argument("--benchmark", default=None, help="vídeo para medir FPS depois de exportar")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--opset", type=int, default=13,
                        help="13 é aceito pelo TensorRT 8.2 do JetPack 4.6 (Jetson Nano)")
    args = parser.parse_args()

    model = YOLO(args.weights)

    if args.mode == "onnx":
        path = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, opset=args.opset)
    else:
        if args.int8 and not args.data:
            parser.error("--int8 precisa de --data para calibrar")
        path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=args.half and not args.int8,
            int8=args.int8,
            data=args.data,
            workspace=args.workspace,
            dynamic=False,
            batch=1,
        )

    print(f"Exportado: {path}")

    if args.benchmark:
        benchmark(str(path), args.benchmark, args.imgsz, args.frames)


if __name__ == "__main__":
    main()
