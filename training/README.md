# Treino do detector IITLS

Pipeline para melhorar o `pesos/best.pt` e gerar o modelo que roda na Jetson.
Tudo aqui é independente do `app.py`: os IDs de classe 0–6 continuam os mesmos,
então um peso novo entra no lugar do antigo sem mudar `config.py`.

## Diagnóstico do best.pt atual

Treinado no Colab (L4) a partir do `yolo11m.pt`, 20 épocas, 960 px, BDD100K
(70k imagens de treino, 10 classes). Validação no BDD100K val (10k imagens):

| classe | P | R | AP50 | AP50-95 |
|---|---|---|---|---|
| todas | 0.761 | 0.546 | 0.606 | 0.355 |
| person | 0.794 | 0.618 | 0.720 | 0.385 |
| rider | 0.699 | 0.484 | 0.537 | 0.288 |
| car | 0.833 | 0.764 | 0.841 | 0.534 |
| bus | 0.711 | 0.607 | 0.690 | 0.540 |
| truck | 0.690 | 0.641 | 0.702 | 0.521 |
| bike | 0.647 | 0.504 | 0.558 | 0.293 |
| motor | 0.738 | 0.458 | 0.530 | 0.273 |

O que limita a acurácia, em ordem de impacto:

1. **Domínio errado.** O BDD100K é câmera de painel (dashcam), na altura do
   carro. As câmeras do IITLS são fixas, elevadas e olham o cruzamento de cima.
   Ângulo, escala e oclusão são outros. Nenhum número de época no BDD100K fecha
   essa diferença, e o mAP acima não diz nada sobre como o modelo vai na rua real.
2. **Recall baixo onde mais importa.** Contagem perde veículo quando o recall cai.
   `motor` (0.46) e `bike` (0.50) são as piores classes, e moto é uma fração
   grande do trânsito brasileiro.
3. **Curva já estabilizou.** O mAP50 subiu só de 0.602 para 0.606 nas últimas 8
   épocas. Treinar mais no mesmo dado rende pouco.
4. **Classes inúteis para contagem.** `traffic light`, `traffic sign` e `train`
   somam ~33% das caixas do BDD100K e disputam capacidade com o que interessa.
5. **Modelo pesado demais para a borda.** YOLO11m a 960 px custa ~150 GFLOPs por
   frame. Na Jetson Nano isso dá na casa de 1–2 FPS (estimativa; medir com
   `export_edge.py --benchmark`).

## Estratégia

```
BDD100K (7 classes) ──► professor YOLO11l/m 960 ──► pré-anota frames das câmeras IITLS
                                                         │ revisão humana (CVAT/Label Studio)
                                                         ▼
                        mix: BDD100K + domínio x8 ──► aluno YOLO11s/n 640 ──► TensorRT FP16 na Jetson
                                                         │
                                       validação SÓ em vídeos de domínio que não entraram no treino
```

O ganho real vem do passo do meio: frames das próprias câmeras, revisados.
Meta inicial: 1.500–3.000 frames cobrindo dia, noite, chuva, contraluz e
horário de pico, de pelo menos 3 pontos de câmera diferentes.

## Passo a passo (Colab ou máquina com GPU)

```bash
pip install -r training/requirements.txt
cd training

# 1. BDD100K -> esquema IITLS (remove luz/placa/trem, corrige caixas inválidas)
python remap_dataset.py --src /content/bdd100k_yolo_kaggle --dst /content/iitls_bdd \
    --source bdd100k --splits train val test

# 2. Professor (opcional se o best.pt atual servir de professor)
python train.py --preset teacher --data /content/iitls_bdd/data.yaml \
    --weights ../pesos/best.pt --name teacher_m960 --hours 10

# 3. Frames das câmeras reais (e/ou do AI City Track 1 que já está no Drive)
python extract_frames.py --videos /content/videos_iitls --out /content/dominio/raw \
    --every-seconds 1 --max-per-video 600

# 4. Pré-anotação com o professor + lista do que revisar primeiro
python pseudo_label.py --model runs/iitls/teacher_m960/weights/best.pt \
    --images /content/dominio/raw --out /content/dominio/pseudo --imgsz 1280 --tta
#    -> importar /content/dominio/pseudo no CVAT/Label Studio, revisar
#       (comece pelos arquivos do topo de review.txt), exportar em YOLO para
#       /content/dominio/revisado

# 5. Mix final e treino do modelo de borda
python build_mix.py --out /content/mix --generic /content/iitls_bdd/train \
    --domain /content/dominio/revisado --domain-repeat 8 --val-fraction 0.2
python train.py --preset edge --data /content/mix/data.yaml --name edge_s640

# 6. Comparar com o peso atual no MESMO conjunto de domínio
python evaluate.py --data /content/mix/data.yaml --imgsz 640 \
    --weights ../pesos/best.pt runs/iitls/edge_s640/weights/best.pt \
    --report runs/iitls/comparacao.md

# 7. Exportar
python export_edge.py onnx --weights runs/iitls/edge_s640/weights/best.pt --imgsz 640
#    na Jetson:
python export_edge.py engine --weights best.pt --imgsz 640 --half --benchmark video.mp4
```

Quando o peso novo ganhar no passo 6, copie para `pesos/best.pt` e ajuste
`IMAGE_SIZE = 640` em `config.py`.

## Escolha do modelo de borda

| modelo | GFLOPs (640) | uso |
|---|---|---|
| YOLO11n | ~6.5 | Jetson Nano (4 GB) com folga para tracking |
| YOLO11s | ~21.5 | Jetson Orin Nano: melhor recall em moto/bike |
| YOLO11m | ~68 | só como professor |

Comece pelo `s`; se não bater o FPS alvo na placa, troque para `n`
(`--weights yolo11n.pt`).

**Jetson Nano vs. Orin Nano.** A Jetson Nano original (Maxwell) está presa no
JetPack 4.6 (Python 3.6, TensorRT 8.2), não acelera INT8 e já saiu de linha. A
Orin Nano roda JetPack 6, aceita INT8 e entrega uma ordem de grandeza a mais de
desempenho. Para um produto novo, a Orin Nano é a escolha segura; o pipeline
acima serve para as duas (`--opset 13` mantém o ONNX compatível com a Nano).

## Outras alavancas (ainda não implementadas)

- **Inferir só na região de contagem.** Recortar o ROI das linhas antes do
  modelo aumenta a resolução efetiva dos veículos e reduz custo.
- **Avaliar contagem, não só mAP.** O AI City 2021 Track 1 (no Drive) tem
  contagem real por movimento; dá para medir erro de contagem do `app.py`.
- **Licença.** BDD100K e AI City têm termos próprios de uso; confirme uso
  comercial antes de vender um modelo treinado neles. Frames próprios
  rotulados eliminam essa dependência.
