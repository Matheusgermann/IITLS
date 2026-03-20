# config.py

"""
Configurações principais da aplicação de contagem de tráfego em vídeo.

Lógica atual:
- Uma linha para o fluxo da esquerda
- Uma linha para o fluxo da direita
- Cada linha conta apenas objetos que cruzarem dentro da sua faixa/região válida
"""

# ============================================================
# CAMINHOS
# ============================================================
MODEL_PATH = "pesos/best.pt"
SOURCE_PATH = "videos_entrada/video_01_1.mp4"
OUTPUT_PATH = "videos_saida/video_01_1.mp4"

# ============================================================
# INFERÊNCIA
# ============================================================
CONFIDENCE_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.50
IMAGE_SIZE = 960
DEVICE = 0
TRACKER_CONFIG = "bytetrack.yaml"

# ============================================================
# CLASSES DE INTERESSE
# ============================================================
CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
}

# ============================================================
# LINHA DA ESQUERDA
# Conta apenas objetos cuja coordenada X do centro esteja dentro
# da faixa definida em LEFT_REGION_X_MIN e LEFT_REGION_X_MAX
# ============================================================

# Linha do fluxo da esquerda
LINE_LEFT_START = (150, 600)
LINE_LEFT_END = (642, 600)


LEFT_REGION_X_MIN = 0
LEFT_REGION_X_MAX = 760

# ============================================================
# LINHA DA DIREITA
# Conta apenas objetos cuja coordenada X do centro esteja dentro
# da faixa definida em RIGHT_REGION_X_MIN e RIGHT_REGION_X_MAX
# ============================================================

# Linha do fluxo da direita
LINE_RIGHT_START = (660, 550)
LINE_RIGHT_END = (1070, 550)

RIGHT_REGION_X_MIN = 760
RIGHT_REGION_X_MAX = 2000

# ============================================================
# FILTROS DE DETECÇÃO
# ============================================================
MIN_BOX_AREA = 600
MAX_BOX_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 4.50

# ============================================================
# SUPRESSÃO DE DUPLICIDADE NO MESMO FRAME
# ============================================================
DUPLICATE_IOU_THRESHOLD = 0.60
DUPLICATE_CENTER_DISTANCE = 40

# ============================================================
# SUPRESSÃO DE DUPLICIDADE DE EVENTO DE CONTAGEM
# Evita contar duas vezes objetos muito próximos, da mesma classe,
# cruzando a mesma linha em frames muito próximos.
# ============================================================
COUNT_EVENT_COOLDOWN_FRAMES = 15
COUNT_EVENT_COOLDOWN_DISTANCE = 90

# ============================================================
# VISUALIZAÇÃO
# True  -> mostra janela com preview
# False -> apenas processa e salva vídeo
# ============================================================
SHOW_WINDOW = True
WINDOW_NAME = "Traffic Counter"

# ============================================================
# DESENHO / OVERLAY
# ============================================================
LINE_LEFT_COLOR = (255, 255, 0)
LINE_RIGHT_COLOR = (0, 255, 255)

BOX_COLOR = (0, 200, 0)
CENTER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

LEFT_TEXT_COLOR = (255, 255, 0)
RIGHT_TEXT_COLOR = (0, 255, 255)

LINE_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
