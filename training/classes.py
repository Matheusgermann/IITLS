# training/classes.py

"""
Esquema de classes do detector IITLS e mapeamentos a partir de datasets públicos.

Os IDs 0..6 são idênticos aos do best.pt atual (BDD100K), então o
CLASS_NAMES de config.py continua válido sem alteração. As classes que não
servem para contagem (traffic light, traffic sign, train) foram removidas:
elas consumiam capacidade do modelo e ~33% das anotações do BDD100K.
"""

IITLS_NAMES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
}

# BDD100K (formato YOLO do Kaggle a7madmostafa/bdd100k-yolo, nc=10)
# 0 person, 1 rider, 2 car, 3 bus, 4 truck, 5 bike, 6 motor,
# 7 traffic light, 8 traffic sign, 9 train
BDD100K_TO_IITLS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# COCO (80 classes, IDs do Ultralytics)
COCO_TO_IITLS = {0: 0, 1: 5, 2: 2, 3: 6, 5: 3, 7: 4}

# Pesos antigos (best.pt, 10 classes) usados como professor no pseudo-label
LEGACY_BDD_MODEL_TO_IITLS = BDD100K_TO_IITLS

SOURCE_MAPPINGS = {
    "bdd100k": BDD100K_TO_IITLS,
    "coco": COCO_TO_IITLS,
    "iitls": {class_id: class_id for class_id in IITLS_NAMES},
}


def mapping_for_model_names(model_names: dict[int, str]) -> dict[int, int]:
    """Mapeia as classes de um modelo qualquer para o esquema IITLS pelo nome."""
    aliases = {
        "person": 0,
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "van": 2,
        "bus": 3,
        "truck": 4,
        "bike": 5,
        "bicycle": 5,
        "motor": 6,
        "motorcycle": 6,
        "motorbike": 6,
    }
    mapping = {}
    for class_id, name in model_names.items():
        target = aliases.get(str(name).lower())
        if target is not None:
            mapping[int(class_id)] = target
    return mapping


def write_data_yaml(path, dataset_root, train, val, test=None) -> None:
    import yaml

    content = {
        "path": str(dataset_root),
        "train": train,
        "val": val,
    }
    if test:
        content["test"] = test
    content["nc"] = len(IITLS_NAMES)
    content["names"] = [IITLS_NAMES[i] for i in range(len(IITLS_NAMES))]

    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, sort_keys=False, allow_unicode=True)
