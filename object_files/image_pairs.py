import sys
import pickle
import torch
import pandas as pd
from transformers import OneFormerForUniversalSegmentation

with open("object_files/painting_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
id2label = model.config.id2label

# based on COCO-Stuff labels (from https://github.com/nightrome/cocostuff/blob/master/labels.md)
COCO_STUFF_LABELS = {
    0: "background",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "refrigerator",
    18: "shower curtain",
    19: "toilet",
    20: "sink",
    21: "bathtub",
    22: "other furniture",
    23: "other structure",
    24: "ceiling",
    25: "flooring",
    26: "road",
    27: "sidewalk",
    28: "parking lot",
    29: "field",
    30: "grass",
    31: "water",
    32: "river",
    33: "lake",
    34: "mountain",
    35: "sky",
    36: "cloud",
    37: "snow",
    38: "mountain snow",
    39: "hill",
    40: "sand",
    41: "dirt",
    42: "road marking",
    43: "pavement",
    44: "bridge",
    45: "rail track",
    46: "fence",
    47: "guard rail",
    48: "billboard",
    49: "pole",
    50: "traffic light",
    51: "fire hydrant",
    52: "stop sign",
    53: "parking meter",
    54: "bench",
    55: "banner",
    56: "blanket",
    57: "branch",
    58: "building",
    59: "bush",
    60: "cage",
    61: "cardboard",
    62: "carpet",
    63: "cloth",
    64: "clothes",
    65: "curtain",
    66: "desk",
    67: "dirt",
    68: "door",
    69: "fence",
    70: "floor",
    71: "flower",
    72: "fog",
    73: "food",
    74: "grass",
    75: "gravel",
    76: "hill",
    77: "house",
    78: "leaves",
    79: "light",
    80: "mat",
    81: "metal",
    82: "mirror",
    83: "moss",
    84: "mountain",
    85: "mud",
    86: "napkin",
    87: "net",
    88: "paper",
    89: "pavement",
    90: "pillow",
    91: "plant",
    92: "plastic",
    93: "platform",
    94: "playingfield",
    95: "railing",
    96: "railroad",
    97: "river",
    98: "road",
    99: "rock",
    100: "roof",
    101: "rug",
    102: "salad",
    103: "sand",
    104: "sea",
    105: "shelf",
    106: "sky",
    107: "skyscraper",
    108: "snow",
    109: "solid",
    110: "stairs",
    111: "stone",
    112: "straw",
    113: "structural",
    114: "table",
    115: "tent",
    116: "textile",
    117: "towel",
    118: "tree",
    119: "vegetable",
    120: "wall",
    121: "water",
    122: "window",
    123: "wood",
    124: "wreck",
    125: "yard",
    126: "curb",
    127: "grass field",
    128: "roadside",
    129: "vegetation",
    130: "pavement area",
    131: "building facade",
    132: "open space",
    133: "urban area",
    134: "suburban area",
    135: "rural area",
    136: "industrial area",
    137: "residential area",
    138: "storefront",
    139: "outdoor furniture",
    140: "signboard",
    141: "awning",
    142: "canopy",
    143: "utility pole",
    144: "electrical box",
    145: "fire escape",
    146: "sidewalk curb",
    147: "curb ramp",
    148: "bollard",
    149: "trash bin",
    150: "mailbox",
    151: "newsstand",
    152: "bicycle rack",
    153: "bus stop",
    154: "taxi stand",
    155: "vending machine",
    156: "ATM",
    157: "elevator"
}

extended_id2label = {**id2label, **COCO_STUFF_LABELS}

def identify_objects(embeddings, confidence_threshold=0.5):
    if embeddings is None:
        return []
    logits = torch.tensor(embeddings, dtype=torch.float)
    probs = torch.softmax(logits, dim=-1)
    confidences, predicted_class_ids = torch.max(probs, dim=-1)

    raw_objects = []
    for class_id, confidence in zip(predicted_class_ids.tolist(), confidences.tolist()):
        if confidence >= confidence_threshold:
            label = extended_id2label.get(class_id, f"Class_{class_id}")
            raw_objects.append((label, confidence))

    aggregated = {}
    for label, conf in raw_objects:
        if label not in aggregated:
            aggregated[label] = {"count": 0, "max_conf": 0.0}
        aggregated[label]["count"] += 1
        if conf > aggregated[label]["max_conf"]:
            aggregated[label]["max_conf"] = conf
    results = []
    for label, stats in aggregated.items():
        results.append((label, stats["count"], stats["max_conf"]))

    return results

records = []
for painting_id, embeddings in embeddings.items():
    objects_info = identify_objects(embeddings)
    if not objects_info:
        row_str = "None"
    else:
        row_str = "; ".join(
            f"{label} (count={count}, max_conf={conf:.2f})"
            for (label, count, conf) in objects_info
        )
    records.append({"Painting": painting_id, "Detected_Objects": row_str})

pd.DataFrame(records).to_csv("object_files/detected_images.csv", index=False)
