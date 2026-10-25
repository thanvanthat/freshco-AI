# Models

Every model FreshCO AI runs, what it is for, and what it cannot do. All inference
is ONNX Runtime on CPU — there is no TensorFlow or PyTorch at runtime, which is
what keeps the whole backend inside the 250 MB limit for a serverless function.

| File | Role | Size | Runs when |
|---|---|---|---|
| `mobilenetv2-12.onnx` | Image classification, plus a 1280-d embedding output | 14.0 MB | Every `/analyze` and `/food/identify` |
| `imagenet_class_index.json` | Class id → label table | 35 KB | With the above |
| `yolox_tiny.onnx` | Object detection + location | 20.2 MB | `/detect`, and `/analyze` when classification fails |
| `freshness_head.npz` | Fresh-vs-rotten head on the embedding | ~10 KB | Scoring fruit and vegetables in `/analyze` and `/detect` |
| `apple_pomegranate_head.npz` | Apple-vs-pomegranate head on the embedding | ~10 KB | When ImageNet answers "pomegranate" |
| `head.npz` + `labels.txt` | Custom multi-class head | — | Only if you train one (not committed) |

---

## 1. MobileNetV2 — image classification

**What it does.** Takes the whole photo and answers "what food is this?".
Outputs 1000 ImageNet class scores; `IMAGENET_FOOD_MAP` in
`app/vision/classifier.py` narrows those to foods FreshCO AI holds data for.

**Source.** ONNX Model Zoo, `mobilenetv2-12`. ImageNet-1k pretrained weights.

**Measured.** Banana, broccoli and lemon each identified at 1.00 confidence
against the deployed backend.

**Cannot identify:** tomato, potato, onion, spinach, carrot, mango, guava, or
any raw meat, fish or dairy. ImageNet-1k has no class for them. This is a
property of the training set, not a tuning problem — no preprocessing change
fixes it. Those photos return `identified: false` rather than a guess.

**Input.** 224×224 RGB, NCHW, normalised with the torchvision mean/std. Using
Keras-style `[-1, 1]` scaling instead would not error; it would silently
degrade every prediction.

**Second output, `embedding`.** The 1280-d pooled features one layer before
the classifier, added by `scripts/expose_embedding.py`. Heads trained here use
it rather than the 1000 logits: the logits are ImageNet's answer to a
different question, and a head on top cannot overrule them -- which is how the
raw-protein head below read mouldy bread as raw meat.

---

## 2. MobileNetV2 + custom classifier

A linear layer trained on top of the frozen MobileNetV2 output. This is
ordinary transfer learning: the frozen features already encode colour, texture
and shape, so the head only separates the classes. It trains on CPU in seconds
from a few hundred images per class and adds kilobytes, not megabytes.

### 2a. `freshness_head.npz` — shipped

**What it does.** Estimates whether a fruit or vegetable is fresh or rotten,
as a probability. It refines the OpenCV freshness formula; it does not replace
it, and it is not consulted for bread, dairy, meat or cooked food.

**Trained on** 2,700 photos of fresh and rotten apple, banana, bitter gourd,
capsicum, cucumber, okra, orange, potato and tomato, sampled from
[Densu341/Fresh-rotten-fruit](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit)
(OpenRAIL). `scripts/collect_fresh_rotten.py` reproduces the sample;
`scripts/train_freshness_head.py` reproduces the head.

**Measured.** Two numbers, and only the second one matters:

| Test set | Result |
|---|---|
| Held-out photos from the training source | 96.5% accuracy |
| 480 **independently photographed** fruits ([Project-AgML](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification), CC-BY-4.0, 5 of 8 kinds never seen in training) | AUC 0.83, against 0.69 for the formula alone |

The training photos include rotated and padded augmentations, and both sets
are studio shots on plain backgrounds. Its probabilities sit near 0 or 1 even
when wrong, and on the independent set it called ~40% of sound fruit rotten.

**How the app uses it, because of that.** A confident "rotten" (≥0.9) caps the
score at 60 — *nearly spoiled, inspect it*, never *spoiled*. A confident
"sound" (≤0.1) lets an item the formula found undamaged read fresh, so shadows
and shine stop costing it the label. On the independent set this cut rotten
fruit shown as **fresh** from 29.5% to 5.2%, with sound fruit shown as
**spoiled** unchanged at 0.5%. Visible mould still reads spoiled through the
formula's own total-loss rule.

### 2b. `apple_pomegranate_head.npz` — shipped

**Why it exists.** ImageNet-1k's only apple is the green Granny Smith, so
MobileNetV2 files red apples under "pomegranate": 37 of 60 independently
photographed apples, at a median confidence of 0.87. It is consulted only when
ImageNet says "pomegranate", and decides between the two.

**Trained on** 320 Project-AgML apple and pomegranate photos, from rows that do
not overlap the evaluation sample (`scripts/train_apple_pomegranate_head.py`
has the exact command).

**Measured, end to end through the classifier:**

| Photos | Before | With the head |
|---|---|---|
| 60 held-out AgML apples | Apple 2, **Pomegranate 37** | **Apple 38, Pomegranate 0** |
| 60 held-out AgML pomegranates | Pomegranate 58 | Pomegranate 57 |
| 15 Wikimedia pomegranates (other source) | Pomegranate 13 | Pomegranate 8, "apple or pomegranate?" 6, **Apple 1** |

Below 0.8 confidence either way it names neither and offers both, which is why
more of the outside-source pomegranates now ask. Unlike fresh-vs-rotten, this
distinction held across sources: 92% of 500 apples from Densu341 were called
apple by the head alone.

**Not shipped: a food-name head.** The same data trains one that is 97% right
on the foods it knows — and, at ≥0.95 confidence, misnames 25% of foods it has
never seen (strawberry → bitter gourd, pomegranate → apple). A closed set of
nine always answers one of nine, so naming stays with ImageNet, the detector
and the vision model.

**Removed: `protein_hint.npz`.** A raw-meat-or-not head that tested at 96.6%
precision on photos from its own source, then called mouldy bread raw meat at
p=1.000. Removed on 30 September 2026; it is why the freshness head is judged
on a second, independent dataset.

### 2c. `head.npz` — yours to train

Train one to add classes the stock model cannot name:

```bash
# data/chicken/*.jpg, data/mutton/*.jpg, ...
python scripts/train_food_head.py --data data --out models
```

Drop `head.npz` and `labels.txt` here and the API switches from `imagenet` to
`custom` mode on the next start. **No head is committed**: a model is only as
trustworthy as the photos behind it, and for raw meat that is a food-safety
question rather than a convenience one.

Roughly 100–300 photos per class is a sensible floor. Shoot them as the app
will see them — varied lighting, backgrounds and angles, including
part-wrapped and on a plate. A set shot in one session on one worktop teaches
the head to recognise your worktop.

---

## 3. YOLOX-Tiny — object detection

**What it does.** Finds *where* each food is, so a fridge shelf becomes
several items rather than one averaged score. Each box is cropped and put
through the same measurement and scoring path as a single scan.

**Source.** [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) release
`0.1.1rc0`, `yolox_tiny.onnx`, COCO-trained. Apache-2.0.

**Detects** banana, apple, orange, broccoli and carrot (mapped to reference
data), plus sandwich, pizza, hot dog, donut and cake (detected, not scored).
Everything else COCO knows — people, cutlery, furniture — is ignored, because
a box around a fork is noise on a food scan.

**Measured** on the same fruit photo, against the SSD MobileNet v1 it replaced:

| | SSD MobileNet v1 | YOLOX-Tiny |
|---|---|---|
| Model size | 29.3 MB | **20.2 MB** |
| Distinct items found | 7 | **12** |
| Top confidence | 0.86 | **0.88** |
| Warm latency | 40–60 ms | 220–260 ms |

YOLOX is slower because decoding and NMS run in NumPy rather than inside the
graph, which is the trade for finding nearly twice as many items correctly.
Still well inside a scan's budget.

### Post-processing, and why it is not optional

Unlike SSD, YOLOX emits **no NMS and no absolute coordinates**. Three steps
happen in `detector.py`, and each fails silently rather than loudly:

1. **Letterbox** onto a 416×416 canvas padded with 114, preserving aspect
   ratio. Stretching to a square instead would squash every object. Pixels
   stay raw BGR 0–255 — applying the ImageNet mean/std the classifier needs
   would not error, it would just make every prediction wrong.
2. **Decode** grid-relative offsets: `(offset + cell) × stride` for centres,
   `exp(raw) × stride` for sizes, across strides 8/16/32 (3549 anchors).
   Skipping this clusters every box in the top-left corner.
3. **NMS**, via `deduplicate` at IoU 0.55. A six-fruit photo produces 110 raw
   boxes; without this the user would see dozens of duplicates of one apple.

**Cannot detect** tomato, potato, onion or raw meat: COCO has no class for
them, the same wall the classifier hits.

---

## 4. Why YOLOX rather than YOLOv8n

YOLOv8n was the original choice and was rejected for two concrete reasons.

**No public ONNX build exists.** Every Hugging Face repo hosting YOLOv8n ships
PyTorch `.pt` only. Producing an ONNX needs `ultralytics` plus `torch`, about
2.5 GB installed, purely as a build-time step for a 12 MB artefact.

**Licensing.** Ultralytics YOLOv5/v8 weights are AGPL-3.0. On a public
repository that carries obligations the Apache-2.0 weights here do not.

YOLOX is the same family of single-stage detector, is **Apache-2.0**, and
publishes ONNX directly — no conversion step and no 2.5 GB dependency. Tiny
(20 MB) is used; Nano (3.7 MB) is available from the same release if cold-start
time ever matters more than accuracy.

SSD MobileNet v1, which YOLOX replaced, was Apache-2.0 and worked. It was
swapped out because YOLOX is 9 MB smaller and found 12 items where SSD found
7 on the same photo.

---

## Licensing of what ships here

| Artefact | Origin | Licence |
|---|---|---|
| `mobilenetv2-12.onnx` | ONNX Model Zoo | Apache-2.0 |
| `yolox_tiny.onnx` | Megvii YOLOX release 0.1.1rc0 | Apache-2.0 |
| `imagenet_class_index.json` | TensorFlow/Keras | Apache-2.0 |
| `mobilenetv2-12.onnx` `embedding` output | Added here, same weights | Apache-2.0 |
| `freshness_head.npz` | Trained here on Densu341/Fresh-rotten-fruit | See below |
| `apple_pomegranate_head.npz` | Trained here on Project-AgML | CC-BY-4.0 data; see below |

The freshness head was trained on a sample of a dataset published under the
OpenRAIL licence, and evaluated (not trained) on Project-AgML's CC-BY-4.0
photos, credited to Sultana, Jahan & Uddin (2022), *Data in Brief* 44:108552.
No training photos are committed. Whether a trained weight is a derivative
work of its training images is unsettled, and the answer may differ by
jurisdiction. If FreshCO AI is ever distributed commercially, retrain the head
on photos you own -- which is worth doing anyway, since phone photos from real
kitchens will outperform studio shots.

Verify these licences yourself before relying on them commercially. They are
recorded here as a starting point, not legal advice.
