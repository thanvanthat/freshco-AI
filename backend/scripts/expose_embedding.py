"""Expose MobileNetV2's 1280-d image embedding as a second graph output.

The trained heads used to sit on the graph's 1000 ImageNet logits. Those are
the network's answer to a different question -- "which of these 1000 objects?"
-- and when that answer points the wrong way, a layer on top cannot recover:
the raw-protein head read mouldy bread as raw meat at p=1.000 because ImageNet
read it as rotisserie. The pooled embedding one layer earlier still holds the
colour and texture the logits throw away.

This adds that tensor as an output named ``embedding`` after the existing
``output``, so callers reading the first output are unaffected. Run once; the
modified graph is committed. Needs ``pip install onnx`` (a build tool only --
the API never imports it).

    python scripts/expose_embedding.py
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

MODEL = Path(__file__).resolve().parents[1] / "models" / "mobilenetv2-12.onnx"

#: The flattened global-average-pool output feeding the final Gemm. Verified by
#: reading the graph: Reshape(464, 471) -> 472 -> Gemm -> output.
POOLED = "472"
EMBEDDING = "embedding"


def main() -> None:
    model = onnx.load(MODEL)
    graph = model.graph

    if any(o.name == EMBEDDING for o in graph.output):
        print("embedding output already present; nothing to do")
        return

    feeds_classifier = [n for n in graph.node if n.op_type == "Gemm" and POOLED in n.input]
    if not feeds_classifier:
        raise SystemExit(f"{POOLED!r} does not feed the classifier; graph has changed")

    graph.node.append(helper.make_node("Identity", [POOLED], [EMBEDDING]))
    graph.output.append(
        helper.make_tensor_value_info(EMBEDDING, TensorProto.FLOAT, ["batch_size", 1280])
    )
    onnx.checker.check_model(model)
    onnx.save(model, MODEL)
    print(f"added {EMBEDDING!r} output to {MODEL.name}")


if __name__ == "__main__":
    main()
