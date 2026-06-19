"""Download dacorvo/mnist-mlp from Hugging Face and export TF.js-loadable weights."""
from pathlib import Path
import json

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def main() -> None:
    repo_id = "dacorvo/mnist-mlp"
    out_dir = Path("public/model")
    out_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_file(safetensors_path)

    layers = {
        "input": {
            "weight": state_dict["input_layer.weight"].numpy().astype("float32"),
            "bias": state_dict["input_layer.bias"].numpy().astype("float32"),
        },
        "mid": {
            "weight": state_dict["mid_layer.weight"].numpy().astype("float32"),
            "bias": state_dict["mid_layer.bias"].numpy().astype("float32"),
        },
        "output": {
            "weight": state_dict["output_layer.weight"].numpy().astype("float32"),
            "bias": state_dict["output_layer.bias"].numpy().astype("float32"),
        },
    }

    manifest = {}
    for layer_name, tensors in layers.items():
        manifest[layer_name] = {}
        for tensor_name, arr in tensors.items():
            filename = f"{layer_name}_{tensor_name}.dat"
            (out_dir / filename).write_bytes(arr.tobytes())
            manifest[layer_name][tensor_name] = {
                "file": filename,
                "shape": list(arr.shape),
            }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Exported model weights to {out_dir}")


if __name__ == "__main__":
    main()
