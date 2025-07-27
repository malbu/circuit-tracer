import argparse
from pathlib import Path
from typing import Union, List

import torch
from safetensors.torch import save_file
import yaml


def _extract_num_layers(enc_state_dict) -> int:
    """Infer the number of layers from the encoder state-dict keys."""
    # keys look like "0.weight", "0.bias", "1.weight", …
    layer_ids: List[int] = []
    for k in enc_state_dict.keys():
        if ".weight" in k and k.split(".")[0].isdigit():
            layer_ids.append(int(k.split(".")[0]))
    if not layer_ids:
        raise ValueError("Could not infer number of layers from encoder state-dict!")
    return max(layer_ids) + 1


def convert_clt_checkpoint(
    clt_ckpt_path: Union[str, Path],
    out_dir: Union[str, Path],
    model_name: str,
    yaml_path: Union[str, Path] | None = None,
    feature_input_hook: str = "mlp.hook_in",
    feature_output_hook: str = "mlp.hook_out",
    device: str | torch.device = "cpu",
) -> Path:
    """Unroll a multi-layer CLT checkpoint into individual per-layer SAE weight
    files plus a YAML registry compatible with circuit-tracer.

    Args:
        clt_ckpt_path: Path to the `.safetensors` / `.pt` file saved via
            ``CrossLayerTranscoder.save``.
        out_dir: Directory in which to write the generated `clt_layer{L}.safetensors` files
            and the YAML registry.
        model_name: Name or path of the underlying language model. Stored in YAML.
        yaml_path: Optional explicit path for the YAML file. Defaults to
            ``out_dir/clt_registry.yaml``.
        feature_input_hook/feature_output_hook: Hook names expected by the loader.
        device: Torch device to load large tensors onto (defaults to CPU).

    Returns:
        Path to the generated YAML registry.
    """
    clt_ckpt_path = Path(clt_ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (torch.save format)
    ckpt = torch.load(str(clt_ckpt_path), map_location=device)
    enc_sd = ckpt["encoders"]
    dec_sd = ckpt["decoders"]

    n_layers = _extract_num_layers(enc_sd)
    # All layers share the same feature dimension n_feat
    sample_key = next(iter(enc_sd))
    n_feat = enc_sd[sample_key].shape[0]  # (n_feat, d_model)

    # Dummy to get d_model from corresponding decoder weight
    sample_dec_key = f"0_0.weight"
    d_model = dec_sd[sample_dec_key].shape[0]

    transcoder_entries = []

    for l in range(n_layers):
        W_enc = enc_sd[f"{l}.weight"].clone().contiguous()  # (n_feat, d_model)
        b_enc = enc_sd[f"{l}.bias"].clone().contiguous()    # (n_feat,)
        W_dec = dec_sd[f"{l}_{l}.weight"].clone().contiguous()  # (d_model, n_feat)
        b_dec = dec_sd[f"{l}_{l}.bias"].clone().contiguous()    # (d_model,)

        W_skip = torch.zeros(d_model, d_model, dtype=W_enc.dtype)

        layer_file = out_dir / f"clt_layer{l}.safetensors"
        save_file(
            {
                "W_enc": W_enc,
                "W_dec": W_dec,
                "b_enc": b_enc,
                "b_dec": b_dec,
                "W_skip": W_skip,
            },
            str(layer_file),
        )

        transcoder_entries.append(
            {
                "id": f"clt-l{l}",
                "layer": l,
                "filepath": str(layer_file.resolve()),
                "d_transcoder": n_feat,
            }
        )

    yaml_dict = {
        "model_name": model_name,
        "d_sae": n_feat,  # legacy field used by some tooling
        "scan": False,
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "transcoders": transcoder_entries,
    }

    yaml_path = Path(yaml_path) if yaml_path is not None else (out_dir / "clt_registry.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a multi-layer CLT checkpoint into per-layer SAE files + YAML registry.")
    parser.add_argument("--ckpt", required=True, help="Path to clt.safetensors or .pt file")
    parser.add_argument("--out_dir", default="clt_gpcr", help="Output directory for layer files and YAML")
    parser.add_argument("--model_name", required=True, help="Name or path of the underlying LM")
    parser.add_argument("--yaml_path", default=None, help="Optional custom path for the YAML registry")
    args = parser.parse_args()

    yaml_out = convert_clt_checkpoint(args.ckpt, args.out_dir, args.model_name, args.yaml_path)
    print("✓ Conversion complete. YAML registry at", yaml_out) 