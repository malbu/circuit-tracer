import argparse, math, pathlib, random, time, os, json, logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm

from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.transcoder.activation_functions import JumpReLU

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)




def set_global_seed(seed: int):
    """Set python, numpy and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




#  reads a .txt file with one SMILES per line




class SmilesStream(IterableDataset):


    def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast):
        self.path = path
        self.tok = tokenizer

    def __iter__(self):
        with open(self.path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                enc = self.tok(ln, add_special_tokens=True, return_tensors="pt")
                yield enc.input_ids[0]




def collate(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)


def sample_activations(model: GPT2LMHeadModel, layer: int, batch_ids: torch.Tensor):


    acts = []

    def hook(_module, inp, _out):
        # inp is a tuple with the MLP input as first element
        acts.append(inp[0].detach())

    handle = model.transformer.h[layer].mlp.register_forward_hook(hook)
    with torch.inference_mode():
        _ = model(
            batch_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
    handle.remove()
    return torch.cat(acts, dim=0)  # (B, T, D)


@torch.no_grad()
def initialise_transcoder(d_model: int, d_trans: int, layer: int) -> SingleLayerTranscoder:
    act_fn = JumpReLU(0.04, 0.1)
    trans = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_trans,
        activation_function=act_fn,
        layer_idx=layer,
        skip_connection=True,
    )
    trans.cuda()
    return trans


def train_transcoder_for_layer(
    model: GPT2LMHeadModel,
    data: DataLoader,
    layer: int,
    d_trans: int,
    n_steps: int,
    lr: float,
    save_every: int,
    out_dir: str,
):
    d_model = model.config.n_embd
    transcoder = initialise_transcoder(d_model, d_trans, layer)

    # ensure JumpReLU.threshold participates in optimisation even if it was
    # created as a plain tensor
    if not isinstance(transcoder.activation_function.threshold, torch.nn.Parameter):
        transcoder.activation_function.threshold = torch.nn.Parameter(
            transcoder.activation_function.threshold.data
        )

    opt = torch.optim.AdamW(transcoder.parameters(), lr=lr)

    for step, batch_ids in enumerate(tqdm(data, desc=f"Layer {layer}", total=n_steps)):
        if step >= n_steps:
            break
        batch_ids = batch_ids.cuda()
        acts = sample_activations(model, layer, batch_ids)  # (B,T,D)
        flat = acts.reshape(-1, d_model)  # treat every position as example
        enc = transcoder.encode(flat)
        recon = transcoder.decode(enc)
        loss_recon = torch.nn.functional.mse_loss(recon, flat)
        loss_sparse = (enc.abs().mean()) * 1e-4
        loss = loss_recon + loss_sparse
        opt.zero_grad()
        loss.backward()
        opt.step()

        if save_every > 0 and (step + 1) % save_every == 0:
            # checkpoint intermediate weights
            ckpt_path = Path(out_dir) / f"layer{layer}_step{step+1}.npz"
            np.savez(
                ckpt_path,
                W_enc=transcoder.W_enc.cpu().numpy(),
                W_dec=transcoder.W_dec.cpu().numpy(),
                b_enc=transcoder.b_enc.cpu().numpy(),
                b_dec=transcoder.b_dec.cpu().numpy(),
                W_skip=transcoder.W_skip.cpu().numpy()
                if transcoder.W_skip is not None
                else None,
                threshold=transcoder.activation_function.threshold.cpu().numpy(),
            )

        if step % 50 == 0:
            logger.info(
                f"layer {layer} step {step}: loss {loss.item():.4e} (recon {loss_recon.item():.4e})"
            )

    return transcoder.cpu()





def main():
    p = argparse.ArgumentParser(description="train SAE transcoders for every layer of a GPT-2 model so that circuit-tracer can analyse it.")
    p.add_argument("--model_dir", required=True, help="path to the fine-tuned GPT-2 dir")
    p.add_argument("--train_smiles", required=True, help="text file containing SMILES for activation sampling")
    p.add_argument("--out_dir", default="smiles_transcoders", help="dir to write .npz files & YAML registry")
    p.add_argument("--seed", type=int, default=2, help="")
    p.add_argument("--d_transcoder", type=int, default=16384, help="width of SAE hidden layer")
    p.add_argument("--steps", type=int, default=500, help="Training steps per layer")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_every", type=int, default=50, help="checkpoint transcoder weights every N steps; 0 to disable")
    args = p.parse_args()

    
    set_global_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    
    model = GPT2LMHeadModel.from_pretrained(args.model_dir, torch_dtype=torch.float32).cuda().eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)

    stream_dataset = SmilesStream(args.train_smiles, tokenizer)
    data_loader = DataLoader(
        stream_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    transcoders = {}

    for layer in range(model.config.n_layer):
        transcoder = train_transcoder_for_layer(
            model,
            data_loader,
            layer,
            args.d_transcoder,
            args.steps,
            args.lr,
            args.save_every,
            args.out_dir,
        )
        path = Path(args.out_dir) / f"layer{layer}.npz"
        np.savez(
            path,
            W_enc=transcoder.W_enc.cpu().numpy(),
            W_dec=transcoder.W_dec.cpu().numpy(),
            b_enc=transcoder.b_enc.cpu().numpy(),
            b_dec=transcoder.b_dec.cpu().numpy(),
            W_skip=transcoder.W_skip.cpu().numpy() if transcoder.W_skip is not None else None,
            threshold=transcoder.activation_function.threshold.cpu().numpy(),
        )
        transcoders[layer] = {
            "layer": layer,
            "filepath": str(path),
            "id": f"smiles-l{layer}"
        }

    # write YAML registry 
    yaml_dict = {
        "model_name": args.model_dir,
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "mlp.hook_out",
        "transcoders": list(transcoders.values()),
    }
    yaml_path = Path(args.out_dir) / "smiles_transcoders.yaml"
    with open(yaml_path, "w") as f:
        import yaml
        yaml.safe_dump(yaml_dict, f)
    logger.info(f"Finished! YAML registry written to {yaml_path}")


if __name__ == "__main__":
    main() 