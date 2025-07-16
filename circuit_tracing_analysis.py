# ugly TL patch
# override the first positional arg

from transformers import modeling_utils as _mu

_orig_from_pretrained = _mu.PreTrainedModel.from_pretrained.__func__

def _fixed_from_pretrained(cls, *args, **kwargs):
    
    hf_path = kwargs.pop("hf_model_path", None)
    if hf_path is not None:
        # make it the first positional argument
        args = (hf_path,) if not args else (hf_path,) + tuple(args[1:])
    elif not args:
        raise TypeError("Need either a positional checkpoint or `hf_model_path=`")
    return _orig_from_pretrained(cls, *args, **kwargs)

_mu.PreTrainedModel.from_pretrained = classmethod(_fixed_from_pretrained)


#tell Transformer Lens to accept any local directory name

import transformer_lens.loading_from_pretrained as _tl_load
_tl_load.get_official_model_name = lambda name: name  # no whitelist check





from pathlib import Path
import glob, tempfile, shutil, os
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files   
import torch, logging
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers import AutoConfig




logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


MODEL_DIR  = Path("my_gpt2_smiles_best")

# Tell TL to use the finetuned checkpoint itself as the model
BASE_NAME  = str(MODEL_DIR)       # points at the directory with config + weights
TRANSCODER_YAML = Path("smiles_transcoders/smiles_transcoders.yaml")

assert MODEL_DIR.exists(), f"{MODEL_DIR} not found "
assert TRANSCODER_YAML.exists(), "train transcoders first with smiles_transcoder_training.py"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_safetensors(checkpoint_dir: Path):

    if list(checkpoint_dir.glob("*.safetensors")):
        return  # Nothing to do.

    bin_files = list(checkpoint_dir.glob("*.bin"))
    if not bin_files:
        return  

    logging.info(f"Converting {len(bin_files)} .bin file(s) in {checkpoint_dir} "
                 f"to safetensors …")

   
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    out_path = checkpoint_dir / "model.safetensors"
    save_file(model.state_dict(), str(out_path))

    # clean up old binary files
    for bf in bin_files:
        bf.unlink()
    logging.info(f"Saved {out_path} and removed old .bin file(s).")


ensure_safetensors(MODEL_DIR)

#  ensure the local GPT22 config exposes n_ctx for Transformer‑Lens

cfg = AutoConfig.from_pretrained(MODEL_DIR)

if not hasattr(cfg, "n_ctx"):
    # use the field that *is* present
    cfg.n_ctx = getattr(cfg, "n_positions", None) \
                or getattr(cfg, "max_position_embeddings", None) \
                or 1024            # sensible default in case neither exists
    cfg.save_pretrained(MODEL_DIR) # writes the updated config.json
    logging.info(f"Added n_ctx={cfg.n_ctx} to {MODEL_DIR}/config.json")


#  make ReplacementModel resilient when the model has no rotary masks

from circuit_tracer.replacement_model import ReplacementModel

def _safe_deduplicate_attention_buffers(self):
    """
    skip the whole routine unless the first layer
    actually exposes rotary_sin
    """
    first_attn = self.blocks[0].attn
    if not hasattr(first_attn, "rotary_sin"):
        # nothing to deduplicate GPT2 style positional encodings
        return

    
    shared_sin = shared_cos = None
    for block in self.blocks:
        if shared_sin is None:                 # first time we see them
            shared_sin, shared_cos = block.attn.rotary_sin, block.attn.rotary_cos
        else:                                  # point every later layer to the same tensors
            block.attn.rotary_sin = shared_sin
            block.attn.rotary_cos = shared_cos
    


ReplacementModel._deduplicate_attention_buffers = _safe_deduplicate_attention_buffers

model = ReplacementModel.from_pretrained(
    BASE_NAME,                    # local directory 
    transcoder_set=str(TRANSCODER_YAML),
    device=device,
    dtype=torch.float32,
)
print("Loaded model with", model.cfg.n_layers, "layers and",
      model.d_transcoder, "transcoder dims / layer")



#PROMPT = "Cn1ncc2c(NC(=O)NCc3cc(C(F)(F)F)nn3-c3ccc(F)cc3)cccc21"   
PROMPT = "Cn1ncc2c(NC(=O)NCc3cc(C(F"
#
# mirrors Anthropic default CLI
#
with torch.enable_grad():           # attribution needs gradients
    graph = attribute(
        prompt=PROMPT,
        model=model,
        max_n_logits=10,
        desired_logit_prob=0.95,
        batch_size=256,
        offload="cpu",              
        verbose=True,
    )

graph_path = Path("graphs")
graph_path.mkdir(exist_ok=True)
pt_file = graph_path / "prompt_result.pt"
graph.to_pt(str(pt_file))
print("Raw graph saved: ", pt_file)

create_graph_files(
    graph_or_path=str(pt_file),
    slug="smiles-fromprompt",            
    output_path=str(graph_path),
)
print("JSON written to", graph_path)
