# ugly TL patch
# override the first positional arg

from transformers import modeling_utils as _mu

_orig_from_pretrained = _mu.PreTrainedModel.from_pretrained.__func__

def _fixed_from_pretrained(cls, *args, **kwargs):
    # steal the TL‑style kwarg if present
    hf_path = kwargs.pop("hf_model_path", None)
    if hf_path is not None:
        # make it the first positional argument (repo / dir path)
        args = (hf_path,) if not args else (hf_path,) + tuple(args[1:])
    elif not args:
        raise TypeError("Need either a positional checkpoint or `hf_model_path=`")
    return _orig_from_pretrained(cls, *args, **kwargs)

_mu.PreTrainedModel.from_pretrained = classmethod(_fixed_from_pretrained)


#tell Transformer Lens to accept any local directory name
# ------------------------------------------------------------------
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

# ─────────────────────────────────────────────────────────────────────────────
# Additional imports for CLT-based interventions
# ─────────────────────────────────────────────────────────────────────────────
import argparse, json
from transformers import GPT2LMHeadModel, AutoTokenizer
# from anthropic_steering import constrained_patch   # still optional



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# 1)  From now on we let ReplacementModel load the CLT itself, via YAML.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 2)  new CLI flags for optional post-attribution intervention
# ─────────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--patch_feature", type=int, default=None,
                help="feature index to steer (global idx, as printed in graph file)")
ap.add_argument("--patch_factor", type=float, default=-1.0,
                help="multiplicative factor for constrained patching")
ap.add_argument("--patch_layers", default="all",
                help="e.g. 5:12  or  all  (affects decoder layers)")

# Ignore unknown flags so that existing hard-coded vars still work.
args, _unknown = ap.parse_known_args()


# ------------------------------------------------------------------
# Local fine‑tuned checkpoint (weights + config live in the same dir)
# ------------------------------------------------------------------
MODEL_DIR  = Path("my_gpt2_smiles_best")

# Tell TL to use the finetuned checkpoint itself as the “model”.
BASE_NAME  = str(MODEL_DIR)       # points at the directory with config + weights
TRANSCODER_YAML = Path("clt_gpcr/clt_registry.yaml")

assert MODEL_DIR.exists(),        f"{MODEL_DIR} not found"
assert TRANSCODER_YAML.exists(),  "Create the YAML envelope first (see diff)."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helper: make sure the finetuned checkpoint lives in *.safetensors format.
# ---------------------------------------------------------------------------
def ensure_safetensors(checkpoint_dir: Path):
    """
    If `checkpoint_dir` only contains *.bin weight files, convert them to a
    single `model.safetensors` file and delete the original *.bin files.
    """
    if list(checkpoint_dir.glob("*.safetensors")):
        return  # Nothing to do.

    bin_files = list(checkpoint_dir.glob("*.bin"))
    if not bin_files:
        return  # No weights at all – let the usual error handling deal with it.

    logging.info(f"Converting {len(bin_files)} .bin file(s) in {checkpoint_dir} "
                 f"to safetensors …")

    # Load with Transformers (weights_only=True avoids optimiser states).
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    out_path = checkpoint_dir / "model.safetensors"
    save_file(model.state_dict(), str(out_path))

    # Clean up old binary files.
    for bf in bin_files:
        bf.unlink()
    logging.info(f"Saved {out_path} and removed old .bin file(s).")

# Ensure the local finetuned checkpoint is safe before we import it.
ensure_safetensors(MODEL_DIR)

# --------------------------------------------------------------------
#  Make sure the local GPT‑2 config exposes `n_ctx` for Transformer‑Lens
# --------------------------------------------------------------------
cfg = AutoConfig.from_pretrained(MODEL_DIR)

if not hasattr(cfg, "n_ctx"):
    # use the field that *is* present
    cfg.n_ctx = getattr(cfg, "n_positions", None) \
                or getattr(cfg, "max_position_embeddings", None) \
                or 1024            # sensible default in case neither exists
    cfg.save_pretrained(MODEL_DIR) # writes the updated config.json
    logging.info(f"Added n_ctx={cfg.n_ctx} to {MODEL_DIR}/config.json")

# ------------------------------------------------------------------
#  Make ReplacementModel resilient when the model has no rotary masks
# ------------------------------------------------------------------
from circuit_tracer.replacement_model import ReplacementModel

def _safe_deduplicate_attention_buffers(self):
    """
    Original helper fails on models that don't use rotary embeddings
    (e.g. GPT‑2).  We skip the whole routine unless the first layer
    actually exposes `rotary_sin`.
    """
    first_attn = self.blocks[0].attn
    if not hasattr(first_attn, "rotary_sin"):
        # nothing to deduplicate – GPT‑2 style positional encodings
        return

    # --- original logic (condensed) --------------------------------
    shared_sin = shared_cos = None
    for block in self.blocks:
        if shared_sin is None:                 # first time we see them
            shared_sin, shared_cos = block.attn.rotary_sin, block.attn.rotary_cos
        else:                                  # point every later layer to the same tensors
            block.attn.rotary_sin = shared_sin
            block.attn.rotary_cos = shared_cos
    # ----------------------------------------------------------------

# ─── Attach the safer version (plain assignment; Python binds automatically) ──
ReplacementModel._deduplicate_attention_buffers = _safe_deduplicate_attention_buffers

model = ReplacementModel.from_pretrained(
    model_name=str(MODEL_DIR),          # finetuned GPT-2 directory
    transcoder_set=str(TRANSCODER_YAML),# <- YAML registry of per-layer SAEs
    device=device,
    dtype=torch.float32,
    #freeze_attn_patterns=True,  # ensure linear attribution assumptions
    #freeze_ln=True,
)
print("Loaded model with", model.cfg.n_layers, "layers and",
      model.d_transcoder, "transcoder dims / layer")



PROMPT = "Cn1ncc2c(NC(=O)NCc3cc(C(F)(F)F)nn3-c3ccc(F)cc3)cccc21"   
#PROMPT = "Cn1ncc2c(NC(=O)NCc3cc(C(F"
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

# ─────────────────────────────────────────────────────────────────────────────
# 3)  after graph export → optional constrained-patch validation
# ─────────────────────────────────────────────────────────────────────────────

if args.patch_feature is not None:
    # Determine layer range from CLI arg
    layers = None if args.patch_layers == "all" else \
             list(range(*map(int, args.patch_layers.split(":"))))

    # Tokenise prompt to obtain ids / mask
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    enc = tokenizer(PROMPT, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    # Use the previously built replacement model (or user-provided one)
    repl_model = model

    delta_logp, delta_act = constrained_patch(
        repl_model,
        input_ids=ids,
        attention_mask=mask,
        feature_idx=args.patch_feature,
        mult=args.patch_factor,
        layer_range=layers,
    )

    print("=== Patching summary ===")
    print(f"Δlog p(target token) = {delta_logp:+.4e}")

    with open(graph_path / "intervention.json", "w") as f:
        json.dump({
            "delta_logp": float(delta_logp),
            "delta_act": {k: float(v) for k, v in delta_act.items()}
        }, f, indent=2)
