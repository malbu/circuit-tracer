

# circuit tracing analysis for the GPT-2 model


from pathlib import Path
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files   
import torch, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MODEL_DIR = Path("my_gpt2_smiles_best")          
TRANSCODER_YAML = Path("smiles_transcoders/smiles_transcoders.yaml")

assert MODEL_DIR.exists(), f"{MODEL_DIR} not found "
assert TRANSCODER_YAML.exists(), "train transcoders first with smiles_transcoder_training.py"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





model = ReplacementModel.from_pretrained(
    str(MODEL_DIR),
    transcoder_set=str(TRANSCODER_YAML),
    device=device,
    dtype=torch.float32,
)
print("Loaded model with", model.cfg.n_layers, "layers and",
      model.d_transcoder, "transcoder dims / layer")



PROMPT = "C"   

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

