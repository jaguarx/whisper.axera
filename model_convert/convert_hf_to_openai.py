from transformers import (
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    WhisperTokenizerFast
)
import re
import torch
import whisper
from export_malaysian import get_args

def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text


# Load HF Model
args = get_args()
openai_name = args.model.split('-')[0]
model = WhisperForConditionalGeneration.from_pretrained(
    f'mesolitica/malaysian-whisper-{args.model}', 
    dtype = torch.float32,
).cpu().eval()

# Rename layers
hf_state_dict = model.state_dict()
for key in list(hf_state_dict.keys()):
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

model = whisper.load_model(openai_name)
dims = model.dims
# Save it
torch.save({
    "dims": model.dims.__dict__,
    "model_state_dict": hf_state_dict
}, f"whisper-malaysian-{args.model}.bin")
print(f"Save to whisper-malaysian-{args.model}.bin")