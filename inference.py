from dinov2_backbone import DinoV2
from dpt_decoder import create_dpt_decoder
import numpy as np
import torch


model_type = "s"
use_vit_with_registers = True
device = "cpu"

dinov2 = DinoV2(
    model_type=model_type,
    reg=use_vit_with_registers,
    device=device
)
dpt_decoder = create_dpt_decoder(
    model_type=model_type,
    output_dim=9
).to(device)

dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
embeddings = dinov2(dummy_image, transform=True)  # ([(embed_1, cls_token_1), ...])

# embedding_final_layer = embeddings[-1][0]
# embedding_final_layer.shape -> 1, 384, 37, 66

with torch.no_grad():
    decoder_output = dpt_decoder(embeddings)
# decoder_output.shape -> 1, 9, 518, 924
