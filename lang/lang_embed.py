from diffusion_policy_3d.CLIP.clip import build_model, load_clip, tokenize
import torch


def _load_clip(device):
    model, _ = load_clip("RN50", device, jit=False)
    clip_rn50 = build_model(model.state_dict()).to(device)
    del model
    return clip_rn50

def encode_text(x, clip_rn50, device):
    with torch.no_grad():
        tokens = tokenize([x]).to(device)
        text_feat, text_emb = clip_rn50.encode_text_with_embeddings(tokens)

    text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
    return text_feat, text_emb, text_mask

def get_lang_embed():
    device = torch.device('cuda:0')
    clip_rn50 = _load_clip(device=device)
    # x = "screw in the yellow light bulb"
    x = "what sup my boy"
    text_feat, text_emb, text_mask = encode_text(x=x, clip_rn50=clip_rn50, device=device)

    # text_feat (1, 1024)
    # text_emb (77, 512)
    # import pdb;pdb.set_trace()
    
if __name__ == "__main__":
    get_lang_embed()