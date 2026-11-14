"""Hugging Face から単一ファイルのcheckpointを取得し、MediaPipe変換スクリプト用の .ckpt にする"""
import json, sys, os
from huggingface_hub import HfApi, hf_hub_download

model_id, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "models.json")))[model_id]
api = HfApi()


def pick(files):
    ckpts = [f for f in files if "/" not in f and f.endswith((".safetensors", ".ckpt"))
             and "inpaint" not in f.lower() and "vae" not in f.lower().replace("novae", "")]
    for p in cfg["prefer"]:
        if p in ckpts:
            return p
    def score(f):
        l = f.lower()
        return (("emaonly" in l or "pruned" in l or "no-ema" in l), f.endswith(".safetensors"))
    return max(ckpts, key=score) if ckpts else None


src = None
for repo in cfg["repos"]:
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        print(f"skip {repo}: {e}")
        continue
    f = pick(files)
    if f:
        src = (repo, f)
        break
    print(f"no single-file checkpoint in {repo}")
if not src:
    sys.exit("checkpointが見つかりません")

repo, fname = src
print(f"using {repo}/{fname}")
path = hf_hub_download(repo, fname)
import torch
if fname.endswith(".safetensors"):
    from safetensors.torch import load_file
    sd = load_file(path)
else:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
torch.save({"state_dict": sd}, out)
json.dump({"repo": repo, "file": fname}, open(out + ".source.json", "w"))
print("keys:", len(sd))
