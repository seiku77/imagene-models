"""models.json の設定に従って元の単一ファイルcheckpointをダウンロードし、パスを出力する"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from huggingface_hub import HfApi, hf_hub_download

model_id = sys.argv[1]
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "models.json")))[model_id]
api = HfApi()
for repo in cfg["repos"]:
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        print(f"skip {repo}: {e}", file=sys.stderr)
        continue
    for f in cfg["prefer"]:
        if f in files:
            print(hf_hub_download(repo, f))
            sys.exit(0)
    cands = [f for f in files if "/" not in f and f.endswith(".safetensors") and "inpaint" not in f.lower()]
    if cands:
        print(hf_hub_download(repo, sorted(cands)[0]))
        sys.exit(0)
sys.exit("not found")
