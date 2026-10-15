"""Write a Kaggle script kernel that runs Crossfire conversations on two T4 GPUs.

    python kaggle/build.py --owner USER --slug zod-strike --commit HASH \
        --run "--arms strike_big_real,strike_big_sham --per-arm 640 --batch 16 --seed-base 40000" --name strike

The kernel clones Zap-Or-Deal at the given commit, copies the calibration into the output
folder, continues from the conversations of any earlier session attached as an input, and runs
scripts/run.py with the PyTorch engine under a time budget. Results go to
/kaggle/working/results/followups/<name>/, which Kaggle keeps as output.
Push with: kaggle kernels push -p kaggle/push/<slug> --accelerator NvidiaTeslaT4
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--owner", required=True, help="Kaggle username that owns the kernel")
ap.add_argument("--slug", required=True)
ap.add_argument("--commit", required=True)
ap.add_argument("--run", required=True, help="arguments for scripts/run.py")
ap.add_argument("--name", required=True, help="folder under results/followups/")
ap.add_argument("--hours", type=float, default=11.3)
ap.add_argument("--resume-from", nargs="*", default=[], help="owner/slug of earlier sessions to continue from")
a = ap.parse_args()

src = f'''"""Crossfire on Kaggle: {a.slug}. Built by kaggle/build.py."""
import os, subprocess, sys, time
from pathlib import Path
START = time.time()
BUDGET = {a.hours}
CODE = "/tmp/zap-or-deal"
OUT = Path("/kaggle/working/results")
COMMIT = "{a.commit}"
NAME = "{a.name}"
RUN = {json.dumps(a.run)}

def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode

sh(f"git clone -q https://github.com/pragyaangaur/Zap-Or-Deal {{CODE}} && git -C {{CODE}} checkout -q {{COMMIT}}")
sh('pip install -q "transformers==5.18.0" "accelerate==1.15.0"')
sh(f"git -C {{CODE}} log -1 --format='code at %H %s'")
(OUT / "followups" / NAME).mkdir(parents=True, exist_ok=True)
sh(f"cp -r {{CODE}}/results/calibration {{OUT}}/")
# Continue from earlier sessions attached as inputs, keeping the longest copy.
dst = OUT / "followups" / NAME / "conversations.jsonl"
for f in Path("/kaggle/input").rglob("conversations.jsonl"):
    if f"followups/{{NAME}}/" not in str(f):
        continue
    n = sum(1 for _ in open(f))
    if not dst.exists() or n > sum(1 for _ in open(dst)):
        dst.write_bytes(f.read_bytes())
        print("resumed from", f, n, "conversations", flush=True)
import torch
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
env = dict(os.environ, CROSSFIRE_RESULTS=str(OUT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
left = BUDGET - (time.time() - START) / 3600
code = sh(f"cd {{CODE}} && {{sys.executable}} scripts/run.py --backend torch {{RUN}} --out {{dst}} --max-hours {{left - 0.25:.2f}}", env=env)
print("exit code", code, flush=True)
sh(f"wc -l {{dst}}")
'''

push = HERE / "push" / a.slug
push.mkdir(parents=True, exist_ok=True)
(push / "run.py").write_text(src)
(push / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.owner}/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "run.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": a.resume_from,
    "model_sources": []}, indent=1))
print("wrote", push)
