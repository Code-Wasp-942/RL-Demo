# 若 permission denied，则把脚本粘贴到 entry command 框内运行

set -euo pipefail
cd "$MAGNUS_HOME/workspace/repository"

pip install magnus-sdk --quiet

OUTDIR="runs/torchrun_n{n}_m{m}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

torchrun --standalone --nproc_per_node=2 ppo_train.py --total_steps 10000 --ckpt-path "$OUTDIR/ppo_ckpt.pt"

export OUTDIR
python - <<'PY'
import os
import magnus

outdir = os.environ["OUTDIR"]
secret = magnus.custody_file(outdir)
line = f"file secret: {{secret}}"

print(line, flush=True)
with open(os.environ["MAGNUS_RESULT"], "w", encoding="utf-8") as f:
    f.write(line + "\\n")
PY