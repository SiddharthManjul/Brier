#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
usage='Usage: gcp_spot.sh up <pusht|aloha> <timing|baselines|pairs|pilot> <first_shard> <n_vms> <n_shards> [machine] | status | down'
[[ $# -gt 0 ]] || { echo "$usage" >&2; exit 2; }
case "$1" in
  up) [[ $# -eq 6 || $# -eq 7 ]] || { echo "$usage" >&2; exit 2; } ;;
  status|down) [[ $# -eq 1 ]] || { echo "$usage" >&2; exit 2; } ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
: "${GCP_PROJECT:?Set GCP_PROJECT to the Google Cloud project id}"
[[ "$GCP_PROJECT" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || { echo 'invalid GCP_PROJECT id' >&2; exit 2; }
command -v gcloud >/dev/null || { echo 'gcloud is required' >&2; exit 2; }
if [[ "$1" != down ]]; then
  [[ "${BRIER_DATA:-}" =~ ^gs://[a-z0-9][a-z0-9_.-]*(/[A-Za-z0-9._/-]*)?$ ]] || { echo 'BRIER_DATA must be a gs:// bucket or simple object prefix' >&2; exit 2; }
fi
case "$1" in
up)
  [[ "$2" == pusht || "$2" == aloha ]] || { echo 'env must be pusht or aloha' >&2; exit 2; }
  [[ "$3" =~ ^(timing|baselines|pairs|pilot)$ ]] || { echo 'invalid phase' >&2; exit 2; }
  for number in "$4" "$5" "$6"; do
    [[ "$number" =~ ^[0-9]+$ && ${#number} -le 18 ]] || { echo 'shard values must be unsigned integers of at most 18 digits' >&2; exit 2; }
  done
  first=$((10#$4)); count=$((10#$5)); shards=$((10#$6))
  (( count > 0 && shards > 0 && first + count <= shards )) || { echo 'n_vms and n_shards must be positive, with first_shard + n_vms <= n_shards' >&2; exit 2; }
  : "${GCP_ZONE:?Set GCP_ZONE to the VM zone}"
  machine=${7:-c2d-highcpu-32}
  [[ "$machine" =~ ^[a-z0-9]+(-[a-z0-9]+)+$ ]] || { echo 'invalid machine type' >&2; exit 2; }
  repo=$(git remote get-url origin); commit=$(git rev-parse HEAD)
  [[ "$repo" =~ ^https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'origin must be a public HTTPS GitHub repository URL without credentials' >&2; exit 2; }
  hf_secret=false
  enabled=$(gcloud services list --enabled --project="$GCP_PROJECT" --filter='config.name=secretmanager.googleapis.com' --format='value(config.name)')
  if [[ -n "$enabled" ]]; then
    secret=$(gcloud secrets list --project="$GCP_PROJECT" --filter='name~"/hf-token$"' --format='value(name)')
    [[ -z "$secret" ]] || hf_secret=true
  fi
  startup=$(mktemp); trap 'rm -f "$startup"' EXIT
  cat > "$startup" <<'STARTUP'
#!/usr/bin/env bash
set -euo pipefail
trap 'result=$?; echo "Brier startup exit status: $result"; poweroff; exit "$result"' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.11 python3.11-venv libosmesa6-dev libgl1 libglib2.0-0 ffmpeg git curl ca-certificates
metadata=http://metadata.google.internal/computeMetadata/v1/instance/attributes
for key in env phase shard n_shards commit repo GCP_PROJECT BRIER_DATA hf_secret; do
  value=$(curl -fsS -H 'Metadata-Flavor: Google' "$metadata/$key")
  printf -v "$key" '%s' "$value"
done
export GCP_PROJECT BRIER_DATA MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git clone "$repo" /opt/brier
cd /opt/brier
git checkout --detach "$commit"
python3.11 -m venv .venv
.venv/bin/python -m pip install uv
cpu_pins=$(.venv/bin/python -c 'import tomllib; d=tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]; p=[v for v in d if v.split("==")[0] in ("torch","torchvision","torchcodec")]; assert len(p)==3 and all("==" in v for v in p), "CPU torch dependencies must be pinned"; print(" ".join(p))')
read -r -a cpu_deps <<< "$cpu_pins"
.venv/bin/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "${cpu_deps[@]}"
.venv/bin/uv pip install --python .venv/bin/python -e .
if [[ "$hf_secret" == true ]]; then
  HF_TOKEN=$(.venv/bin/python - <<'PY'
import base64, os, google.auth
from google.auth.transport.requests import AuthorizedSession
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
response = AuthorizedSession(credentials).get(f"https://secretmanager.googleapis.com/v1/projects/{os.environ['GCP_PROJECT']}/secrets/hf-token/versions/latest:access", timeout=30)
response.raise_for_status()
print(base64.b64decode(response.json()["payload"]["data"]).decode(), end="")
PY
)
  export HF_TOKEN
fi
if [[ "$phase" == timing ]]; then
  .venv/bin/python -m brier_sim.farm timing --env "$env"
else
  result=0
  .venv/bin/python -m brier_sim.farm run --env "$env" --phase "$phase" --shard "$shard" --n-shards "$n_shards" --procs "$(nproc)" --max-hours 20 || result=$?
  [[ "$result" == 0 || "$result" == 3 ]] || exit "$result"
fi
STARTUP
  for ((shard=first; shard<first+count; shard++)); do
    gcloud compute instances create "brier-roll-$2-$shard" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
      --provisioning-model=SPOT --instance-termination-action=DELETE --machine-type="$machine" \
      --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=50GB \
      --service-account="brier-rollouts@$GCP_PROJECT.iam.gserviceaccount.com" --scopes=cloud-platform \
      --metadata="env=$2,phase=$3,shard=$shard,n_shards=$shards,commit=$commit,repo=$repo,GCP_PROJECT=$GCP_PROJECT,BRIER_DATA=$BRIER_DATA,hf_secret=$hf_secret" \
      --metadata-from-file="startup-script=$startup"
  done
  ;;
status)
  gcloud compute instances list --project="$GCP_PROJECT" --filter='name~^brier-roll-' --format='table(name,zone.basename(),status)'
  .venv/bin/python - <<'PY'
import os
from collections import Counter
from urllib.parse import urlsplit
from google.cloud import storage
root = urlsplit(os.environ["BRIER_DATA"])
prefix = root.path.lstrip("/").rstrip("/")
prefix = (prefix + "/" if prefix else "") + "episodes/"
counts = Counter()
for blob in storage.Client().list_blobs(root.netloc, prefix=prefix, retry=None):
    parts = blob.name.removeprefix(prefix).split("/")
    if len(parts) == 4 and parts[-1].endswith(".parquet"):
        counts[parts[0].removeprefix("env="), parts[1].removeprefix("tier=")] += 1
for env in ("pusht", "aloha"):
    print(f"{env}: baselines={counts[env, 'baseline']} pairs={counts[env, 'cheap'] + counts[env, 'expensive']} pilot={counts[env, 'pilot']}")
PY
  ;;
down)
  instances=$(gcloud compute instances list --project="$GCP_PROJECT" --filter='name~^brier-roll-' --format='csv[no-heading](name,zone.basename())')
  while IFS=, read -r name zone; do
    [[ -z "$name" ]] || gcloud compute instances delete "$name" --project="$GCP_PROJECT" --zone="$zone" --quiet
  done <<< "$instances"
  ;;
esac
