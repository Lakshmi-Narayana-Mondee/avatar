apt update
apt-get update
apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv python3-dev libgl-dev ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
source download_weights.sh
python3 scripts/api.py

