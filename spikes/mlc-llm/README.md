# Install on runpod

1. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Install git-lfs (`apt-get update; apt-get install git-lfs`)
3. Create new environment with python 3.11 (`uv venv -p 3.11; source .venv/bin/activate`)
4. `python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu123 mlc-ai-nightly-cu123`