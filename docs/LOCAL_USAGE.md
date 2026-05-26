# Local Usage

The released model checkpoints are hosted on Hugging Face, not in this GitHub repository.

## Clone a model repository

Example:

    git clone https://huggingface.co/Ethosoft/nedoqwen_0.8b_pretrained_sft
    cd nedoqwen_0.8b_pretrained_sft

Make sure Git LFS is installed:

    sudo apt install git-lfs
    git lfs install
    git lfs pull

The `checkpoint.pt` file should be around 1.6GB. If it is around 100-200 bytes, it is a Git LFS pointer file and the model will not load.

## Install dependencies

    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch numpy

## Run sampling

    PYTHONPATH=. python3 scripts/30_sample_qwen_style.py \
      --ckpt checkpoint.pt \
      --vocab tokenizer/vocab_65536.jsonl \
      --prompt "Kullanıcı talimatı:
    Fransa'nın başkenti nedir?

    Asistan cevabı:
    " \
      --temperature 0.5 \
      --top-p 0.85 \
      --max-new-tokens 30
