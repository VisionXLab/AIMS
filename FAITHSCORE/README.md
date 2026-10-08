### FAITHSCORE
This evaluation code is adapted from the official [FAITHSCORE repository](https://github.com/bcdnlp/FAITHSCORE). 

We make several modifications to facilitate model downloading and dependency setup, support parallel OpenAI API evaluation, and enable automatic resumption from interrupted evaluations.

- Set Environment
<!-- export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
 export TMPDIR=/root/autodl-tmp/pip_tmp -->
```python
cd AIMS/FAITHSCORE
conda create -n faithscore python=3.10
conda activate faithscore
python -m pip install "pip<24.1"
pip install -r faithscore_working_requirements.txt
# Then install the package
pip install faithscore==0.0.9

# 1. Download the OFA model from ModelScope.
# Since the FAITHSCORE environment uses relatively old versions of ModelScope and `datasets`,
# we recommend using a separate environment with the latest ModelScope to download the OFA model.
# https://www.modelscope.cn/models/iic/ofa_visual-question-answering_pretrain_large_en/

# 2. Set the NLTK data path in `FAITHSCORE/src/faithscore/framework_in_package.py`.
# The required NLTK 3.8.1 data can be downloaded directly from
# Hugging Face: `sharon11/aims_benchmarks/nltk_3-8-1`.
# After downloading, set `NLTK_DATA` to the local path of `nltk_3-8-1`.
NLTK_DATA = "<path_to_nltk_3-8-1>"
if NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA)
from nltk.stem import WordNetLemmatizer
_lemmatizer = WordNetLemmatizer()

cp FAITHSCORE/src/faithscore/framework_in_package.py $CONDA_PREFIX/lib/python3.10/site-packages/faithscore/framework.py

cp FAITHSCORE/src/faithscore/prompts/prompt_de_atomic.txt $CONDA_PREFIX/lib/python3.10/site-packages/faithscore/prompts/prompt_de_atomic.txt
```
- Test Environment
```python
python - <<'PY'
import torch
import torchaudio
import torchvision
import transformers
import modelscope

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("modelscope:", modelscope.__version__)

from transformers.models.bert.tokenization_bert import BasicTokenizer
print("BasicTokenizer OK")

from faithscore.framework import FaithScore
print("FaithScore import OK")
PY
# wished:
BasicTokenizer OK
FaithScore import OK
```
- Install llava
```python
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout 786aa6a19ea10edc6f574ad2e16276974e9aaa3a
pip install -e . --no-deps
```

- Run FAITHSCORE evaluation:

```bash
bash scripts/run_faith_eval.sh
```

Main arguments:
- `--image_dir`: Path to the image directory of the FAITHSCORE dataset.
- `--answer_path`: Path to the model response JSONL file.
- `--model_path`: Path to the OFA model. It can be the ModelScope model ID `iic/ofa_visual-question-answering_pretrain_large_en` or a local model path.
- `--openai_key`: OpenAI API key. Stage 3 uses `gpt-3.5-turbo` for evaluation.
- `--openai_url`: OpenAI-compatible API endpoint.
- `--vem_type ofa`: Use OFA as the visual entailment model (VEM).
- `--openai_num_workers 10`: Number of concurrent workers for OpenAI API requests.
- `--ofa_batch_size 4`: Batch size for OFA inference.

