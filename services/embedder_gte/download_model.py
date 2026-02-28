import os
import shutil

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from huggingface_hub import snapshot_download

# 1) Модель в HF_HOME/models--.../snapshots/<hash>
m_path = snapshot_download(
    repo_id='Alibaba-NLP/gte-multilingual-base',
    cache_dir='/hf_cache',
    local_dir=None,
    local_dir_use_symlinks=False,
)

# 2) Динамический код в HF_HOME/modules/transformers_modules/Alibaba-NLP--new-impl/<hash>
n_path = snapshot_download(
    repo_id='Alibaba-NLP/new-impl',
    cache_dir='/hf_cache',
    local_dir=None,
    local_dir_use_symlinks=False,
)

modules_root = '/hf_cache/modules/transformers_modules/Alibaba-NLP--new-impl'
target = os.path.join(modules_root, os.path.basename(n_path))
os.makedirs(modules_root, exist_ok=True)
shutil.copytree(n_path, target, dirs_exist_ok=True)

print('MODEL_SNAPSHOT:', m_path)
print('NEW_IMPL_MODULE:', target)
