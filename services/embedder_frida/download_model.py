import os

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='ai-forever/FRIDA',
    local_dir='/opt/hf/models/FRIDA',
    local_dir_use_symlinks=False,
)
print('FRIDA model downloaded to /opt/hf/models/FRIDA')
