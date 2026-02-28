import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = os.getenv('MODEL_NAME', 'Alibaba-NLP/gte-multilingual-base')
CACHE_DIR = os.getenv('HF_HOME', '/hf_cache')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '8192'))

# Sanity-чек офлайн-кэша (только в режиме TRANSFORMERS_OFFLINE=1)
if os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1':
    for must in [
        f'{CACHE_DIR}/models--Alibaba-NLP--gte-multilingual-base',
        f'{CACHE_DIR}/modules/transformers_modules/Alibaba-NLP--new-impl',
    ]:
        if not os.path.exists(must):
            raise RuntimeError(f'Missing HF offline cache dir: {must}')

app = FastAPI(title='GTE Sparse Encoder', version='1.0')


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    token_weights: List[Dict[str, float]]


class _GTEEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=CACHE_DIR,
        ).to(self.device).eval()

        self.unused_tokens = {
            t for t in [
                getattr(self.tokenizer, 'cls_token_id', None),
                getattr(self.tokenizer, 'eos_token_id', None),
                getattr(self.tokenizer, 'pad_token_id', None),
                getattr(self.tokenizer, 'unk_token_id', None),
            ] if t is not None
        }

    @torch.no_grad()
    def encode(self, text: str) -> Dict[str, float]:
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, return_dict=True)

        token_weights = torch.relu(out.logits).squeeze(-1)
        tw = token_weights[0].cpu().numpy().tolist()
        ids = enc['input_ids'][0].cpu().numpy().tolist()

        result: Dict[str, float] = defaultdict(float)
        for w, idx in zip(tw, ids):
            if idx in self.unused_tokens or w <= 0:
                continue
            tok = self.tokenizer.decode([int(idx)], skip_special_tokens=True).strip()
            if w > result[tok]:
                result[tok] = float(w)
        return dict(result)


print('Loading GTE model...')
_encoder = _GTEEncoder()
print('GTE model loaded')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_NAME}


@app.post('/encode', response_model=EncodeResponse)
def encode(req: EncodeRequest):
    token_weights = _encoder.encode(req.text)
    return EncodeResponse(token_weights=[token_weights])
