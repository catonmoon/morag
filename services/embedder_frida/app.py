import os
from base64 import b64encode
from typing import List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Полный офлайн на рантайме
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

MODEL_PATH = os.environ.get('MODEL_PATH', '/opt/hf/models/FRIDA')

app = FastAPI(title='FRIDA Embeddings (OpenAI-compatible)', version='1.0')

_local_files_only = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
print('Loading FRIDA model...')
_model = SentenceTransformer(MODEL_PATH, device='cpu', local_files_only=_local_files_only)
print('FRIDA model loaded')


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Optional[str] = 'float'  # 'float' | 'base64'
    user: Optional[str] = None


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_PATH}


@app.post('/v1/embeddings')
def create_embeddings(req: EmbeddingsRequest):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(400, 'input is empty')

    vecs = _model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)

    data = []
    for i, v in enumerate(vecs):
        if req.encoding_format == 'base64':
            payload = b64encode(v.astype(np.float32).tobytes()).decode('ascii')
        else:
            payload = v.tolist()
        data.append({'object': 'embedding', 'index': i, 'embedding': payload})

    return {
        'object': 'list',
        'data': data,
        'model': MODEL_PATH,
        'usage': {'prompt_tokens': 0, 'total_tokens': 0},
    }
