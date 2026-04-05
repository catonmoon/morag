#!/usr/bin/env python3
"""Native FRIDA embedder server (MPS/CUDA/CPU).

Аналог Docker-версии app.py, но запускается нативно на хосте
для использования MPS (Apple Silicon) или CUDA GPU.

Usage:
    python services/embedder_frida/app_native.py                  # auto device
    python services/embedder_frida/app_native.py --device mps     # force MPS
    python services/embedder_frida/app_native.py --port 8082
    python services/embedder_frida/app_native.py --model ai-forever/FRIDA
"""
from __future__ import annotations

import argparse
from base64 import b64encode
from typing import Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, list[str]]
    encoding_format: Optional[str] = 'float'
    user: Optional[str] = None


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def create_app(model_name: str, device: str | None = None) -> FastAPI:
    device = device or _detect_device()
    app = FastAPI(title='FRIDA Embeddings (native)', version='1.0')

    print(f'Loading FRIDA model: {model_name} (device={device})')
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f'FRIDA model loaded: dim={dim}, device={device}')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': model_name, 'device': device, 'dim': dim}

    @app.post('/v1/embeddings')
    def create_embeddings(req: EmbeddingsRequest):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts:
            raise HTTPException(400, 'input is empty')

        vecs = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)

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
            'model': model_name,
            'usage': {'prompt_tokens': 0, 'total_tokens': 0},
        }

    return app


def main():
    parser = argparse.ArgumentParser(description='Native FRIDA embedder server')
    parser.add_argument('--model', default='ai-forever/FRIDA', help='Model name or path')
    parser.add_argument('--device', default=None, help='Device: mps, cuda, cpu (auto if omitted)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    app = create_app(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == '__main__':
    main()
