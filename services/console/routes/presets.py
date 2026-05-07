"""GET /api/presets и POST /api/presets/apply — wizard-пресеты.

apply route добавляет item в `llms[]` или `sources[]` в config.local.yml,
с replace-by-(name) для idempotency. Не deep-merge'ит top-level (lists в
overlay перезаписываются целиком — пишем новый полный список).
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from services.console.config_io import patch_local, read_layered, read_local, validate_merged
from services.console.presets import (
    EMBEDDER_PRESETS,
    LLM_PRESETS,
    SOURCE_PRESETS,
    apply_preset,
    serialize_preset,
    unique_name,
)

router = APIRouter()

# Secret-поля, которые при Edit сохраняются из existing item если в форме пусто.
# Иначе UI вынудил бы юзера заново вводить ключи при любой правке (т.к. /api/config
# отдаёт их замаскированными как '***').
SECRET_FIELDS = ('api_key', 'password', 'api_token')


def _preserve_secrets(new_item: dict, existing: dict | None) -> dict:
    """Подкопировать secret-поля из existing item если в new_item их нет."""
    if not existing:
        return new_item
    for sk in SECRET_FIELDS:
        if not new_item.get(sk) and existing.get(sk):
            new_item[sk] = existing[sk]
    return new_item


class ApplyPresetRequest(BaseModel):
    target: Literal['llm', 'source', 'embedder']
    preset_id: str
    form: dict[str, Any]


class DeleteItemRequest(BaseModel):
    target: Literal['llm', 'source']           # embedder не удаляется (он один)
    name: str
    kind: str | None = None                     # обязателен для source (kind+name = ключ)


@router.get('')
async def list_presets() -> dict[str, list[dict[str, Any]]]:
    return {
        'llm': [serialize_preset(p) for p in LLM_PRESETS],
        'source': [serialize_preset(p) for p in SOURCE_PRESETS],
        'embedder': [serialize_preset(p) for p in EMBEDDER_PRESETS],
    }


@router.post('/apply')
async def apply(req: ApplyPresetRequest, request: Request) -> dict[str, Any]:
    """Добавить новый item в llms[] / sources[] или заменить indexing.dense_embedder.

    - llm/source: replace-by-name (для llms) или (kind, name) (для sources). Иначе append.
    - embedder: replace целиком (он один в схеме, не пул).
    """
    cfg_path = request.app.state.config_path
    try:
        item = apply_preset(req.target, req.preset_id, req.form)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Bad form data: {e}') from e

    current_local = read_local(cfg_path)

    if req.target == 'embedder':
        # indexing.dense_embedder — один объект, не список. Replace целиком.
        # Preserve secret из existing если в форме пусто.
        merged_view = read_layered(cfg_path)
        existing_emb = (merged_view.get('indexing') or {}).get('dense_embedder')
        item = _preserve_secrets(item, existing_emb)

        indexing = dict(current_local.get('indexing') or {})
        indexing['dense_embedder'] = item
        candidate_local = {**current_local, 'indexing': indexing}

        try:
            validate_merged(cfg_path, candidate_local)
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=e.errors(include_url=False, include_input=False, include_context=False),
            ) from e

        patch_local(cfg_path, {'indexing': {'dense_embedder': item}})
        return {'ok': True, 'added': item}

    # llm / source: list-append с replace-by-key. Читаем merged-вид (primary + local)
    # для апсерта — потому что lists в deep_merge перезаписываются целиком: если запишем
    # в local только новый item, primary-shipped llms/sources исчезнут из merged-вида.
    merged_view = read_layered(cfg_path)
    list_field = 'llms' if req.target == 'llm' else 'sources'
    current_list = list(merged_view.get(list_field, []))

    # Editing-mode: form может прислать имя. Если присланное имя совпадает с
    # existing — это edit (preserve secrets, replace by name). Иначе — add,
    # auto-generated имя через collision-detect.
    incoming_name = item.get('name', '')
    if req.target == 'llm':
        existing = next(
            (it for it in current_list if it.get('name') == incoming_name), None,
        )
    else:  # source
        existing = next(
            (it for it in current_list
             if it.get('kind') == item.get('kind') and it.get('name') == incoming_name),
            None,
        )

    # Add (no existing match): collision-detect — если имя занято, добавим -2/-3.
    # local source — singleton, не дублируется (имя 'doc' жёстко).
    if existing is None and req.target != 'embedder':
        if req.target == 'llm':
            taken = {it.get('name') for it in current_list}
        else:
            taken = {
                it.get('name') for it in current_list
                if it.get('kind') == item.get('kind')
            }
        item['name'] = unique_name(incoming_name, taken)

    item = _preserve_secrets(item, existing)

    if req.target == 'llm':
        merged_list = _upsert_by_name(current_list, item, key='name')
    else:  # source
        merged_list = _upsert_by_name(current_list, item, key=('kind', 'name'))

    candidate_local = {**current_local, list_field: merged_list}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    patch_local(cfg_path, {list_field: merged_list})
    return {'ok': True, 'added': item, 'list_size': len(merged_list)}


@router.post('/embedder/probe-dim')
async def probe_embedder_dim(req: ApplyPresetRequest, request: Request) -> dict[str, Any]:
    """Узнать размерность вектора у embedder'а — для UI-кнопки «выяснить» рядом с полем dim.

    Принимает форму как для apply (target='embedder'), но dim необязателен —
    эмбеддер вызывается с любым dim (он не валидирует выдачу), возвращается
    реальная длина возвращённого вектора.
    """
    try:
        # Подменим dim на dummy чтобы пройти int(form['dim']) в build функции
        form = {**req.form, 'dim': req.form.get('dim') or '1'}
        item = apply_preset(req.target, req.preset_id, form)
    except Exception as e:
        return {'ok': False, 'error': f'Bad form data: {e}'}

    from morag.indexing.embedder import HttpEmbedder
    try:
        embedder = HttpEmbedder(
            item['base_url'], item['model'], 1,
            api_key=item.get('api_key') or 'ollama',
            timeout=15, max_retries=0,
        )
        vec = await embedder.embed_batch(['ping'])
        return {'ok': True, 'dim': len(vec[0])}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


@router.post('/test')
async def test_preset(req: ApplyPresetRequest, request: Request) -> dict[str, Any]:
    """Проверить подключение по полям формы — БЕЗ записи в config.local.yml.

    Цель: дать юзеру кнопку «Проверить» в форме add/edit, чтобы убедиться
    что введённые base_url/model/api_key работают, до сохранения.
    Поддерживает target='llm' и target='embedder' (для source — нет смысла).
    """
    try:
        item = apply_preset(req.target, req.preset_id, req.form)
    except Exception as e:
        return {'ok': False, 'detail': f'Bad form data: {e}'}

    if req.target == 'llm':
        return await _ping_llm(item)
    if req.target == 'embedder':
        return await _ping_embedder(item)
    return {'ok': False, 'detail': f"target={req.target!r} не поддерживается для проверки"}


async def _ping_llm(item: dict) -> dict[str, Any]:
    """Отправляет 'ping' с max_tokens=1 — провайдер должен ответить хоть чем-то."""
    import time
    from morag.llm.client import GenerationParams, LLMClient
    client = LLMClient(
        base_url=item['base_url'],
        model=item['model'],
        api_key=item['api_key'],
        timeout=15,
        max_retries=0,
        enable_thinking=False,
    )
    try:
        t0 = time.monotonic()
        answer = await client.complete(
            messages=[{'role': 'user', 'content': 'ping'}],
            params=GenerationParams(temperature=0),
            max_tokens=1,
        )
        ms = int((time.monotonic() - t0) * 1000)
        # Сократим ответ до 40 символов чтобы не вылезал длинный текст
        snippet = answer.strip().replace('\n', ' ')
        if len(snippet) > 40:
            snippet = snippet[:40] + '…'
        return {
            'ok': True,
            'detail': f'модель ответила за {ms} мс — «ping» → «{snippet}»',
        }
    except Exception as e:
        return {'ok': False, 'detail': f'{type(e).__name__}: {e}'}


async def _ping_embedder(item: dict) -> dict[str, Any]:
    """Эмбеддит 'ping' и проверяет что возвращается вектор ожидаемой размерности."""
    import time
    from morag.indexing.embedder import HttpEmbedder
    try:
        embedder = HttpEmbedder(
            item['base_url'], item['model'], item.get('dim'),
            api_key=item.get('api_key') or 'ollama',
            timeout=15, max_retries=0,
        )
        t0 = time.monotonic()
        vec = await embedder.embed_batch(['ping'])
        ms = int((time.monotonic() - t0) * 1000)
        return {
            'ok': True,
            'detail': f'эмбеддер вернул вектор размерности {len(vec[0])} за {ms} мс',
        }
    except Exception as e:
        return {'ok': False, 'detail': f'{type(e).__name__}: {e}'}


@router.post('/delete')
async def delete_item(req: DeleteItemRequest, request: Request) -> dict[str, Any]:
    """Удалить item из llms[] или sources[] в config.local.yml.

    Защита: если после удаления indexing.llm/.vision ссылается на удалённый
    item — Pydantic-валидация выбросит 400.
    """
    cfg_path = request.app.state.config_path
    list_field = 'llms' if req.target == 'llm' else 'sources'

    if req.target == 'source' and not req.kind:
        raise HTTPException(status_code=400, detail='kind required for source delete')

    current_local = read_local(cfg_path)
    merged_view = read_layered(cfg_path)
    current_list = list(merged_view.get(list_field, []))

    def matches(item: dict) -> bool:
        if req.target == 'llm':
            return item.get('name') == req.name
        return item.get('kind') == req.kind and item.get('name') == req.name

    new_list = [it for it in current_list if not matches(it)]
    if len(new_list) == len(current_list):
        raise HTTPException(status_code=404, detail=f'{req.target} {req.name!r} not found')

    candidate_local = {**current_local, list_field: new_list}
    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    patch_local(cfg_path, {list_field: new_list})
    return {'ok': True, 'list_size': len(new_list)}


def _upsert_by_name(items: list[dict], new_item: dict, key) -> list[dict]:
    """Replace item с тем же `key` или append если не найден.

    `key` может быть строкой (для llms.name) или tuple (для sources.kind+name).
    """
    if isinstance(key, str):
        def get_key(x): return x.get(key)
    else:
        def get_key(x): return tuple(x.get(k) for k in key)

    new_key = get_key(new_item)
    result = []
    replaced = False
    for existing in items:
        if get_key(existing) == new_key:
            result.append(new_item)
            replaced = True
        else:
            result.append(existing)
    if not replaced:
        result.append(new_item)
    return result


# ---------------------------------------------------------------------------
# POST /api/presets/roles — set indexing.llm + indexing.vision
# ---------------------------------------------------------------------------

class SetRolesRequest(BaseModel):
    llm: str            # имя из llms-pool для default-роли (text)
    vision: str         # имя из llms-pool для vision-роли


@router.post('/roles')
async def set_roles(req: SetRolesRequest, request: Request) -> dict[str, Any]:
    """Установить indexing.llm (default text) и indexing.vision."""
    cfg_path = request.app.state.config_path
    current_local = read_local(cfg_path)

    indexing = dict(current_local.get('indexing') or {})
    indexing['llm'] = req.llm
    indexing['vision'] = req.vision
    candidate_local = {**current_local, 'indexing': indexing}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    patch_local(cfg_path, {'indexing': {'llm': req.llm, 'vision': req.vision}})
    return {'ok': True, 'llm': req.llm, 'vision': req.vision}
