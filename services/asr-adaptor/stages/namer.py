"""Авто-наминг: Speaker_N → реальное имя. ИСТОЧНИК ИСТИНЫ — ИНТРО выпуска (ведущие представляют
себя и гостей), реестр (глобальные имена) — fallback.

Коррекция ложных voice-матчей (кейс ep18 Ермаков↔Натёкин): CAM++ может на пороге 0.55 пришить голос
гостя к чужому центроиду → реестр скажет «Натёкин», а интро («да, действительно, Петя Ермаков», и это
говорит сам [Speaker_4]) — «Ермаков». При конфликте побеждает ИНТРО (per-episode override) + флаг.

Защита от свопа ведущих: если LLM по интро присвоил метке имя ДРУГОГО известного (по реестру)
присутствующего спикера — это, скорее, перепутанная атрибуция, а не ложный матч → оставляем реестр.

Консервативно: не назван ни интро, ни реестром → остаётся Speaker_N. LLM-сбой → только реестр/Speaker_N.
"""
from __future__ import annotations

import re

_SYS = (
    'Тебе дано НАЧАЛО подкаста — реплики с метками говорящих в формате `[Speaker_N] текст`. '
    'Ведущие в начале представляются и представляют гостей по имени («меня зовут …», «со мной …», '
    '«сегодня с нами …», «у нас в гостях …»), гость часто подтверждает своё имя сам. '
    'Сопоставь КАЖДОЙ метке `[Speaker_N]`, которую можно опознать, реальное ПОЛНОЕ Имя и Фамилию '
    '(не уменьшительное: «Петя»→«Пётр») в ПРАВИЛЬНОМ написании — черновой ASR искажает фамилии '
    '(«Колотезев»/«Калабизев»→«Колодезев», «Малык»→«Малых»). Если говорящего по имени НЕ назвали — null. '
    'НЕ выдумывай: только явное называние из текста. Атрибутируй метку по тому, в чьей реплике звучит '
    'самоназывание («меня зовут …», «я …»). Верни СТРОГО JSON '
    '{"speakers":[{"label":"Speaker_N","name":"Имя Фамилия"|null}, …]} без иного текста.'
)

_SCHEMA = {
    'type': 'object',
    'properties': {'speakers': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'label': {'type': 'string'}, 'name': {'type': ['string', 'null']}},
        'required': ['label', 'name']}}},
    'required': ['speakers'],
}


def _norm(s: str) -> str:
    return re.sub(r'[^0-9a-zа-яё]', '', (s or '').lower())


def _name_in_intro(name: str, intro_norm: str) -> bool:
    """Имя/фамилия из реестра звучит в интро (устойчиво к ASR-гарблу: сверяем префикс 5 букв каждого
    слова). «Дмитрий Колодезев» ↔ интро «…Колодич» → 'колод' совпал → подтверждено."""
    for part in re.split(r'\s+', name):
        pn = _norm(part)
        if len(pn) >= 4 and pn[:5] in intro_norm:
            return True
    return False


async def name_speakers(turns: list[dict], registry_names: dict, llm,
                        intro_turns: int = 8, max_tokens: int = 400) -> tuple[dict, list]:
    """turns (со `speaker`=Speaker_N) + registry_names {Speaker_N: имя} → ({Speaker_N: итоговое_имя},
    конфликты). Итоговое имя = интро (истина) → реестр (fallback) → Speaker_N (не назван)."""
    present = sorted({t['speaker'] for t in turns}, key=lambda s: int(s.split('_')[1]))
    intro = '\n'.join(f"[{t['speaker']}] {t.get('final', t.get('text', ''))}"
                      for t in turns[:intro_turns])

    intro_map: dict = {}
    try:
        res = await llm.complete_json(
            [{'role': 'system', 'content': _SYS}, {'role': 'user', 'content': intro}],
            schema=_SCHEMA, schema_name='speaker_names', max_tokens=max_tokens)
        for r in (res or {}).get('speakers', []):
            lbl, nm = r.get('label'), r.get('name')
            nm = nm.strip() if isinstance(nm, str) else ''
            # модель иногда отдаёт строку "null"/"none" вместо JSON null — это НЕ имя
            if lbl in present and nm and nm.lower() not in ('null', 'none', 'неизвестно', '-', '—', 'n/a'):
                intro_map[lbl] = nm
    except Exception:
        pass  # LLM-сбой → fallback только на реестр

    intro_norm = _norm(intro)
    # имена известных присутствующих спикеров — для детекта свопа меток
    reg_present_norm = {_norm(registry_names[s]): s for s in present if registry_names.get(s)}

    out, conflicts = {}, []
    for sid in present:
        intro_name = intro_map.get(sid)
        reg_name = registry_names.get(sid)
        conflict = bool(intro_name and reg_name and _norm(intro_name) != _norm(reg_name))

        if reg_name and _name_in_intro(reg_name, intro_norm):
            # имя из реестра ЗВУЧИТ в интро → личность подтверждена → канон реестра (игнор мис-атрибуции LLM)
            out[sid] = reg_name
            if conflict:
                conflicts.append({'speaker': sid, 'intro': intro_name, 'registry': reg_name,
                                  'resolution': 'kept_registry', 'reason': 'registry_corroborated'})
        elif conflict and _norm(intro_name) in reg_present_norm and reg_present_norm[_norm(intro_name)] != sid:
            # интро присвоило метке имя ДРУГОГО присутствующего known-спикера → своп атрибуции, не матч-ошибка
            out[sid] = reg_name
            conflicts.append({'speaker': sid, 'intro': intro_name, 'registry': reg_name,
                              'resolution': 'kept_registry', 'reason': 'label_swap'})
        elif conflict:
            # реестр НЕ подтверждён интро, интро называет иного человека → ложный voice-матч, истина = интро
            out[sid] = intro_name
            conflicts.append({'speaker': sid, 'intro': intro_name, 'registry': reg_name,
                              'resolution': 'used_intro', 'reason': 'false_voice_match'})
        elif intro_name:
            out[sid] = intro_name           # новый гость, реестр не знает
        elif reg_name:
            out[sid] = reg_name             # известный, интро не назвал
        else:
            out[sid] = sid                  # не назван — остаётся Speaker_N
    return out, conflicts
