"""Автозагрузка ссылок из вопроса: идентификатор в кавычках может содержать пробелы.

У локального источника `doc_id` — это путь файла, а каталоги называют по-человечески
(«Python 2023»). Голый регэксп обрывал такой id на пробеле, автозагрузка грузила несуществующий
документ, и агент отвечал из ничего — молча. Здесь проверяется, что кавычечная форма берётся
целиком, а её обрубок до пробела не становится вторым «документом».
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402


def _pipeline() -> Pipeline:
    p = object.__new__(Pipeline)
    # источник «local:demo» известен конфигу; остальное — чужое
    p._resolve_ref = lambda ref: ('local', 'demo') if ref.startswith('local:demo:') else None
    return p


def test_quoted_id_with_spaces_is_taken_whole():
    p = _pipeline()
    text = ('Вопрос — про запись. Идентификатор: "local:demo:Курсы/Python 2023/2024-08-01-x/record.md" — '
            'передавай дословно: get_doc("local:demo:Курсы/Python 2023/2024-08-01-x/record.md", query).')
    assert p._extract_refs(text) == ['local:demo:Курсы/Python 2023/2024-08-01-x/record.md']


def test_unquoted_id_without_spaces_still_works():
    p = _pipeline()
    text = 'Смотри local:demo:Доклады/2026/2026-03-12-kafka/record.md, там всё.'
    assert p._extract_refs(text) == ['local:demo:Доклады/2026/2026-03-12-kafka/record.md']


def test_guillemets_count_as_quotes_and_foreign_sources_are_ignored():
    p = _pipeline()
    text = 'Запись «local:demo:Курсы/Основы 2023/lesson-1/record.md» и чужое "local:other:a b/c.md".'
    assert p._extract_refs(text) == ['local:demo:Курсы/Основы 2023/lesson-1/record.md']
