"""Тесты scripts/migrate_config.py — pure-function migrate()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from migrate_config import migrate


class TestSchemaVersion:
    def test_adds_schema_version_1(self):
        new = migrate({'sources': {}, 'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'}})
        assert new['schema_version'] == 1


class TestSources:
    def test_local_documents(self):
        new = migrate({
            'sources': {'local_documents': {'path': 'data/'}},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
        })
        assert new['sources'] == [{'kind': 'local', 'name': 'docs', 'path': 'data/'}]

    def test_confluence(self):
        new = migrate({
            'sources': {'confluence': {
                'url': 'https://corp/', 'username': 'u', 'api_token': 't',
                'spaces': ['DOCS'],
            }},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
        })
        assert len(new['sources']) == 1
        s = new['sources'][0]
        assert s['kind'] == 'confluence'
        assert s['name'] == 'main'
        assert s['url'] == 'https://corp/'
        assert s['spaces'] == ['DOCS']

    def test_jira_drops_api_token(self):
        new = migrate({
            'sources': {'jira': {
                'url': 'https://j/', 'username': 'u', 'api_token': 'cloud-token',
            }},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
        })
        s = new['sources'][0]
        assert s['kind'] == 'jira'
        # api_token удаляется (на новой схеме только on-prem с password)
        assert 'api_token' not in s

    def test_jira_keeps_password(self):
        new = migrate({
            'sources': {'jira': {
                'url': 'https://j/', 'username': 'u', 'password': 'p',
            }},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
        })
        s = new['sources'][0]
        assert s['password'] == 'p'

    def test_all_three_sources(self):
        new = migrate({
            'sources': {
                'local_documents': {'path': 'a'},
                'confluence': {'url': 'b', 'username': 'u', 'password': 'p'},
                'jira': {'url': 'c', 'username': 'u', 'password': 'p'},
            },
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
        })
        kinds = [s['kind'] for s in new['sources']]
        assert kinds == ['local', 'confluence', 'jira']


class TestLLMs:
    def test_text_only_llm(self):
        new = migrate({
            'sources': {'local_documents': {'path': 'a'}},
            'llm': {
                'base_url': 'http://x',
                'model': 'qwen3:4b',
                'api_key': 'ollama',
            },
        })
        assert len(new['llms']) == 1
        m = new['llms'][0]
        assert m['name'] == 'main'
        assert m['model'] == 'qwen3:4b'
        assert m['capabilities'] == ['text']

    def test_vision_added_with_capabilities(self):
        new = migrate({
            'sources': {'local_documents': {'path': 'a'}},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
            'llm_vision': {'base_url': 'y', 'model': 'qwen2.5-vl', 'api_key': 'k'},
        })
        names = {llm['name'] for llm in new['llms']}
        assert names == {'main', 'vision'}
        vision = next(llm for llm in new['llms'] if llm['name'] == 'vision')
        assert vision['capabilities'] == ['text', 'vision']

    def test_indexing_role_mapping_auto_wired(self):
        new = migrate({
            'sources': {'local_documents': {'path': 'a'}},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
            'llm_vision': {'base_url': 'y', 'model': 'v', 'api_key': 'k'},
            'indexing': {'concurrency': 2},
        })
        assert new['indexing']['llm'] == 'main'
        assert new['indexing']['vision'] == 'vision'
        # Прочее indexing-поле сохранилось
        assert new['indexing']['concurrency'] == 2


class TestPassthrough:
    def test_qdrant_section_passes_through(self):
        new = migrate({
            'sources': {'local_documents': {'path': 'a'}},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
            'qdrant': {'host': 'qdrant', 'port': 6333},
        })
        assert new['qdrant'] == {'host': 'qdrant', 'port': 6333}

    def test_pdf_section_passes_through(self):
        pdf = {'mode': 'vision', 'dpi': 144}
        new = migrate({
            'sources': {'local_documents': {'path': 'a'}},
            'llm': {'base_url': 'x', 'model': 'm', 'api_key': 'k'},
            'pdf': pdf,
        })
        assert new['pdf'] == pdf


class TestEndToEnd:
    def test_migrated_config_validates_under_new_schema(self):
        """Главный тест — мигрированный config действительно проходит Pydantic."""
        from morag.config import Config

        old = {
            'sources': {
                'local_documents': {'path': 'data/'},
                'confluence': {
                    'url': 'https://corp/', 'username': 'u', 'api_token': 't',
                },
            },
            'llm': {
                'base_url': 'http://localhost:11434/v1',
                'model': 'qwen3:4b',
                'api_key': 'ollama',
            },
            'llm_vision': {
                'base_url': 'http://localhost:11434/v1',
                'model': 'qwen2.5-vl',
                'api_key': 'ollama',
            },
            'qdrant': {'host': 'qdrant', 'port': 6333},
            'indexing': {
                'concurrency': 2,
                'dense_embedder': {
                    'model': 'qwen3-embedding:4b',
                    'base_url': 'http://x',
                    'dim': 2560,
                },
            },
        }
        new = migrate(old)

        # Должен пройти валидацию Pydantic новой схемы
        cfg = Config.model_validate(new)
        assert cfg.schema_version == 1
        assert len(cfg.sources) == 2
        assert len(cfg.llms) == 2
        assert cfg.indexing.llm.default == 'main'
        assert cfg.indexing.vision == 'vision'
