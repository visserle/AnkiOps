from __future__ import annotations

from ankiops.init import initialize_collection


def test_initialize_collection_scaffolds_llm_configs(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.init.get_collection_dir", lambda: tmp_path)
    monkeypatch.setattr("ankiops.init._setup_git", lambda _collection_dir: None)

    collection_dir = initialize_collection("TestProfile")

    assert collection_dir == tmp_path
    assert (tmp_path / "llm/tasks/grammar.yaml").exists()
    assert (tmp_path / "llm/providers/ollama-local.yaml").exists()
    assert (tmp_path / "llm/providers/openai-default.yaml").exists()
