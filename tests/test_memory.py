"""Tests for the short-term and long-term memory store."""

from __future__ import annotations

from cortex.memory import LongTermMemory, Memory, ShortTermMemory


def test_short_term_buffer_window():
    stm = ShortTermMemory(max_items=3)
    for i in range(5):
        stm.add("user", f"msg-{i}")
    assert len(stm) == 3
    recent = stm.recent()
    assert recent[0][1] == "msg-2"
    assert recent[-1][1] == "msg-4"


def test_short_term_recent_n():
    stm = ShortTermMemory()
    stm.add("user", "a")
    stm.add("assistant", "b")
    assert stm.recent(1) == [("assistant", "b")]


def test_long_term_add_and_count():
    ltm = LongTermMemory(db_path=":memory:")
    ltm.add("The capital of France is Paris.")
    ltm.add("Python is a programming language.")
    assert ltm.count() == 2


def test_long_term_keyword_recall():
    ltm = LongTermMemory(db_path=":memory:")
    ltm.add("The capital of France is Paris.")
    ltm.add("The Eiffel Tower is in Paris.")
    ltm.add("Bananas are yellow fruit.")

    hits = ltm.recall("Where is Paris located", top_k=2)
    assert hits, "expected at least one keyword hit"
    assert all("paris" in h.content.lower() for h in hits)


def test_long_term_recall_empty_query():
    ltm = LongTermMemory(db_path=":memory:")
    ltm.add("something")
    assert ltm.recall("", top_k=3) == []


def test_long_term_clear():
    ltm = LongTermMemory(db_path=":memory:")
    ltm.add("x")
    ltm.clear()
    assert ltm.count() == 0


def test_composed_memory(tmp_path):
    mem = Memory.create(db_path=str(tmp_path / "m.sqlite"))
    mem.observe("user", "hello")
    rid = mem.remember("The agent uses a ReAct loop.")
    assert isinstance(rid, int)
    hits = mem.recall("ReAct loop")
    assert hits and "ReAct" in hits[0].content


def test_vector_recall_falls_back_to_keyword(tmp_path):
    # use_vectors=True but sentence-transformers is not installed in CI, so the
    # store should silently fall back to keyword recall and still work.
    ltm = LongTermMemory(db_path=str(tmp_path / "v.sqlite"), use_vectors=True)
    assert ltm.use_vectors is False
    ltm.add("Quantum computing uses qubits.")
    hits = ltm.recall("qubits", top_k=1)
    assert hits and "qubit" in hits[0].content.lower()
