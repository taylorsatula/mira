from __future__ import annotations

from types import SimpleNamespace

from lt_memory.processing.execution_strategy import DirectExecutionStrategy


class FakeExtractionEngine:
    def build_extraction_payload(self, chunk):
        return SimpleNamespace(
            system_prompt="system",
            user_prompt="extract this",
            short_to_uuid={},
            memory_context={},
        )


class FakeMemoryProcessor:
    def process_extraction_response(self, **kwargs):
        return SimpleNamespace(memories=[])


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return "{}"

    def extract_text_content(self, response):
        return response


def test_extraction_is_direct_and_uses_batch_model_config() -> None:
    llm = RecordingLLM()
    strategy = DirectExecutionStrategy(
        extraction_engine=FakeExtractionEngine(),
        memory_processor=FakeMemoryProcessor(),
        vector_ops=object(),
        db=object(),
        llm_provider=llm,
        linking_service=object(),
    )
    execution_id = strategy.execute_extraction(
        "11111111-1111-1111-1111-111111111111",
        [SimpleNamespace(chunk_index=0, segment_id=None)],
    )
    assert execution_id.startswith("direct_")
    assert len(llm.calls) == 1
    assert llm.calls[0]["model_config"] == "batch"
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "extract this"}]
    assert "batch_id" not in llm.calls[0]
    assert "custom_id" not in llm.calls[0]
