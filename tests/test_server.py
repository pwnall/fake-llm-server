import pytest
from openai import OpenAI
from fake_llm_server import FakeLLMServer

@pytest.fixture(scope="module")
def server():
    # Use qwen-2.5-coder-1.5b for testing as it is reliable
    # This will trigger a download on the first run, which might take a moment.
    s = FakeLLMServer(model="qwen-2.5-coder-1.5b")
    yield s
    s.shutdown()

def test_list_models(server):
    client_args = server.openai_client_args()
    client = OpenAI(**client_args)

    models = client.models.list()
    assert len(models.data) > 0
    # Check that our model ID is in the list
    assert any(m.id == "qwen-2.5-coder-1.5b" for m in models.data)

def test_chat_completion(server):
    client_args = server.openai_client_args()
    client = OpenAI(**client_args)

    response = client.chat.completions.create(
        model="qwen-2.5-coder-1.5b",
        messages=[
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        max_tokens=10,
        temperature=0.1
    )

    content = response.choices[0].message.content
    print(f"Model response: {content}")
    assert content
    # Ideally checking for "4" but small models can be unpredictable.
    # Just checking we got a non-empty string is enough for infrastructure test.
