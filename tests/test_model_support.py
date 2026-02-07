"""Tests for verifying support and basic functionality of various LLM models."""

from openai import OpenAI

from fake_llm_server import FakeLLMServer


def run_deterministic_test(model_name: str, prompt: str, expected_output: str) -> None:
    """Runs a deterministic test for a specific model with a prompt and expected output.

    Args:
        model_name: The name of the model to test.
        prompt: The input prompt for the model.
        expected_output: The expected string output from the model.
    """
    print(f"\nTesting {model_name}...")
    server = FakeLLMServer(model_name=model_name)
    try:
        client = OpenAI(**server.openai_client_args())
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,  # Increased slightly to capture full outputs
        )
        content = response.choices[0].message.content
        print(f"Output: {content!r}")
        # We strip whitespace from the ends for robust comparison
        assert content and content.strip() == expected_output.strip()
    finally:
        server.shutdown()


def test_gemma_3_270m_simple() -> None:
    """Verifies gemma-3-270m with a simple factual question."""
    # 270M gave "1" for "1+1?". Let's try a completion that is very standard.
    # "The color of the sky is" -> "blue"
    # But it's an instruct model. "What color is the sky? Answer with one word."
    run_deterministic_test(
        "gemma-3-270m",
        "What color is the sky? Answer with just the word.",
        "Blue",
    )


def test_gemma_3_1b_fact() -> None:
    """Verifies gemma-3-1b with a simple factual question."""
    run_deterministic_test(
        "gemma-3-1b",
        "What is the capital of France? Answer with just the city name.",
        "Paris",
    )


def test_qwen_2_5_coder_1_5b_code() -> None:
    """Verifies qwen-2.5-coder-1.5b with a code generation prompt."""
    # It's a coder model, ask for a python function signature
    prompt = (
        "Write a Python function signature for adding two numbers named 'add'. "
        "Do not write the body."
    )
    run_deterministic_test(
        "qwen-2.5-coder-1.5b",
        prompt,
        "```python\ndef add(a, b):\n    pass\n```",
    )


def test_qwen_2_5_coder_3b_code() -> None:
    """Verifies qwen-2.5-coder-3b with a code generation prompt."""
    expected = (
        'Certainly! Here is the Python code to print "Hello World":\n\n'
        '```python\nprint("Hello World")\n```'
    )
    run_deterministic_test(
        "qwen-2.5-coder-3b",
        "Print 'Hello World' in Python. Code only.",
        expected,
    )


def test_llama_3_2_3b_instruct_chat() -> None:
    """Verifies llama-3.2-3b-instruct with a simple greeting."""
    run_deterministic_test(
        "llama-3.2-3b-instruct",
        "Say 'Hello, World!'",
        "Hello, World!",
    )


def test_smollm3_reasoning() -> None:
    """Verifies smollm3 with a reasoning prompt."""
    # SmolLM3 3B has reasoning capabilities <think>
    run_deterministic_test(
        "smollm3",
        "Which is larger, 5 or 10? Answer with just the number.",
        "<think>\n\n</think>\n10 is larger than 5.",
    )
