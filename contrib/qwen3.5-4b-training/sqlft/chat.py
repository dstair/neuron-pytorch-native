"""Chat-template prompt/completion for reasoning-trace training and thinking-mode eval.

Qwen3.5 is a hybrid reasoning model: its chat template opens the assistant turn with
`<think>\\n` (thinking on, the default) or `<think>\\n\\n</think>\\n\\n` (enable_thinking=False).
The plain `-- DuckDB SQL generation` format in sqlft.data bypasses the template and so
measures the model with thinking OFF. This module is the other path.

The user message is byte-identical to what the gpt-oss-120b teacher saw in
labs/lab14_distill.py (PREAMBLE + the plain eval prompt), so a trace is trained under the
exact conditioning it was generated with. `labs/lab15_trace_data.py` asserts the match.

Assistant target (thinking on), matching how the template renders a past assistant turn
with reasoning:
    <think>\\n{trace}\\n</think>\\n\\n```sql\\n{sql}\\n```<|im_end|>
The prompt ends at `<think>\\n`, so the supervised completion is
    {trace}\\n</think>\\n\\n```sql\\n{sql}\\n```   (+ eos = <|im_end|>)
"""

from __future__ import annotations

PREAMBLE = (
    "You are an expert data engineer writing SQL for DuckDB (not SQLite). Write ONE DuckDB "
    "SQL query that answers the question over the schema below.\n"
    "- Use only tables and columns that exist in the schema.\n"
    "- Quote identifiers that contain spaces or special characters with double quotes, "
    "never backticks.\n"
    "- The Evidence line, when present, is external knowledge that maps the question's "
    "wording onto columns and values; follow it.\n"
    "- Return exactly the columns the question asks for, nothing extra.\n"
    "Think it through, then give the final query in a single ```sql block and nothing "
    "after it.\n\n"
)

THINK_CLOSE = "</think>"


def user_message(plain_prompt: str) -> str:
    """plain_prompt is sqlft.bird.render_prompt(...) / sqlft.data.render_prompt(row)."""
    return PREAMBLE + plain_prompt


def render_chat_prompt(tokenizer, plain_prompt: str, *, thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message(plain_prompt)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def render_think_completion(trace: str, sql: str) -> str:
    return f"{trace.strip()}\n{THINK_CLOSE}\n\n```sql\n{sql.strip()}\n```"


def split_thinking(generated: str, *, thinking: bool) -> tuple[str, str, bool]:
    """-> (thinking_text, answer_text, think_closed).

    With thinking on, the prompt already ends in `<think>\\n`, so the generation is
    `{thinking}</think>{answer}`. No `</think>` means the think budget ran out: there is no
    answer, and the row is graded as truncated, not as a parse error.
    """
    if not thinking:
        return "", generated, True
    if THINK_CLOSE not in generated:
        return generated, "", False
    think, _, answer = generated.rpartition(THINK_CLOSE)
    return think, answer, True
