from __future__ import annotations

import re


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)


def is_reasoning_model(model_name_or_path: str | None) -> bool:
    if not model_name_or_path:
        return False
    normalized = str(model_name_or_path).casefold()
    return "instruct" not in normalized


def should_use_post_think_response(
    model_name_or_path: str | None = None,
    text: str | None = None,
) -> bool:
    if is_reasoning_model(model_name_or_path):
        return True

    response_text = str(text or "")
    think_close_match = _THINK_CLOSE_PATTERN.search(response_text)
    if think_close_match is None:
        return False

    post_think_text = response_text[think_close_match.end() :].strip()
    if not post_think_text:
        return False

    if _THINK_PATTERN.search(response_text) is not None:
        return False

    return _ANSWER_PATTERN.search(response_text) is None


def has_reasoning_block(text: str, *, use_post_think_response: bool = False) -> bool:
    if use_post_think_response:
        return _THINK_CLOSE_PATTERN.search(text or "") is not None
    return _THINK_PATTERN.search(text or "") is not None


def extract_response_block(text: str, *, use_post_think_response: bool = False) -> str | None:
    response_text = str(text or "")

    if use_post_think_response:
        match = _THINK_CLOSE_PATTERN.search(response_text)
        if match is None:
            return None
        post_think_text = response_text[match.end() :].strip()
        if not post_think_text:
            return None
        answer_match = _ANSWER_PATTERN.search(post_think_text)
        if answer_match is not None:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                return answer_text
        return post_think_text

    match = _ANSWER_PATTERN.search(response_text)
    if match is None:
        return None
    answer_text = match.group(1).strip()
    return answer_text if answer_text else None


def extract_response_text_for_metrics_with_status(
    text: str,
    *,
    use_post_think_response: bool = False,
    fallback_to_response_text: bool = False,
) -> tuple[str, bool]:
    response_block = extract_response_block(text, use_post_think_response=use_post_think_response)
    if response_block is not None:
        return response_block, True

    response_text = str(text or "")
    if fallback_to_response_text:
        fallback_text = response_text.strip()
        return fallback_text, bool(fallback_text)

    return "", False


def extract_response_text_for_metrics(
    text: str,
    *,
    use_post_think_response: bool = False,
    fallback_to_response_text: bool = False,
) -> str:
    evaluation_text, _ = extract_response_text_for_metrics_with_status(
        text,
        use_post_think_response=use_post_think_response,
        fallback_to_response_text=fallback_to_response_text,
    )
    return evaluation_text
