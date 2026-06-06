"""Text preprocessing for classical machine learning models."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z\s']")
EXTRA_SPACES_PATTERN = re.compile(r"\s+")


def clean_text_for_classical_ml(text: str) -> str:
    """Clean social-media text for TF-IDF based models.

    The function removes URLs, mentions, hashtags signs, non-letter symbols,
    and extra spaces. It does not print or log the original text.
    """
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" usermention ", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = EXTRA_SPACES_PATTERN.sub(" ", text).strip()
    return text


def clean_text_for_transformer(text: str) -> str:
    """Minimal cleaning for transformer tokenizers.

    Transformer models benefit from preserving more context, so this function
    only normalizes URLs and mentions.
    """
    text = str(text)
    text = URL_PATTERN.sub(" [URL] ", text)
    text = MENTION_PATTERN.sub(" [USER] ", text)
    text = EXTRA_SPACES_PATTERN.sub(" ", text).strip()
    return text


def label_name(label: int) -> str:
    """Human-readable label name."""
    return "OFFENSIVE" if int(label) == 1 else "NOT_OFFENSIVE"
