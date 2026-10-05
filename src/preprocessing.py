
import re

# Contractions that carry negation. We expand them so "not" is never lost.
# Order matters: the special cases must come before the general "n't" rule.
CONTRACTIONS = [
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"n't", " not"),      # isn't -> is not, don't -> do not, didn't -> did not
    (r"'re", " are"),
    (r"'ve", " have"),
    (r"'ll", " will"),
    (r"'d", " would"),
    (r"'m", " am"),
    (r"'s", ""),           # "product's" -> "product" (ambiguous, so we drop it)
]


def clean_text(text):
    """Take one raw review and return cleaned text.

    Steps: handle missing -> lowercase -> remove HTML -> remove URLs ->
    expand contractions -> remove punctuation/digits -> fix whitespace.
    Negation words (not, no, never) are deliberately KEPT.
    """
    # 1. Missing or non-text values become an empty string
    if not isinstance(text, str):
        return ""

    # 2. Lowercase (so "Great" and "great" are the same word)
    text = text.lower()

    # 3. Remove HTML tags such as <br /> or <b>
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Remove URLs
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)

    # 5. Expand contractions (convert curly apostrophes to normal ones first)
    text = text.replace("’", "'")
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)

    # 6. Keep only letters; punctuation, digits and symbols become spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 7. Collapse repeated spaces and trim the ends
    text = re.sub(r"\s+", " ", text).strip()

    return text