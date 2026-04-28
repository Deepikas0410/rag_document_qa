def normalize_text(text):
    return text.lower().strip()


# =========================
# Exact Match
# =========================
def exact_match(predicted, actual):
    return int(normalize_text(predicted) == normalize_text(actual))


# =========================
# F1 Score
# =========================
def f1_score(predicted, actual):
    pred_tokens = normalize_text(predicted).split()
    actual_tokens = normalize_text(actual).split()

    common = set(pred_tokens) & set(actual_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(actual_tokens)

    return 2 * precision * recall / (precision + recall)