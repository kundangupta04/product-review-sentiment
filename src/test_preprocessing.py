from src.preprocessing import clean_text

# (input, expected output)
test_cases = [
    ("Great product!!! Loved it :)", "great product loved it"),
    ("This is <b>awesome</b><br />Buy now", "this is awesome buy now"),
    ("Visit http://example.com for details", "visit for details"),
    ("The product is NOT good.", "the product is not good"),
    ("It doesn't work and I can't return it", "it does not work and i can not return it"),
    ("I won't buy this. Never again!", "i will not buy this never again"),
    ("   too    many     spaces   ", "too many spaces"),
    ("Delivered in 2 days, rated 5/5", "delivered in days rated"),
    (None, ""),
    (float("nan"), ""),
    ("!!! ??? ...", ""),
]

passed = 0
for raw, expected in test_cases:
    result = clean_text(raw)
    ok = result == expected
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {raw!r}\n         -> {result!r}")
    if not ok:
        print(f"         expected: {expected!r}")

print(f"\n{passed}/{len(test_cases)} tests passed")

# Negation words must survive cleaning
for word in ["not", "no", "never"]:
    assert word in clean_text(f"This is {word} good").split(), f"'{word}' was removed!"
print("Negation words (not, no, never) are preserved.")