
import os
import sys

sys.path.insert(0, os.getcwd())
APP_PATH = os.path.join(os.getcwd(), "app.py")

from streamlit.testing.v1 import AppTest


def run_app(review_text):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception, f"App crashed on load: {at.exception}"
    at.text_area[0].input(review_text).run()
    at.button[0].click().run()
    assert not at.exception, f"App crashed after click: {at.exception}"
    return at


# 1. Page loads with the right title
at = AppTest.from_file(APP_PATH, default_timeout=30).run()
assert at.title[0].value == "Product Review Sentiment Analysis"
print("[PASS] App loads with the correct title")

# 2. Positive review
at = run_app("The product is excellent and worth the money.")
assert len(at.success) == 1
print("[PASS] Positive review ->", at.success[0].value)

# 3. Negative review
at = run_app("The product stopped working after two days. Very disappointed.")
assert len(at.error) == 1
print("[PASS] Negative review ->", at.error[0].value)

# 4. Empty input shows a warning, not a crash
at = run_app("")
assert len(at.warning) == 1
print("[PASS] Empty input -> warning:", at.warning[0].value)

print("\nSTEP 9 TEST PASSED")