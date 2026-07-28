"""Unit tests for raven.common.text.speakable."""

from raven.common import text


class TestUnspeakable:
    def test_empty(self):
        assert not text.is_speakable("")

    def test_whitespace_only(self):
        assert not text.is_speakable("   \t\n ")

    def test_lone_markdown_bullet(self):
        # The case that motivated the predicate: an answer ending "...naked eye:\n*" leaves
        # a final line that is just this, which survives Markdown stripping as literal text.
        assert not text.is_speakable("*")

    def test_markdown_leftovers(self):
        for fragment in ("**", "***", "---", "___", "```", "|", "#", "- ", "> "):
            assert not text.is_speakable(fragment), fragment

    def test_punctuation_only(self):
        for fragment in ("?!", "...", "-->", "…", "—", "(", "«»"):
            assert not text.is_speakable(fragment), fragment


class TestSpeakable:
    def test_plain_sentence(self):
        assert text.is_speakable("The capital of France is Paris.")

    def test_single_letter(self):
        assert text.is_speakable("I")

    def test_digits(self):
        assert text.is_speakable("42")

    def test_alphanumeric_amid_punctuation(self):
        # Punctuation does not disqualify text that also carries pronounceable content.
        assert text.is_speakable("*emphasis*")
        assert text.is_speakable("- a bullet with content")

    def test_non_ascii_scripts(self):
        # Unicode-aware, so the predicate does not quietly mute Raven's non-English paths:
        # Finnish (the subtitler's target language), and scripts with no ASCII at all.
        for fragment in ("Hyvää päivää", "Здравствуйте", "こんにちは", "مرحبا"):
            assert text.is_speakable(fragment), fragment
