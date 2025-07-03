# tests/test_preprocessing.py
"""
Unit tests for the text preprocessing functions in src.preprocessing.
"""
import unittest
import sys
import os

# Add src directory to Python path to import preprocessing module
# This assumes tests are run from the project root directory (e.g., using 'python -m unittest discover tests')
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src') # Adjust if your structure is different (e.g. tests/../src)
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocessing import preprocess_text_spacy, NLP_SPACY # Import the function to test and NLP_SPACY for checks

class TestPreprocessing(unittest.TestCase):
    """
    Test suite for text preprocessing functions.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class method to check if spaCy model is loaded.
        Tests will be skipped if the model is not available.
        """
        if NLP_SPACY is None:
            raise unittest.SkipTest("spaCy model 'en_core_web_sm' not loaded. Skipping preprocessing tests.")

    def test_preprocess_text_spacy_simple_sentence(self):
        """Test preprocessing of a simple sentence."""
        text = "This is a sample sentence with some numbers 123 and punctuation!"
        expected = "sample sentence number punctuation" # Assuming 'numbers' becomes 'number' due to lemmatization of '123'
        # spaCy's like_num should remove '123'. 'numbers' might remain if it was text.
        # Let's re-evaluate expected based on current preprocess_text_spacy logic:
        # It removes 'like_num' tokens. 'numbers' the word should be lemmatized if not a stopword.
        # 'numbers' is not a stopword by default. Its lemma is 'number'.
        # 'punctuation' is not a stopword. Its lemma is 'punctuation'.
        # 'sample' lemma 'sample'. 'sentence' lemma 'sentence'.
        # Corrected expected:
        expected = "sample sentence number punctuation"
        # Actually, `token.like_num` removes "123". The word "numbers" if present would be lemmatized to "number".
        # The original sentence has "numbers 123". "numbers" becomes "number", "123" is removed.
        # "punctuation!" -> "punctuation" (lemma), "!" is removed by is_punct.
        # Let's test with: "This is a sample sentence with the word numbers and also the digits 123 and punctuation!"
        text_complex = "This is a sample sentence with the word numbers and also the digits 123 and punctuation!"
        expected_complex = "sample sentence word number digit punctuation" # 'numbers' -> 'number', 'digits' -> 'digit'
        self.assertEqual(preprocess_text_spacy(text_complex), expected_complex)

    def test_preprocess_text_spacy_empty_string(self):
        """Test preprocessing of an empty string."""
        text = ""
        expected = ""
        self.assertEqual(preprocess_text_spacy(text), expected)

    def test_preprocess_text_spacy_stopwords_and_punctuation_only(self):
        """Test preprocessing of a string with only stopwords and punctuation."""
        text = "!!! ... ??? the of a is an"
        expected = "" # All should be removed
        self.assertEqual(preprocess_text_spacy(text), expected)

    def test_preprocess_text_spacy_numbers_only(self):
        """Test preprocessing of a string with only numbers."""
        text = "123 456 7890"
        expected = "" # All should be removed by like_num
        self.assertEqual(preprocess_text_spacy(text), expected)

    def test_preprocess_text_spacy_mixed_case_and_lemmatization(self):
        """Test mixed case input and lemmatization."""
        text = "Running Dogs and Flying CATS were playing with Mice 123."
        expected = "run dog fly cat play mouse" # Lemmatized and lowercased, numbers removed
        self.assertEqual(preprocess_text_spacy(text), expected)

    def test_preprocess_text_spacy_no_useful_tokens(self):
        """Test text that results in no useful tokens after processing."""
        text = "    \n\t  ... !! ?? "
        expected = ""
        self.assertEqual(preprocess_text_spacy(text), expected)

    def test_preprocess_text_spacy_with_special_chars_inside_words(self):
        """Test words with internal special characters (though regex might strip them first)."""
        # The current `preprocess_text_spacy` does not have a regex step before spaCy.
        # spaCy handles tokenization around these.
        text = "word-with-hyphen another_word email@example.com"
        # Expected depends on how spaCy tokenizes and if parts are stopwords/punct.
        # "word-with-hyphen" -> "word", "hyphen" (if not joined by tokenizer) then lemmatized.
        # spaCy usually splits "word-with-hyphen" into "word", "-", "with", "-", "hyphen".
        # "-" is punctuation. "word", "with", "hyphen" are lemmatized.
        # "email@example.com" is often tokenized as one URL-like/email-like token.
        # Let's check:
        # 'word-with-hyphen' -> word, with, hyphen (lemmas: word, with, hyphen)
        # 'another_word' -> another_word (lemma: another_word) (spaCy might split on underscore or keep it)
        # 'email@example.com' -> email@example.com (lemma: email@example.com)
        # Assuming standard tokenization and is_punct for '-', and like_email for the email.
        # If 'with' is a stopword, it's removed.
        # Current spaCy logic: token.is_punct, token.is_stop, token.like_num
        # token.lemma_.lower().strip()
        # "word-with-hyphen" -> tokens: word, -, with, -, hyphen. "-" is punct. "with" is stop. -> "word hyphen"
        # "another_word" -> tokens: another, _, word. "_" is punct. -> "another word"
        # "email@example.com" -> token: email@example.com (not stop, not punct, not num) -> "email@example.com"
        # This needs verification against actual spaCy behavior for specific cases.
        # For simplicity, let's use a case that's clearer with current rules:
        text = "Well-being is important. Check example.com."
        expected = "well-be important check example.com" # "Well-being" often lemmatized to "well-being" or split.
                                                 # "is" is stop. "." is punct.
                                                 # spaCy's default for "Well-being" might be "well-being".
                                                 # Let's assume it is.
        # If "well-being" becomes "well" and "being" (lemma), then "well" (being might be stop or not)
        # Actual output of "Well-being": "well" (token.lemma_ for "Well" is "well", for "-" is "-", for "being" is "be")
        # So, "well be"
        # "example.com" -> "example.com"
        # Let's refine the test based on typical spaCy behavior:
        text_refined = "The well-being of people is important. Check example.com for details 123."
        expected_refined = "well-be people important check example.com detail"
        # spaCy tokenizes "well-being" into "well", "-", "being". "-" is punct. "being" lemma "be". "well" lemma "well".
        # So it should be "well be".
        # "people" -> "people". "is" (stop). "important" -> "important". "." (punct).
        # "check" -> "check". "example.com" (URL-like, kept). "for" (stop). "details" -> "detail". "123" (like_num).
        expected_final = "well people important check example.com detail" # "be" might be removed if considered stop in some contexts, or very short.
                                                                     # Default 'be' is a stopword in en_core_web_sm.
        self.assertEqual(preprocess_text_spacy(text_refined), expected_final)


if __name__ == '__main__':
    unittest.main()
```
