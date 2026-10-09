from jarvis.nlp.preprocess import NUM, UNK, clean_text, normalize_wake_word, strip_wake_word


def test_clean_text_expands_contractions_and_times():
    assert clean_text("What's the time? I can't wait") == "what is the time? i can not wait"
    assert clean_text("Remind me at 5:30PM") == "remind me at 5:30 pm"
    assert clean_text("call at 5 p.m.") == "call at 5 pm"
    assert clean_text("whats up") == "what is up"


def test_normalize_keeps_intent_words(preprocessor):
    assert preprocessor.normalize("Who are you?") == ["who", "you"]
    assert preprocessor.normalize("How are you?") == ["how", "you"]
    assert preprocessor.normalize("What is your name") == ["what", "your", "name"]
    assert preprocessor.normalize("what can you do") == ["what", "can", "you", "do"]


def test_normalize_lemmatises_and_maps_numbers(preprocessor):
    assert preprocessor.normalize("Delete all my reminders!") == ["delete", "all", "my", "reminder"]
    assert preprocessor.normalize("remind me in 10 minutes") == ["remind", "me", NUM, "minute"]
    assert preprocessor.normalize("in ten minutes") == [NUM, "minute"]


def test_template_placeholder_becomes_unknown(preprocessor):
    assert preprocessor.normalize("weather in xxslot") == ["weather", UNK]


def test_stemming_option():
    from jarvis.nlp.preprocess import TextPreprocessor

    stemmer = TextPreprocessor(method="stem")
    assert stemmer.normalize("reminders running") == ["remind", "run"]


def test_sentence_split(preprocessor):
    assert len(preprocessor.sentences("One sentence. Another one! A third?")) == 3


def test_wake_word_handling():
    assert strip_wake_word("hey jarvis, what's the weather in chicago") == "what's the weather in chicago"
    assert strip_wake_word("what time is it jarvis") == "what time is it"
    assert strip_wake_word("hey jarvis") == ""
    assert normalize_wake_word("thanks jarvis") == "thanks you"
    assert normalize_wake_word("what does jarvis stand for") == "what does you stand for"
    assert normalize_wake_word("hello jarvis") == "hello you"
