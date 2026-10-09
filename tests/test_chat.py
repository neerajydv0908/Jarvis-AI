import json

import pytest

from jarvis.brain import cornell
from jarvis.brain.chat import ChatEngine
from jarvis.brain.chat_training import DialoguePairs, format_prompt


def test_rule_based_small_talk():
    chat = ChatEngine(enabled=False, fallback_responses=["Tell me more."], seed=0)
    assert not chat.generative
    assert any(word in chat.reply("I'm so bored") for word in ("joke", "Wikipedia"))
    assert chat.reply("qwerty asdf") == "Tell me more."


def test_clean_generated_text():
    assert ChatEngine._clean(" Sure, let's go.\nUser: hi") == "Sure, let's go."
    assert ChatEngine._clean("ok") == ""                       # too short
    assert ChatEngine._clean("I will kill you") == ""          # blocked words never reach the speaker


def test_prompt_format():
    assert format_prompt("hi", [("hello", "hey")]) == "User: hello\nJarvis: hey\nUser: hi\nJarvis:"


def test_cornell_original_format(tmp_path):
    folder = tmp_path / "cornell movie-dialogs corpus"
    folder.mkdir()
    sep = " +++$+++ "
    (folder / "movie_lines.txt").write_text("\n".join([
        sep.join(["L1", "u0", "m0", "BIANCA", "Can we make this quick?"]),
        sep.join(["L2", "u2", "m0", "CAMERON", "Well, I thought we'd start with pronunciation."]),
        sep.join(["L3", "u0", "m0", "BIANCA", "Not the <u>hacking</u> part."]),
    ]), encoding="iso-8859-1")
    (folder / "movie_conversations.txt").write_text(
        sep.join(["u0", "u2", "m0", "['L1', 'L2', 'L3']"]) + "\n", encoding="iso-8859-1")
    pairs = cornell.load_pairs(tmp_path)
    assert pairs == [("Can we make this quick?", "Well, I thought we'd start with pronunciation."),
                     ("Well, I thought we'd start with pronunciation.", "Not the hacking part.")]


def test_cornell_convokit_format(tmp_path):
    folder = tmp_path / "movie-corpus"
    folder.mkdir()
    # ConvoKit's real key is "reply-to"
    rows = [{"id": "L1", "reply-to": None, "text": "Where are you going?"},
            {"id": "L2", "reply-to": "L1", "text": "Home."},
            {"id": "L3", "reply-to": "L2", "text": "Why?"}]
    (folder / "utterances.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert cornell.load_pairs(tmp_path, min_words=1) == [("Where are you going?", "Home."), ("Home.", "Why?")]


def test_missing_corpus(tmp_path):
    assert cornell.find_corpus(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        cornell.load_pairs(tmp_path)


def _tiny_gpt2(tmp_path):
    """A 2-layer GPT-2 with a home-made tokenizer: exercises the real code path offline."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    corpus = ["User: hello there\nJarvis: hi, how are you?", "User: what are you doing\nJarvis: just waiting.",
              "User: i am tired\nJarvis: get some rest."] * 20
    bpe = tokenizers.ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=300, min_frequency=1, special_tokens=["<|endoftext|>"])
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|endoftext|>",
                                                     pad_token="<|endoftext|>")
    config = transformers.GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=32, n_layer=2, n_head=2,
                                     bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    return transformers.GPT2LMHeadModel(config), tokenizer


def test_fine_tune_and_generate(tmp_path):
    from jarvis.brain.chat_training import fine_tune

    model, tokenizer = _tiny_gpt2(tmp_path)
    pairs = [("hello there", "hi, how are you?"), ("what are you doing", "just waiting."),
             ("i am tired", "get some rest.")] * 10
    dataset = DialoguePairs(pairs[:1], tokenizer, max_length=64)
    ids, labels = dataset[0]
    prompt_len = len(tokenizer(format_prompt("hello there"), add_special_tokens=False)["input_ids"])
    assert labels[:prompt_len] == [-100] * prompt_len        # loss only on Jarvis's words
    assert labels[-1] == tokenizer.eos_token_id

    metrics = fine_tune(model, tokenizer, pairs, "cpu", epochs=3, batch_size=8, lr=5e-3, log_every=0)
    assert metrics["val_loss_after"] < metrics["val_loss_before"]

    out = tmp_path / "chat"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    chat = ChatEngine(out, device="cpu", seed=0)
    assert chat.generative
    assert isinstance(chat.reply("hello there"), str) and chat.reply("hello there")
