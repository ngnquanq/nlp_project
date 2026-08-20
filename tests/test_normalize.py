from mt_pipeline.normalize import (
    clean_llm_generation,
    detokenize_chinese,
    score_form_chinese,
)


def test_chinese_round_trip():
    raw = "尊 母 黎 氏 曰 靈 顯"
    readable = detokenize_chinese(raw)
    assert readable == "尊母黎氏曰靈顯"
    assert score_form_chinese(readable) == raw


def test_clean_llm_generation_removes_known_wrappers():
    assert clean_llm_generation("<think>hidden</think>\n译文：尊 母") == "尊母"

