# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    dataset = load_dataset("tweet_eval", "emotion", split="train[:1000]")
    corpus = [x["text"].strip() for x in dataset if x["text"].strip()]
    return corpus