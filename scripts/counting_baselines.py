"""Count-based reference models (no neural network) for the validation set used in the study.

Reproduces the unigram (6.32) and bigram (4.94) validation losses reported in the report
(Section 3.4). Uses exactly the same book, cleaning, tokenizer and 90/10 split as the notebook.

    pip install tiktoken
    python scripts/counting_baselines.py
"""
import collections
import math
import re
import urllib.request

import tiktoken

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"   # Pride and Prejudice
ALPHA = 0.1                                                    # add-alpha smoothing of the unigram model
LAMBDAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)                       # bigram/unigram interpolation weights


def strip_gutenberg(text):
    """Same cleaning as the notebook: licence header/footer, [Illustration] tags, indentation."""
    start = text.find("*** START OF")
    start = text.find("\n", start) + 1 if start != -1 else 0
    end = text.find("*** END OF")
    text = text[start:end if end != -1 else len(text)]
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req).read().decode("utf-8-sig", errors="ignore")
    ids = tiktoken.get_encoding("gpt2").encode(strip_gutenberg(raw), allowed_special={"<|endoftext|>"})
    split = int(0.9 * len(ids))
    train, val = ids[:split], ids[split:]
    print(f"tokens: {len(ids):,} (train {len(train):,}, val {len(val):,})")

    V, N = 50257, len(train)
    uni = collections.Counter(train)
    big = collections.Counter(zip(train, train[1:]))
    prev = collections.Counter(train[:-1])
    p_uni = lambda t: (uni[t] + ALPHA) / (N + ALPHA * V)

    print(f"uniform guessing : {math.log(V):.3f}")
    print(f"unigram          : {-sum(math.log(p_uni(t)) for t in val) / len(val):.3f}")
    for lam in LAMBDAS:
        loss = -sum(math.log(lam * (big[(a, b)] / prev[a] if prev[a] else 0.0) + (1 - lam) * p_uni(b))
                    for a, b in zip(val, val[1:])) / (len(val) - 1)
        print(f"bigram (lambda={lam}): {loss:.3f}")
    print("Note: lambda is selected on the validation set, so the best bigram value is slightly optimistic.")


if __name__ == "__main__":
    main()
