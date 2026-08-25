"""Vectorized text cleanup with np.strings and variable-width StringDType."""

import numpy as np


TOLERANCES = {}


def results():
    string_dtype = np.dtypes.StringDType()
    raw = np.array(["  ADA_Lovelace ", "grace_HOPPER", " Linus_torvalds  "], dtype=string_dtype)
    stripped = np.strings.strip(raw)
    lowered = np.strings.lower(stripped)
    spaced = np.strings.replace(lowered, "_", " ")
    title_words = np.strings.title(spaced)
    lengths = np.strings.str_len(title_words)
    contains_space = np.strings.find(title_words, " ")
    return {
        "cleaned": title_words,
        "lengths": lengths,
        "separator_positions": contains_space,
    }


def main():
    out = results()
    print("cleaned names:", out["cleaned"])
    print("lengths:", out["lengths"])
    assert np.array_equal(out["cleaned"], ["Ada Lovelace", "Grace Hopper", "Linus Torvalds"])
    assert np.array_equal(out["lengths"], [12, 12, 14])
    assert np.array_equal(out["separator_positions"], [3, 5, 5])


if __name__ == "__main__":
    main()
