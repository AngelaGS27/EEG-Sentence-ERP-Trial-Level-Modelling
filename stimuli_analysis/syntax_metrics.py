import argparse
import pandas as pd
import numpy as np
import spacy


def dependency_distance(doc):
    distances = []
    for token in doc:
        if token.head.i != token.i:
            distances.append(abs(token.i - token.head.i))
    return np.mean(distances) if distances else np.nan


def parse_depth(token):
    if token.head == token:
        return 0
    return 1 + parse_depth(token.head)


def compute_syntax(sentence, nlp):
    doc = nlp(str(sentence))

    depths = [parse_depth(tok) for tok in doc]
    n_subordinate = sum(1 for tok in doc if tok.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"})

    return {
        "syntax_n_tokens": len(doc),
        "syntax_mean_dependency_distance": dependency_distance(doc),
        "syntax_max_parse_depth": max(depths) if depths else np.nan,
        "syntax_mean_parse_depth": np.mean(depths) if depths else np.nan,
        "syntax_n_subordinate_clauses": n_subordinate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tsv")
    parser.add_argument("--text-col", default="sentence")
    parser.add_argument("--output", default="language_outputs/syntax_metrics.tsv")
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    nlp = spacy.load(args.model)

    rows = []
    for sent in df[args.text_col]:
        rows.append(compute_syntax(sent, nlp))

    syntax_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), syntax_df], axis=1)

    out.to_csv(args.output, sep="\t", index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()