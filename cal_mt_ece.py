#!/usr/bin/env python3

import argparse, os
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import sacrebleu
import matplotlib.pyplot as plt

def load_marian(direction):
    if direction == "en-de":
        name = "Helsinki-NLP/opus-mt-en-de"
    elif direction == "de-en":
        name = "Helsinki-NLP/opus-mt-de-en"
    else:
        raise ValueError("direction must be 'en-de' or 'de-en'")
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

def get_text_pairs(dataset, direction):
    src, tgt = [], []
    for item in dataset:
        if direction == "en-de":
            src.append(item["translation"]["en"])
            tgt.append(item["translation"]["de"])
        else:
            src.append(item["translation"]["de"])
            tgt.append(item["translation"]["en"])
    return src, tgt

@torch.no_grad()
def translate_with_conf(model, tokenizer, sentences, device, max_length=128):
    """
    Translate and compute per-sentence 'confidence' = mean chosen-token probability
    obtained from generate(..., output_scores=True).
    Returns: translations [N], confidences [N] in [0,1].
    """
    outs, confs = [], []
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model.generate(
            **enc,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,   # gives list[time] of logits
            num_beams=1           # greedy for cleaner confidence interpretation
        )
        # Decode hypothesis
        hyp = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        outs.append(hyp)

        # Compute mean probability of the chosen token at each time step
        # out.scores is a list of length T, each tensor shape [batch=1, vocab]
        # out.sequences contains BOS + tokens; chosen token at step t corresponds to sequences[0, t+1]
        scores = out.scores                          # list of length T
        seq = out.sequences[0]                       # shape [T+1] including BOS
        step_probs = []
        for t, logits_t in enumerate(scores):
            token_id = int(seq[t+1].item())          # token generated at step t
            probs_t = torch.softmax(logits_t[0], dim=-1)  # [vocab]
            step_probs.append(float(probs_t[token_id].item()))
        conf = float(np.mean(step_probs)) if step_probs else 0.0
        confs.append(conf)
    return outs, np.array(confs, dtype=np.float32)

def sentence_chrf_scores(hyps, refs):
    chrf = sacrebleu.CHRF()  # default: beta=2, char order up to 6
    scores = [chrf.sentence_score(h, [r]).score/100.0 for h, r in zip(hyps, refs)]  # normalize to [0,1]
    return np.array(scores, dtype=np.float32)

def compute_ece(conf, labels, n_bins=10):
    """Classic ECE: bucket by confidence, compare mean confidence vs accuracy."""
    assert conf.shape == labels.shape
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_confs, bin_accs, bin_sizes = [], [], []
    for b in range(n_bins):
        idx = (conf > bins[b]) & (conf <= bins[b+1]) if b < n_bins-1 else (conf > bins[b]) & (conf <= bins[b+1]+1e-8)
        n = int(np.sum(idx))
        if n == 0:
            bin_confs.append(np.nan); bin_accs.append(np.nan); bin_sizes.append(0)
            continue
        acc = float(np.mean(labels[idx]))
        cbar = float(np.mean(conf[idx]))
        ece += (n / len(conf)) * abs(acc - cbar)
        bin_confs.append(cbar); bin_accs.append(acc); bin_sizes.append(n)
    return ece, np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes), bins

def reliability_plot(bin_confs, bin_accs, direction, out_png):
    # Clean NaNs
    m = ~np.isnan(bin_confs) & ~np.isnan(bin_accs)
    x = bin_confs[m]; y = bin_accs[m]
    plt.figure()
    plt.plot([0,1],[0,1],"--",label="Perfect calibration")
    plt.plot(x, y, marker="o")
    plt.xlabel("Mean confidence (bin)")
    plt.ylabel("Accuracy (bin)")
    plt.title(f"Reliability Diagram – {direction.upper()}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[save] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test[:100]", help="HF split slice (e.g., test[:50])")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="reliability_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load WMT14 de-en test slice once
    ds = load_dataset("wmt14", "de-en", split=args.split)

    results = {}
    for direction in ["en-de", "de-en"]:
        print(f"\n=== Direction: {direction} ===")
        tokenizer, model, device = load_marian(direction)
        src, refs = get_text_pairs(ds, direction)

        # Translate + confidence
        hyps, conf = translate_with_conf(model, tokenizer, src, device, max_length=args.max_len)

        # Sentence-level chrF in [0,1]
        chrf = sentence_chrf_scores(hyps, refs)

        # Convert chrF to binary "correctness" by median threshold (data-driven)
        thr = float(np.median(chrf))
        labels = (chrf >= thr).astype(np.float32)

        # ECE + reliability
        ece, bin_confs, bin_accs, bin_sizes, bins = compute_ece(conf, labels, n_bins=args.n_bins)
        results[direction] = dict(ece=ece, thr=thr, conf=conf, chrf=chrf)

        # Plot
        out_png = os.path.join(args.out_dir, f"reliability_{direction.replace('-','_')}.png")
        reliability_plot(bin_confs, bin_accs, direction, out_png)

        # Quick printout
        print(f"median chrF threshold: {thr:.3f}")
        print(f"ECE({direction}) = {ece:.4f}")
        print(f"mean confidence: {np.mean(conf):.3f} | mean chrF: {np.mean(chrf):.3f}")

    # Directional asymmetry summary
    if all(k in results for k in ("en-de", "de-en")):
        delta_ece = results["en-de"]["ece"] - results["de-en"]["ece"]
        print("\n==== Summary ====")
        print(f"ECE EN→DE = {results['en-de']['ece']:.4f}")
        print(f"ECE DE→EN = {results['de-en']['ece']:.4f}")
        print(f"ΔECE (EN→DE - DE→EN) = {delta_ece:.4f}")

if __name__ == "__main__":
    main()