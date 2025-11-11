import torch
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import sacrebleu
import evaluate
import math
# from comet import download_model,load_from_checkpoint
from sacrebleu import corpus_bleu

# import torch
# from torch.utils.data import DataLoader

# # 🔧 macOS 安全补丁：禁用 multiprocessing_context 逻辑
# _orig_init = DataLoader.__init__
# def _patched_init(self, *args, **kwargs):
#     kwargs.pop("multiprocessing_context", None)  # 强制移除该参数
#     _orig_init(self, *args, **kwargs)
# DataLoader.__init__ = _patched_init

# ==========================
# 1. 初始化模型和Tokenizer
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
model.train()  # 注意：必须train模式才能启用dropout

# ==========================
# 2. 数据加载（示例：WMT14子集）
# ==========================
# 这里假设你已经准备好一个list形式的句对
# 为方便演示，我们用HuggingFace的wmt14子集（或你自己的数据）
dataset = load_dataset("wmt14", "de-en", split="test[:5]")  # 仅取前200条做demo

def get_text_pairs(direction="en-de"):
    src, tgt = [], []
    for item in dataset:
        if direction == "en-de":
            src.append(item["translation"]["en"])
            tgt.append(item["translation"]["de"])
        else:
            src.append(item["translation"]["de"])
            tgt.append(item["translation"]["en"])
    return src, tgt

# ==========================
# 3. MC Dropout推理函数
# ==========================
def mc_dropout_translate(model, tokenizer, sentences, src_lang, tgt_lang, K=10, max_length=100):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    entropies = []
    outputs_all = []

    for _ in range(K):
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=max_length,
                num_beams=1,
                do_sample=True
            )
            logits = model(**encoded, labels=generated_tokens).logits
            probs = F.softmax(logits, dim=-1)
            # 对每次生成求平均熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()
            entropies.append(entropy)

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs_all.append(decoded)

    # 平均多次采样的熵
    mean_entropy = np.mean(entropies)
    return outputs_all[0], mean_entropy


# ==========================
# 4. 校准指标计算函数
# ==========================
def expected_calibration_error(confidences, accuracies, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    acc_bins, conf_bins = [], []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            acc_bin = accuracies[mask].mean()
            conf_bin = confidences[mask].mean()
            acc_bins.append(acc_bin)
            conf_bins.append(conf_bin)
            ece += abs(acc_bin - conf_bin) * mask.mean()
        else:
            acc_bins.append(np.nan)
            conf_bins.append(np.nan)
    return ece, acc_bins, conf_bins, bins

def brier_score(probs, labels):
    return np.mean((np.array(probs) - np.array(labels))**2)

# =====================
# 5. 主实验函数
# =====================
def run_direction(direction):
    src_lang, tgt_lang = ("en_XX", "de_DE") if direction == "en-de" else ("de_DE", "en_XX")
    src, tgt = get_text_pairs(direction)

    translations, entropies, confidences, accuracies = [], [], [], []
    print(f"Running {direction.upper()}...")

    for s, t in tqdm(zip(src, tgt), total=len(src)):
        outs, entropy = mc_dropout_translate(model, tokenizer, [s], src_lang, tgt_lang)
        translations.append(outs[0])
        entropies.append(entropy)
        conf = math.exp(-entropy)
        confidences.append(conf)
        accuracies.append(1.0 if t.strip() == outs[0].strip() else 0.0)

    # ---- 校准指标 ----
    ece, acc_bins, conf_bins, bins = expected_calibration_error(np.array(confidences), np.array(accuracies))
    brier = brier_score(confidences, accuracies)

    # ---- 翻译质量 ----
    bleu = sacrebleu.corpus_bleu(translations, [tgt]).score
    # comet = evaluate.load("comet")
    # # comet_score = comet.compute(predictions=translations, references=tgt)["mean_score"]
    # comet_score = comet.compute(
    # sources=src,
    # predictions=translations,
    # references=tgt,
    # gpus=0,
    # progress_bar=False
    # )["mean_score"]


    # 直接加载 COMET 模型，避免 evaluate 的多进程封装
    # comet_model_path = "Unbabel/wmt22-comet-da"
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_model = load_from_checkpoint(comet_model_path)
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(src, translations, tgt)]

    
    # 在 macOS/CPU 下强制单线程推理
    comet_model.hparams.num_workers = 1
    comet_outputs = comet_model.predict(data, batch_size=4, gpus=0, progress_bar=False)
    comet_score = comet_outputs.system_score

    print(f"{direction.upper()} BLEU: {bleu:.2f}, COMET: {comet_score:.3f}")
    print(f"{direction.upper()} ECE: {ece:.4f}, Brier: {brier:.4f}")

    # ---- 可靠性曲线 ----
    plt.figure(figsize=(5, 5))
    plt.plot(conf_bins, acc_bins, marker='o', label=f"{direction.upper()}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram ({direction.upper()})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reliability_{direction}.png", dpi=300)
    plt.close()

    return ece, bleu, comet_score

# =====================
# 6. 执行双向实验
# =====================
ece_en_de, bleu_en_de, comet_en_de = run_direction("en-de")
ece_de_en, bleu_de_en, comet_de_en = run_direction("de-en")

print("\n=== Summary ===")
print(f"EN-DE: BLEU={bleu_en_de:.2f}, COMET={comet_en_de:.3f}, ECE={ece_en_de:.4f}")
print(f"DE-EN: BLEU={bleu_de_en:.2f}, COMET={comet_de_en:.3f}, ECE={ece_de_en:.4f}")
print(f"ΔECE = {ece_en_de - ece_de_en:.4f}")