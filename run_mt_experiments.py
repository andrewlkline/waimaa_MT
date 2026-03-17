"""
run_mt_experiments.py
Retrieval-augmented few-shot MT experiments for Waima'a → English.
Replicates and extends the methodology of Merx et al. (2024) on Mambai.

Models supported:
  --model gemini-1.5-pro
  --model gemini-1.5-flash
  --model ollama:llama3.1        (requires Ollama running locally)
  --model ollama:mistral

Usage:
  python3 run_mt_experiments.py \\
      --corpus parallel_corpus.json \\
      --dictionary gloss_dictionary.json \\
      --model gemini-1.5-flash \\
      --api-key YOUR_KEY \\
      --n-tfidf 5 --n-embed 5 --use-dict \\
      --out-dir results/

Install dependencies first:
  pip install scikit-learn sacrebleu google-generativeai
  pip install sentence-transformers        # for --n-embed > 0
"""

import json
import re
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict


# ── Optional imports (fail gracefully with clear messages) ────────────────────

def require(pkg, install):
    try:
        return __import__(pkg)
    except ImportError:
        raise SystemExit(f"\nMissing package: {pkg}\nInstall with: pip install {install}\n")


# ── Constants ─────────────────────────────────────────────────────────────────

TEST_GENRE   = "Pear_bendita2"   # held-out test set
PROMPT_TEMPLATE = """\
You are a translator for the Waima'a language (also spelled Waimaha), \
spoken in Timor-Leste. Translate the English sentence into Waima'a. \
Provide only the Waima'a translation, nothing else.

# Example sentences
{examples}

# Dictionary entries
{dictionary}

# Translate this sentence
English: {source}
Waima'a:"""

PROMPT_TEMPLATE_STRICT = """\
Translate the following English sentence into Waima'a (a language of Timor-Leste).

Rules:
- Output ONLY the Waima'a translation
- Do NOT explain your reasoning
- Do NOT say you cannot translate
- Do NOT add any commentary before or after
- If uncertain, output your best attempt based on the examples below

# Example sentences
{examples}

# Dictionary entries
{dictionary}

English: {source}
Waima'a:"""


# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus(path):
    pairs = json.loads(Path(path).read_text(encoding="utf-8"))
    train = [p for p in pairs if p["genre"] != TEST_GENRE]
    test  = [p for p in pairs if p["genre"] == TEST_GENRE]
    print(f"Corpus: {len(train)} train pairs, {len(test)} test pairs")
    return train, test


def load_dictionary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def build_tfidf(train):
    sklearn_feature = require("sklearn.feature_extraction.text",
                               "scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    english_sents = [p["english"] for p in train]
    vectorizer = TfidfVectorizer().fit(english_sents)
    matrix = vectorizer.transform(english_sents)
    return vectorizer, matrix, cosine_similarity


def retrieve_tfidf(source, vectorizer, matrix, cosine_similarity, train, n):
    if n == 0:
        return []
    import numpy as np
    vec = vectorizer.transform([source])
    sims = cosine_similarity(vec, matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:n]
    return [train[i] for i in top_idx]


# ── Semantic embedding retrieval ──────────────────────────────────────────────

def build_embeddings(train):
    SentenceTransformer = require("sentence_transformers",
                                  "sentence-transformers").SentenceTransformer
    import numpy as np
    print("Loading LaBSE embedding model (first run downloads ~1.8GB)...")
    model = SentenceTransformer("LaBSE")
    english_sents = [p["english"] for p in train]
    print(f"Encoding {len(english_sents)} training sentences...")
    embeddings = model.encode(english_sents, show_progress_bar=True,
                              batch_size=64, normalize_embeddings=True)
    return model, embeddings


def retrieve_embed(source, embed_model, embeddings, train, n):
    if n == 0:
        return []
    import numpy as np
    vec = embed_model.encode([source], normalize_embeddings=True)
    sims = (embeddings @ vec.T).flatten()
    top_idx = np.argsort(sims)[::-1][:n]
    return [train[i] for i in top_idx]


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(source, retrieved, dictionary, use_dict, strict=False):
    # Deduplicate retrieved examples (tfidf + embed may overlap)
    seen = set()
    unique = []
    for p in retrieved:
        key = p["english"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    example_lines = "\n".join(
        f"English: {p['english']}\nWaima'a: {p['waimaa']}"
        for p in unique
    ) or "(none)"

    dict_lines = ""
    if use_dict and dictionary:
        source_words = set(re.findall(r"[a-z']+", source.lower()))
        entries = []
        for word, glosses in dictionary.items():
            if word.lower() in source_words:
                entries.append(f"Waima'a: {word} = {', '.join(glosses)}")
        dict_lines = "\n".join(entries) if entries else "(none)"
    else:
        dict_lines = "(none)"

    template = PROMPT_TEMPLATE_STRICT if strict else PROMPT_TEMPLATE
    return template.format(
        examples=example_lines,
        dictionary=dict_lines,
        source=source,
    )


# ── Model backends ────────────────────────────────────────────────────────────

def call_gemini(prompt, model_name, api_key, retry=5):
    """Call Gemini REST API directly — no SDK, no dependency issues."""
    import urllib.request
    import urllib.error
    import json as _json
    # Strip accidental "models/" prefix
    model_name = model_name.removeprefix("models/")
    # Unversioned aliases (e.g. gemini-2.0-flash) only work on v1, not v1beta
    # Map them to their stable versioned equivalents
    alias_map = {
        "gemini-2.0-flash":      "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-001",
    }
    model_name = alias_map.get(model_name, model_name)

    # gemini-2.0 models require v1; newer models use v1beta
    version = "v1" if model_name.startswith("gemini-2.0") else "v1beta"
    url = (f"https://generativelanguage.googleapis.com/{version}/models/"
           f"{model_name}:generateContent?key={api_key}")
    # Disable thinking for 2.5+ models — thinking tokens are billed as output
    # and add no quality benefit for a simple translation task
    thinking_config = {}
    if any(x in model_name for x in ["2.5", "3-", "3."]):
        thinking_config = {"thinkingConfig": {"thinkingBudget": 0}}

    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, **thinking_config},
    }).encode()

    for attempt in range(retry):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = _json.loads(r.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if attempt < retry - 1:
                # 429 = quota exceeded: wait longer before retry
                wait = 60 if e.code == 429 else 2 ** attempt
                print(f"  Gemini HTTP {e.code} ({body[:80]}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Gemini failed after {retry} attempts: {e.code} {body[:200]}")
                return ""
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Gemini failed after {retry} attempts: {e}")
                return ""


def call_ollama(prompt, model_name, retry=3):
    """Calls Ollama running locally at http://localhost:11434"""
    import urllib.request, json as _json
    url = "http://localhost:11434/api/generate"
    payload = _json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }).encode()
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = _json.loads(r.read())
                return data.get("response", "").strip()
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Ollama failed: {e}. Is `ollama serve` running?")
                return ""


def call_claude(prompt, model_name, api_key, retry=5):
    """Call Anthropic Claude API directly via REST."""
    import urllib.request
    import urllib.error
    import json as _json

    url = "https://api.anthropic.com/v1/messages"
    payload = _json.dumps({
        "model":      model_name,
        "max_tokens": 256,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = _json.loads(r.read())
            return data["content"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if attempt < retry - 1:
                wait = 60 if e.code == 429 else 2 ** attempt
                print(f"  Claude HTTP {e.code} ({body[:80]}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Claude failed after {retry} attempts: {e.code} {body[:200]}")
                return ""
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Claude failed after {retry} attempts: {e}")
                return ""


def translate(prompt, model_spec, api_key):
    if model_spec.startswith("ollama:"):
        return call_ollama(prompt, model_spec[7:])
    elif model_spec.startswith("claude-"):
        return call_claude(prompt, model_spec, api_key)
    else:
        return call_gemini(prompt, model_spec, api_key)


# ── Evaluation ────────────────────────────────────────────────────────────────

def score(hypotheses, references):
    sacrebleu = require("sacrebleu", "sacrebleu")
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    chrfpp = sacrebleu.corpus_chrf(hypotheses, [references],
                                   beta=2, word_order=2)
    return {
        "bleu":    round(bleu.score, 2),
        "chrf":    round(chrf.score, 2),
        "chrfpp":  round(chrfpp.score, 2),
    }


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(train, test, dictionary, model_spec, api_key,
                   n_tfidf, n_embed, use_dict, strict_prompt=False,
                   tfidf_components=None, embed_components=None):

    label = (f"model={model_spec}  n_tfidf={n_tfidf}  "
             f"n_embed={n_embed}  use_dict={use_dict}")
    print(f"\n{'─'*60}")
    print(f"Running: {label}")
    print(f"{'─'*60}")

    hypotheses = []
    references = [p["waimaa"] for p in test]

    for i, pair in enumerate(test):
        source = pair["english"]

        # Retrieve examples
        tfidf_hits = retrieve_tfidf(source, *tfidf_components, train, n_tfidf) \
                     if tfidf_components and n_tfidf > 0 else []
        embed_hits = retrieve_embed(source, *embed_components, train, n_embed) \
                     if embed_components and n_embed > 0 else []

        # Interleave: tfidf first, then embed (dedup handled in build_prompt)
        retrieved = tfidf_hits + embed_hits

        prompt = build_prompt(source, retrieved, dictionary, use_dict, strict=strict_prompt)
        translation = translate(prompt, model_spec, api_key)
        hypotheses.append(translation)

        print(f"  [{i+1:2d}/{len(test)}] ENG: {source[:50]}")
        print(f"         WAI: {translation[:50]}")

        # Rate limiting: 1s gap is enough for paid tiers
        if not model_spec.startswith("ollama:"):
            time.sleep(1)

    scores = score(hypotheses, references)
    print(f"\n  BLEU={scores['bleu']}  ChrF={scores['chrf']}  ChrF++={scores['chrfpp']}")

    return {
        "model":     model_spec,
        "n_tfidf":   n_tfidf,
        "n_embed":   n_embed,
        "use_dict":  use_dict,
        "strict_prompt": strict_prompt,
        "n_test":    len(test),
        **scores,
        "hypotheses": hypotheses,
        "references": references,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Waima'a MT experiments via retrieval-augmented LLM prompting"
    )
    parser.add_argument("--corpus",     required=True, help="parallel_corpus.json")
    parser.add_argument("--dictionary", required=True, help="gloss_dictionary.json")
    parser.add_argument("--model",      required=True,
                        help="e.g. gemini-2.0-flash, gemini-2.5-flash, ollama:llama3.1")
    parser.add_argument("--api-key",    default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--claude-key", default="",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--n-tfidf",    type=int, default=5,
                        help="TF-IDF examples per prompt (default 5)")
    parser.add_argument("--n-embed",    type=int, default=0,
                        help="Embedding examples per prompt (default 0; requires sentence-transformers+torch)")
    parser.add_argument("--use-dict",   action="store_true",
                        help="Include dictionary entries in prompt")
    parser.add_argument("--strict-prompt", action="store_true",
                        help="Use stricter prompt that forbids meta-commentary (good for Sonnet)")
    parser.add_argument("--sweep",      action="store_true",
                        help="Run full hyperparameter sweep (ignores n-tfidf/n-embed/use-dict)")
    parser.add_argument("--out-dir",    default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    # API key from env if not passed
    import os
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    claude_key = args.claude_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if args.model.startswith("claude-"):
        if not claude_key:
            raise SystemExit("Pass --claude-key or set ANTHROPIC_API_KEY env var")
        api_key = claude_key  # reuse api_key slot for routing
    elif not args.model.startswith("ollama:") and not api_key:
        raise SystemExit("Pass --api-key or set GEMINI_API_KEY environment variable")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_corpus(args.corpus)
    dictionary  = load_dictionary(args.dictionary)

    # Build retrieval indices (once, reused across all experiments)
    print("\nBuilding TF-IDF index...")
    tfidf_components = build_tfidf(train)

    embed_components = None
    need_embed = args.n_embed > 0 or (args.sweep and False)  # embeddings opt-in only
    if need_embed:
        print("Building embedding index...")
        embed_components = build_embeddings(train)

    # Define experiment grid
    if args.sweep:
        if args.n_embed > 0:
            # Full sweep including embeddings (requires sentence-transformers)
            configs = [
                # (n_tfidf, n_embed, use_dict)
                (0,  0,  False),
                (0,  0,  True),
                (10, 0,  False),
                (10, 0,  True),
                (0,  10, False),
                (0,  10, True),
                (5,  5,  False),
                (5,  5,  True),
            ]
        else:
            # TF-IDF only sweep (default, no extra dependencies)
            configs = [
                # (n_tfidf, n_embed, use_dict)
                (0,  0,  False),
                (0,  0,  True),
                (5,  0,  False),
                (5,  0,  True),
                (10, 0,  False),
                (10, 0,  True),
            ]
    else:
        configs = [(args.n_tfidf, args.n_embed, args.use_dict)]

    all_results = []
    for n_tfidf, n_embed, use_dict in configs:
        result = run_experiment(
            train, test, dictionary,
            model_spec=args.model,
            api_key=api_key,
            n_tfidf=n_tfidf,
            n_embed=n_embed,
            use_dict=use_dict,
            strict_prompt=args.strict_prompt,
            tfidf_components=tfidf_components,
            embed_components=embed_components if n_embed > 0 else None,
        )
        all_results.append(result)

    # Save full results JSON
    safe_model = args.model.replace("/", "-").replace(":", "-")
    results_path = out_dir / f"results_{safe_model}.json"
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFull results saved to {results_path}")

    # Save summary CSV (scores only, no hypotheses)
    csv_path = out_dir / f"summary_{safe_model}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n_tfidf", "n_embed", "use_dict", "strict_prompt",
            "n_test", "bleu", "chrf", "chrfpp"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items()
                             if k not in ("hypotheses", "references")})
    print(f"Summary CSV saved to {csv_path}")

    # Print leaderboard
    print(f"\n{'═'*60}")
    print("RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"{'n_tfidf':>8} {'n_embed':>8} {'dict':>6} {'BLEU':>6} {'ChrF':>6} {'ChrF++':>7}")
    for r in sorted(all_results, key=lambda x: -x["bleu"]):
        print(f"{r['n_tfidf']:8d} {r['n_embed']:8d} {str(r['use_dict']):>6} "
              f"{r['bleu']:6.1f} {r['chrf']:6.1f} {r['chrfpp']:7.1f}")


if __name__ == "__main__":
    main()