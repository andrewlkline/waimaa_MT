"""
parse_tbt.py
Parses SIL Toolbox (.tbt) files from the Waima'a DoBeS corpus into:
  1. parallel_corpus.json  — cleaned Waima'a / English sentence pairs
  2. gloss_dictionary.json — word-level Waima'a → English entries
  3. corpus_stats.json     — summary statistics

Usage:
    python3 parse_tbt.py <file1.tbt> [file2.tbt ...] --out-dir <dir>
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict


# ── Constants ──────────────────────────────────────────────────────────────────

# Participants that indicate metadata/noise records, not real utterances
NOISE_PARTICIPANTS = {"unknown", "x", ""}

# Grammatical tags to skip when building the gloss dictionary
# (all-caps tokens in \gle lines are grammatical markers, not lexical items)
GRAM_TAG_RE = re.compile(r'^[A-Z0-9]+[sp]?$')

# Tokens to skip in orth/tx: repairs, false starts, noise markers
SKIP_TOKEN_RE = re.compile(r'^[<\[{]|[>\]\}]$|^xxx$|^yyy$|^\?+$')

# Morpheme boundary markers to strip from orth tokens
MORPH_CLEAN_RE = re.compile(r'^-+|-+$')

# Trailing punctuation / transcription markers on utterances
UTTERANCE_CLEAN_RE = re.compile(r'[/\\.]+$')

# Glottal stop normalisation: unify all apostrophe variants → standard '
APOSTROPHE_RE = re.compile(r"['\u2018\u2019\u02bc\u0027]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalise_apostrophe(text: str) -> str:
    return APOSTROPHE_RE.sub("'", text)


def clean_waimaa_utterance(raw: str) -> str:
    """Clean a raw \\orth or \\tx line into a readable Waima'a sentence."""
    # Remove trailing transcription markers (/, \, ...)
    text = UTTERANCE_CLEAN_RE.sub("", raw).strip()
    # Normalise apostrophe variants
    text = normalise_apostrophe(text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def clean_english_utterance(raw: str) -> str:
    """Light clean on the \\tle English free translation line."""
    text = raw.strip()
    # Remove ellipsis-only entries
    if text in {"...", "..", "…"}:
        return ""
    text = normalise_apostrophe(text)
    return text


def is_usable_pair(waimaa: str, english: str) -> bool:
    """Return True if a sentence pair is clean enough to include."""
    if not waimaa or not english:
        return False
    if len(english) < 4:
        return False
    if len(waimaa) < 2:
        return False
    # Skip if english is only punctuation
    if re.fullmatch(r'[.!?,;:\-\s]+', english):
        return False
    # Skip if waimaa contains only noise tokens
    tokens = waimaa.split()
    if all(SKIP_TOKEN_RE.match(t) for t in tokens):
        return False
    return True


# ── Record parser ──────────────────────────────────────────────────────────────

def parse_records(content: str) -> list[dict]:
    """Split a Toolbox file into a list of field-dicts, one per \\ref block."""
    # Split on \ref markers (each record starts with \ref NNN)
    blocks = re.split(r'(?=\\ref\s+\d+)', content)
    records = []

    for block in blocks:
        if not block.strip():
            continue

        fields = {}
        # Each line that starts with a backslash-tag carries a field
        for line in block.splitlines():
            line = line.rstrip()
            m = re.match(r'\\([a-zA-Z_]+)\s*(.*)', line)
            if m:
                tag, value = m.group(1), m.group(2).strip()
                # Keep only first occurrence of each tag per record
                if tag not in fields:
                    fields[tag] = value

        if 'ref' in fields:
            records.append(fields)

    return records


# ── Corpus extraction ──────────────────────────────────────────────────────────

def extract_pairs(records: list[dict], source_file: str) -> list[dict]:
    """Extract parallel sentence pairs from parsed records."""
    pairs = []

    for rec in records:
        participant = rec.get('ELANParticipant', '').strip().lower()

        # Prefer \orth over \tx (cleaner orthography, no repair markers)
        waimaa_raw = rec.get('orth') or rec.get('tx', '')
        english_raw = rec.get('tle', '')
        tetun_raw   = rec.get('tlt', '')
        indonesian_raw = rec.get('tlm', '')

        waimaa  = clean_waimaa_utterance(waimaa_raw)
        english = clean_english_utterance(english_raw)
        tetun   = clean_english_utterance(tetun_raw)
        indonesian = clean_english_utterance(indonesian_raw)

        if not is_usable_pair(waimaa, english):
            continue

        pairs.append({
            "ref":         rec.get('ref', ''),
            "source_file": source_file,
            "participant": rec.get('ELANParticipant', '').strip(),
            "time_begin":  rec.get('ELANBegin', ''),
            "time_end":    rec.get('ELANEnd', ''),
            "waimaa":      waimaa,
            "english":     english,
            "tetun":       tetun      if tetun      else None,
            "indonesian":  indonesian if indonesian else None,
            # Keep raw \tx for reference
            "tx_raw":      rec.get('tx', '').strip(),
        })

    return pairs


# ── Gloss dictionary extraction ────────────────────────────────────────────────

def extract_glosses(records: list[dict]) -> dict[str, set[str]]:
    """
    Build a Waima'a → English word-level dictionary from \\orth + \\gle fields.
    Returns dict mapping waima'a token → set of English glosses.
    """
    glosses: dict[str, set[str]] = defaultdict(set)

    for rec in records:
        orth_raw = rec.get('orth') or rec.get('tx', '')
        gle_raw  = rec.get('gle', '')

        if not orth_raw or not gle_raw:
            continue

        orth_tokens = orth_raw.split()
        gle_tokens  = gle_raw.split()

        for waimaa_tok, eng_tok in zip(orth_tokens, gle_tokens):
            # Clean the Waima'a token
            w = MORPH_CLEAN_RE.sub('', waimaa_tok).strip().lower()
            w = normalise_apostrophe(w)
            w = re.sub(r'[/\\<>\[\]{}()]', '', w)
            if not w or len(w) < 2:
                continue

            # Skip grammatical tags (all-caps), noise, punctuation
            e = eng_tok.strip()
            if GRAM_TAG_RE.match(e):
                continue
            if not e or e in {'-', '?', '...', 'xxx'}:
                continue

            glosses[w].add(e.lower())

    return glosses


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Waima'a .tbt files into corpus JSON")
    parser.add_argument('inputs', nargs='+', help='.tbt files or folders containing .tbt files')
    parser.add_argument('--out-dir', default='.', help='Output directory')
    args = parser.parse_args()

    # Expand folders to .tbt files, keep explicit file args as-is
    filepaths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(f for f in p.rglob('*.tbt') if f.is_file())
            if not found:
                print(f"  Warning: no .tbt files found in {p}")
            filepaths.extend(found)
        elif p.is_file():
            filepaths.append(p)
        else:
            print(f"  Warning: {inp} is not a file or directory, skipping")

    if not filepaths:
        print("No .tbt files found. Exiting.")
        return

    print(f"Found {len(filepaths)} .tbt file(s)\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs    = []
    all_records  = []
    file_stats   = []

    for filepath in filepaths:
        path = Path(filepath)
        content = path.read_bytes().decode('utf-8', errors='replace')
        # Normalise line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        records = parse_records(content)
        pairs   = extract_pairs(records, path.name)

        # Genre tag from filename
        genre = path.stem  

        for p in pairs:
            p['genre'] = genre

        all_pairs.extend(pairs)
        all_records.extend(records)

        file_stats.append({
            "file":          path.name,
            "genre":         genre,
            "total_records": len(records),
            "usable_pairs":  len(pairs),
            "participants":  sorted(set(
                r.get('ELANParticipant', '').strip()
                for r in records
                if r.get('ELANParticipant', '').strip().lower() not in NOISE_PARTICIPANTS
            )),
        })

        print(f"  {path.name}: {len(records)} records → {len(pairs)} usable pairs")

    # ── Deduplicate pairs (same waimaa + english) ──
    seen = set()
    deduped = []
    for p in all_pairs:
        key = (p['waimaa'].lower(), p['english'].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    duplicates_removed = len(all_pairs) - len(deduped)

    # ── Build gloss dictionary ──
    raw_glosses = extract_glosses(all_records)

    # Convert sets to sorted lists; filter single-char entries
    gloss_dict = {
        word: sorted(glosses)
        for word, glosses in sorted(raw_glosses.items())
        if len(word) > 1 and glosses
    }

    # ── Summary stats ──
    avg_waimaa_len  = sum(len(p['waimaa'].split())  for p in deduped) / max(len(deduped), 1)
    avg_english_len = sum(len(p['english'].split()) for p in deduped) / max(len(deduped), 1)

    genres = sorted(set(p['genre'] for p in deduped))
    speakers = sorted(set(
        p['participant'] for p in deduped
        if p['participant'].lower() not in NOISE_PARTICIPANTS
    ))

    stats = {
        "total_pairs":        len(deduped),
        "duplicates_removed": duplicates_removed,
        "dictionary_entries": len(gloss_dict),
        "avg_waimaa_tokens":  round(avg_waimaa_len, 2),
        "avg_english_tokens": round(avg_english_len, 2),
        "genres":             genres,
        "speakers":           speakers,
        "files":              file_stats,
    }

    # ── Write outputs ──
    corpus_path = out_dir / 'parallel_corpus.json'
    dict_path   = out_dir / 'gloss_dictionary.json'
    stats_path  = out_dir / 'corpus_stats.json'

    corpus_path.write_text(json.dumps(deduped,    ensure_ascii=False, indent=2), encoding='utf-8')
    dict_path.write_text(  json.dumps(gloss_dict, ensure_ascii=False, indent=2), encoding='utf-8')
    stats_path.write_text( json.dumps(stats,      ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n✓ parallel_corpus.json  — {len(deduped)} pairs")
    print(f"✓ gloss_dictionary.json — {len(gloss_dict)} entries")
    print(f"✓ corpus_stats.json     — summary")
    print(f"\n  Avg Waima'a length : {avg_waimaa_len:.1f} tokens")
    print(f"  Avg English length : {avg_english_len:.1f} tokens")
    print(f"  Genres             : {', '.join(genres)}")
    print(f"  Speakers           : {', '.join(speakers)}")


if __name__ == '__main__':
    main()