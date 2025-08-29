import csv
import json
import random
from pathlib import Path


def read_csv_rows(csv_path: Path):
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get('question') or '').strip()
            a = (row.get('answer') or '').strip()
            if not q or not a:
                continue
            yield {
                'question': q,
                'answer': a,
                'source': (row.get('source') or '').strip(),
                'focus_area': (row.get('focus_area') or '').strip(),
            }


def to_instruct_format(example):
    system = (
        "You are a helpful medical assistant. Answer carefully and concisely. "
        "If unsure, say you don't know."
    )
    instruction = example['question']
    # Optionally ground with metadata
    meta_bits = []
    if example.get('source'):
        meta_bits.append(f"Source: {example['source']}")
    if example.get('focus_area'):
        meta_bits.append(f"Focus: {example['focus_area']}")
    meta = ('\n' + '\n'.join(meta_bits)) if meta_bits else ''
    input_text = f"{instruction}{meta}"
    return {
        'system': system,
        'instruction': instruction,
        'input': input_text,
        'output': example['answer'],
    }


def main():
    root = Path(__file__).parent
    csv_path = root / 'medquad.csv'
    out_dir = root / 'data_prepared'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_csv_rows(csv_path))
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    split = max(1, int(0.95 * n))
    train_rows = rows[:split]
    val_rows = rows[split:]

    def write_jsonl(path: Path, items):
        with path.open('w', encoding='utf-8') as f:
            for ex in items:
                f.write(json.dumps(to_instruct_format(ex), ensure_ascii=False) + '\n')

    write_jsonl(out_dir / 'train.jsonl', train_rows)
    write_jsonl(out_dir / 'val.jsonl', val_rows)
    print(f"Wrote {len(train_rows)} train and {len(val_rows)} val examples to {out_dir}")


if __name__ == '__main__':
    main()


