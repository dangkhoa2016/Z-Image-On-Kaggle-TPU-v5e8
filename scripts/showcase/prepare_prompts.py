import json
from pathlib import Path

base = Path(__file__).resolve().parents[2]
root = base / 'evidence/showcase'
root.mkdir(parents=True, exist_ok=True)
items = json.loads((Path(__file__).with_name('creative-prompts.json')).read_text())
assert len(items) == 8
assert [item['seed'] for item in items] == [501, 502, 503, 504, 615, 506, 507, 508]
assert len({item['prompt'] for item in items}) == 8
(root / 'prompts.json').write_text(json.dumps(items[:6], indent=2, ensure_ascii=False) + '\n')
(root / 'creative-extra-prompts.json').write_text(json.dumps(items[6:], indent=2, ensure_ascii=False) + '\n')
print('SIX_UNIQUE_PROMPTS', len({item['prompt'] for item in items[:6]}))
print('IDS', [item['id'] for item in items[:6]])
