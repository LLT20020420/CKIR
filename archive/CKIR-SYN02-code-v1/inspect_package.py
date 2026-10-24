"""Verify frozen package bytes and syntax without importing scientific modules."""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for item in manifest['files']:
 p=root/item['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
 if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),item['path'],'exec')
entries=json.loads((root/'source_locations.json').read_text(encoding='utf-8'))
for e in entries:
 lines=(root/e['file']).read_text(encoding='utf-8').splitlines()
 assert e['symbol'].split('.')[-1] in lines[e['line_start']-1]
print(json.dumps({'status':'PASS_FIXED_SOURCE_PACKAGE','files':len(manifest['files']),'entrypoints':len(entries),'new_training':0,'new_predictions':0}))
