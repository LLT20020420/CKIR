"""Check release bytes and syntax using only the Python standard library."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text('utf-8'))
expected={entry['path'] for entry in manifest['files']}|{'SOURCE_MANIFEST.json'}
ignored={'.git','__pycache__','.ckir-tests'}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
        and not ignored.intersection(p.relative_to(root).parts) and p.suffix not in ('.pyc','.pyo')}
if actual!=expected:
    raise ValueError('file inventory differs from the fixed release manifest')
for entry in manifest['files']:
    p=root/entry['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError(f'file hash mismatch: {entry["path"]}')
    if p.stat().st_size!=entry['bytes']:
        raise ValueError(f'file length mismatch: {entry["path"]}')
    if p.suffix=='.py':compile(p.read_text('utf-8'),entry['path'],'exec')
print(json.dumps({'status':'PASS_FIXED_RELEASE_PACKAGE','version':manifest['version'],'files':len(manifest['files'])+1,
                  'training_performed':False,'model_accuracy_evaluated':False}))
