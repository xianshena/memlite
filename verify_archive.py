"""Verify manifest bytes only; never run model code."""
from pathlib import Path
import hashlib,json,sys,zipfile
def sha(b):return hashlib.sha256(b).hexdigest()
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
if p.is_dir():
    m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
    for r in m['files']:
        b=(p/r['path']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'],r['path']
else:
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        m=json.loads(z.read('manifest.json'))
        for r in m['files']:
            b=z.read(r['path']);assert len(b)==r['bytes'] and sha(b)==r['sha256'],r['path']
print('ALL_MANIFEST_FILES_VERIFIED',len(m['files']))
