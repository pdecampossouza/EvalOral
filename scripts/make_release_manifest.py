#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or p.name=='SHA256SUMS.txt': continue
    h=hashlib.sha256(p.read_bytes()).hexdigest(); rows.append((p.relative_to(ROOT).as_posix(),p.stat().st_size,h))
(ROOT/'SHA256SUMS.txt').write_text('\n'.join(f'{h}  {path}' for path,_,h in rows)+'\n',encoding='utf-8')
summary={'files':len(rows),'bytes':sum(x[1] for x in rows)}
print(json.dumps(summary,indent=2))
