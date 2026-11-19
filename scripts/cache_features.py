"""Encode all Flowers102 images once with the pinned upstream CLIP fork."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/HyCal-CIL'))
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader


class Images(Dataset):
    def __init__(self,records,transform):
        self.records,self.transform=records,transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self,i):
        with Image.open(self.records[i]['path']) as im:
            return self.transform(im)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=ROOT/'data/manifests/flowers102/records.json'
    records=json.loads(manifest.read_text(encoding='utf-8'))
    classes=json.loads((manifest.parent/'classes.json').read_text(encoding='utf-8'))
    classes=sorted(classes,key=lambda v:v['classname'])
    target=ROOT/'artifacts/features/flowers102/features.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise RuntimeError(f'Cache already exists: {target}; verify provenance before reusing')
    model,_,transform=clip.load(str(ROOT/'artifacts/models/ViT-B-16.pt'),device=device,jit=False)
    assert next(model.parameters()).dtype==torch.float32
    model.eval().requires_grad_(False)
    loader=DataLoader(Images(records,transform),batch_size=args.batch_size,
                      shuffle=False,num_workers=args.workers,pin_memory=device=='cuda')
    started=time.perf_counter()
    memory_start=__import__('psutil').Process().memory_info().rss
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    features=[]
    with torch.inference_mode():
        texts=[f"a photo of a {c['classname'].replace('_',' ')}." for c in classes]
        tokens=clip.tokenize(texts).to(device)
        text_features=model.encode_text(tokens)
        text_features=text_features/(text_features.norm(dim=-1,keepdim=True)+1e-10)
        for i,batch in enumerate(loader):
            emb=model.encode_image(batch.to(device,non_blocking=True))
            emb=emb/(emb.norm(dim=-1,keepdim=True)+1e-10)
            features.append(emb.cpu())
            n=min((i+1)*args.batch_size,len(records))
            if i==0 or n%256==0 or n==len(records):
                elapsed=time.perf_counter()-started
                print(f'encoded {n}/{len(records)} elapsed={elapsed:.1f}s rate={n/elapsed:.1f}/s',flush=True)
    image_features=torch.cat(features)
    assert image_features.shape==(len(records),512)
    assert torch.isfinite(image_features).all() and torch.isfinite(text_features).all()
    meta={'model':'ViT-B/16','upstream_commit':'4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41',
          'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'weight_sha256':'5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f',
          'preprocess':str(transform),'dtype':str(image_features.dtype),'device':device,
          'torch':torch.__version__,'batch_size':args.batch_size,'seconds':time.perf_counter()-started,
          'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
          'process_rss_end_bytes':__import__('psutil').Process().memory_info().rss,
          'process_rss_start_bytes':memory_start}
    payload={'image_features':image_features,'text_features':text_features.cpu(),
             'sample_ids':[r['sample_id'] for r in records],
             'labels':torch.tensor([r['label'] for r in records]),
             'splits':[r['split'] for r in records],'classes':classes,'metadata':meta}
    temporary=target.with_suffix('.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,target)
    (target.parent/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
