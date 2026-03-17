import os
import json
import pickle
from collections import defaultdict
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import logging
import time

def coco_to_xyxy(bbox_coco, width, height):
    # Keep unchanged
    x, y, w, h = bbox_coco
    if w <= 0 or h <= 0: return None
    x1=max(0, float(x)); y1=max(0, float(y))
    x2=min(float(width), float(x) + float(w)); y2=min(float(height), float(y) + float(h))
    # Return as integers or floats? IoU usually uses floats. Keep as floats for precision.
    return [x1, y1, x2, y2]

class RefDataset:

    def __init__(self, annotation_path: str, split: str, refs_filename: str, image_dir: str, num_samples: Optional[int] = None):
        logging.info(f"Loading RefCOCO dataset annotations from: {annotation_path}")
        logging.info(f"Split: {split}")
        logging.info(f"Using refs file: {refs_filename}")
        self.annotation_path = annotation_path
        self.split = split
        self.refs_filename = refs_filename
        self.image_dir = image_dir
        self.num_samples = num_samples
        self.instances_file = os.path.join(annotation_path, 'instances.json')
        self.refs_file = os.path.join(annotation_path, self.refs_filename)
        if not os.path.exists(self.instances_file): raise FileNotFoundError(f"COCO instances file not found: {self.instances_file}")
        if not os.path.exists(self.refs_file): raise FileNotFoundError(f"Refs file not found: {self.refs_file}")
        logging.info("Loading instances (COCO annotations)..."); st = time.time()
        with open(self.instances_file, 'r') as f: instances_data = json.load(f)
        logging.info(f"Instances loaded in {time.time() - st:.2f}s.")
        self.raw_images = instances_data.get('images', []); self.raw_annotations = instances_data.get('annotations', [])
        self.raw_categories = instances_data.get('categories', []); del instances_data
        logging.info(f"Loading referring expressions ({self.refs_filename})..."); st = time.time()
        with open(self.refs_file, 'rb') as f: self.raw_refs = pickle.load(f)
        logging.info(f"Refs loaded in {time.time() - st:.2f}s.")
        logging.info("Creating indices..."); st = time.time()
        self.create_indices()
        logging.info(f"Indices created in {time.time() - st:.2f}s.")
        if hasattr(self, 'raw_images'): del self.raw_images;
        if hasattr(self, 'raw_annotations'): del self.raw_annotations
        if hasattr(self, 'raw_categories'): del self.raw_categories;
        if hasattr(self, 'raw_refs'): del self.raw_refs
        logging.info(f"Filtering samples for split '{self.split}'..."); st = time.time()
        self._create_and_filter_samples()
        logging.info(f"Found {len(self.samples)} samples for split '{self.split}' in {time.time() - st:.2f}s.")

    def create_indices(self): # Condensed version
        self.Imgs={i['id']: i for i in self.raw_images}; self.Anns={}; self.imgToAnns=defaultdict(list)
        for a in self.raw_annotations: self.Anns[a['id']]=a; self.imgToAnns[a.get('image_id',-1)].append(a)
        self.Cats={c['id']: c['name'] for c in self.raw_categories}; self.catNameToId={c['name']: c['id'] for c in self.raw_categories}
        self.Refs={}; self.Sents={}; self.imgToRefs=defaultdict(list); self.refToAnn={}; self.annToRef=defaultdict(list)
        self.catToRefs=defaultdict(list); self.sentToRef={}; self.sentToTokens={}
        ref_id_dup=set(); sent_id_dup=set()
        for r in self.raw_refs:
            rid,aid,imid,cid=r.get('ref_id'),r.get('ann_id'),r.get('image_id'),r.get('category_id')
            if any(i is None for i in [rid,aid,imid,cid]): continue
            if rid in self.Refs and rid not in ref_id_dup: ref_id_dup.add(rid) # Track dups once
            self.Refs[rid]=r; self.imgToRefs[imid].append(r); self.catToRefs[cid].append(r)
            if aid in self.Anns: self.refToAnn[rid]=self.Anns[aid]; self.annToRef[aid].append(r)
            if 'sentences' in r and isinstance(r['sentences'], list):
                for s in r['sentences']:
                    sid=s.get('sent_id');
                    if sid is None: continue
                    if sid in self.Sents and sid not in sent_id_dup: sent_id_dup.add(sid)
                    self.Sents[sid]=s; self.sentToRef[sid]=r
                    if 'tokens' in s: self.sentToTokens[sid]=s['tokens']

    def _create_and_filter_samples(self): # Condensed version
        self.samples=[]; ref_ids=set()
        for rid,r in self.Refs.items():
            if r.get('split')==self.split:
                ref_ids.add(rid)
                sents=r.get('sentences',[])
                if isinstance(sents,list):
                    for i,_ in enumerate(sents): self.samples.append((rid,i))
        if not self.samples: logging.info(f"W: No samples for split '{self.split}' in '{self.refs_filename}'.")
        self.ref_ids_for_split=list(ref_ids)
        if self.num_samples is not None and len(self.samples) > self.num_samples:
            logging.info(f"Filtering to {self.num_samples} samples from {len(self.samples)}")
            self.samples = self.samples[:self.num_samples]
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, index: int): # Condensed version
        if not(0<=index<len(self.samples)): raise IndexError(f"Index {index} OOB ({len(self.samples)})")
        rid,sidx=self.samples[index]
        try:
            r=self.Refs[rid]
            ann=self.refToAnn[rid]
            imid=r['image_id']
            img=self.Imgs[imid]
            cid=ann['category_id']
            cname=self.Cats[cid]
            sinfo=r['sentences'][sidx]
            fname=img['file_name']

            if not fname:
                raise ValueError(f"Img {imid} missing file_name")
            subdir=""
            if 'train2014' in fname: subdir='train2014'
            elif 'val2014' in fname: subdir='val2014'
            if not subdir:
                ipath=os.path.join(self.image_dir, fname)
            else:
                ipath=os.path.join(self.image_dir, subdir, fname)            
            query = sinfo.get('sent', sinfo.get('raw', ''))

            return {
                'id':index,
                'query':query,
                'answer':coco_to_xyxy(ann.get('bbox'), img['width'], img['height']),
                'image': Image.open(ipath).convert('RGB') if os.path.exists(ipath) else None,
                'image_path':ipath,

                'sample_id':index,
                'ref_id':rid,
                'ann_id':ann.get('id'),
                'image_id':imid,
                'category_id':cid,
                'category_name':cname,
                'sentence':sinfo,
                'bbox':ann.get('bbox'),
                'iscrowd':ann.get('iscrowd'), 
                'image_info':img,
                'split':r.get('split'),
                }
        except KeyError as e: raise KeyError(f"Data inconsistency key {e} sample {index} ref {rid}")
        except IndexError: raise IndexError(f"Sent index {sidx} OOB ref {rid}")
        except Exception as e: raise RuntimeError(f"Error get item {index}: {e}") from e
