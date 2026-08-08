#!/usr/bin/env python3
# PHASE 1 -- train the YOLO26-x TEACHER on the clean split.
# Same recipe/imgsz as the baseline so the teacher is a fair, stronger model.
# Produces the weights the student will distill from in Phase 2.
import os, sys
_d = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_d,"ultralytics")) and not os.path.isfile(os.path.join(_d,"ultralytics","__init__.py")):
    sys.path=[p for p in sys.path if os.path.abspath(p)!=_d]
    sys.path=[p for p in sys.path if p not in ("",".")]
from ultralytics import YOLO

ROOT="/home/pavinder/plant_project"
PT=f"{ROOT}/yolo26x.pt"            # teacher weights (must exist; see note in chat)
DATA=f"{ROOT}/plantdataset_noleak/dataset.yaml"
PROJECT=f"{ROOT}/distill_runs"
IMGSZ=1024; BATCH=8; DEVICE=0; WORKERS=8; FREEZE_N=11

def log_per_class(m):
    b=m.box; nm=m.names if hasattr(m,"names") else {}
    print("\n=== Teacher Per-class AP ===")
    try:
        for j,ci in enumerate(list(b.ap_class_index)):
            name=nm.get(ci,str(ci)) if isinstance(nm,dict) else str(ci)
            print(f"{ci:>3}  {str(name)[:30]:30}  AP50={b.ap50[j]:.4f}  AP50-95={b.ap[j]:.4f}")
    except Exception as e: print("parse fail:",e)

print("="*70+"\nTEACHER STAGE 1 (frozen warmup, 10 ep)\n"+"="*70)
model=YOLO(PT)
model.train(data=DATA,epochs=10,imgsz=IMGSZ,batch=BATCH,device=DEVICE,workers=WORKERS,
    freeze=FREEZE_N,optimizer="SGD",lr0=5e-4,cos_lr=False,amp=False,
    mosaic=0.3,mixup=0.0,copy_paste=0.0,close_mosaic=0,
    project=PROJECT,name="teacher_stage1",exist_ok=True,verbose=True)
s1=f"{PROJECT}/teacher_stage1/weights/best.pt"

print("="*70+"\nTEACHER STAGE 2 (full fine-tune, 130 ep)\n"+"="*70)
model=YOLO(s1)
model.train(data=DATA,epochs=130,imgsz=IMGSZ,batch=BATCH,device=DEVICE,workers=WORKERS,
    freeze=0,optimizer="SGD",lr0=5e-3,cos_lr=True,amp=False,
    mosaic=0.8,mixup=0.15,copy_paste=0.1,close_mosaic=10,
    project=PROJECT,name="teacher_stage2",exist_ok=True,verbose=True)
s2=f"{PROJECT}/teacher_stage2/weights/best.pt"

print("="*70+"\nTEACHER FINAL VAL\n"+"="*70)
best=YOLO(s2)
m=best.val(data=DATA,imgsz=IMGSZ,batch=BATCH,device=DEVICE,verbose=True)
print(f"\n[TEACHER] mAP@50={m.box.map50:.4f}  mAP@50-95={m.box.map:.4f}")
log_per_class(m)
print(f"\nTeacher weights for distillation: {s2}")
