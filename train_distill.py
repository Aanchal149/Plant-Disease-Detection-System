#!/usr/bin/env python3
# PHASE 2 -- distill the trained YOLO26-x teacher into a YOLO26-l student.
# Student loss = detection_loss + ALPHA * kd_cls_loss(student_logits, teacher_logits).
# Implemented by subclassing DetectionTrainer and wrapping the model's loss.
import os, sys
_d = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_d,"ultralytics")) and not os.path.isfile(os.path.join(_d,"ultralytics","__init__.py")):
    sys.path=[p for p in sys.path if os.path.abspath(p)!=_d]
    sys.path=[p for p in sys.path if p not in ("",".")]
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
sys.path.append(_d)
from distill_utils import ClsLogitCapture, kd_cls_loss

ROOT="/home/pavinder/plant_project"
STUDENT_PT=f"{ROOT}/yolo26l.pt"
TEACHER_PT=f"{ROOT}/distill_runs/teacher_stage2/weights/best.pt"
DATA=f"{ROOT}/plantdataset_noleak/dataset.yaml"
PROJECT=f"{ROOT}/distill_runs"
IMGSZ=1024; BATCH=8; DEVICE=0; WORKERS=8
ALPHA=float(os.environ.get("KD_ALPHA","1.0"))   # KD weight; tune if needed

class DistillTrainer(DetectionTrainer):
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # load frozen teacher on same device
        dev = next(self.model.parameters()).device
        tck = torch.load(TEACHER_PT, map_location=dev, weights_only=False)
        self.teacher = (tck["model"] if isinstance(tck,dict) and "model" in tck else tck).float()
        self.teacher.to(dev).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        # hooks to capture cls logits from both nets
        self.s_cap = ClsLogitCapture(self.model)
        self.t_cap = ClsLogitCapture(self.teacher)
        # wrap the student model's loss to add the KD term
        _orig_loss = self.model.loss
        trainer = self
        def loss_with_kd(batch, preds=None):
            trainer.s_cap.clear()
            det_loss, det_items = _orig_loss(batch, preds)   # runs student fwd -> fills s_cap
            with torch.no_grad():
                trainer.t_cap.clear()
                img = batch["img"] if isinstance(batch, dict) else batch
                trainer.teacher(img)                          # fills t_cap
            kd = kd_cls_loss(trainer.s_cap.buf, trainer.t_cap.buf)
            total = det_loss + ALPHA * kd
            # append kd to loss_items so it prints in the table
            det_items = torch.cat([det_items, kd.detach().reshape(1)])
            return total, det_items
        self.model.loss = loss_with_kd
        print(f"[distill] teacher loaded, KD wired, ALPHA={ALPHA}")

if __name__ == "__main__":
    print("="*70+"\nDISTILL: YOLO26-x -> YOLO26-l\n"+"="*70)
    model = YOLO(STUDENT_PT)
    model.train(trainer=DistillTrainer,
        data=DATA,epochs=130,imgsz=IMGSZ,batch=BATCH,device=DEVICE,workers=WORKERS,
        optimizer="SGD",lr0=5e-3,cos_lr=True,amp=False,
        mosaic=0.8,mixup=0.15,copy_paste=0.1,close_mosaic=10,
        project=PROJECT,name="student_distill",exist_ok=True,verbose=True)
    best=YOLO(f"{PROJECT}/student_distill/weights/best.pt")
    m=best.val(data=DATA,imgsz=IMGSZ,batch=BATCH,device=DEVICE,verbose=True)
    print(f"\n[STUDENT-DISTILL] mAP@50={m.box.map50:.4f}  mAP@50-95={m.box.map:.4f}")
    print("Compare against baseline 0.5735.")
