#!/usr/bin/env python3
# Distillation utilities: response-based KD on the classification-head logits
# at the 3 detection scales. Chosen because teacher (YOLO26-x) and student
# (YOLO26-l) differ in backbone WIDTH -> feature channels don't match, but the
# class-head outputs share shape (nc classes, same grid at same imgsz). So no
# channel adapters are needed -- the most robust KD form for this pair.
import torch
import torch.nn.functional as F


def find_detect(model):
    for m in model.modules():
        if type(m).__name__ == "Detect":
            return m
    raise RuntimeError("No Detect module found")


class ClsLogitCapture:
    """Registers forward hooks on Detect.cls_head[i] to grab per-scale logits."""
    def __init__(self, model):
        det = find_detect(model)
        if not hasattr(det, "cls_head"):
            raise RuntimeError("Detect has no .cls_head -- check YOLO26 head attribute name")
        self.buf = []
        self.handles = []
        for m in det.cls_head:
            self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        self.buf.append(out)

    def clear(self):
        self.buf = []

    def remove(self):
        for h in self.handles:
            h.remove()


def kd_cls_loss(student_logits, teacher_logits):
    """MSE between student and teacher class logits, averaged over scales.
    Teacher detached. Returns 0-dim tensor. Shapes must match per scale."""
    assert len(student_logits) == len(teacher_logits) and len(student_logits) > 0, \
        f"scale count mismatch: {len(student_logits)} vs {len(teacher_logits)}"
    loss = 0.0
    for s, t in zip(student_logits, teacher_logits):
        assert s.shape == t.shape, f"shape mismatch {tuple(s.shape)} vs {tuple(t.shape)}"
        loss = loss + F.mse_loss(s.flatten(2), t.flatten(2).detach())
    return loss / len(student_logits)
