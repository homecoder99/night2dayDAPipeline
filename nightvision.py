#!/usr/bin/env python
"""
Night-only YOLOv8 pipeline
1. 라벨·이미지 검증 및 CLAHE/Retinex 전처리 (+cache)
2. anchor_auto 로 데이터셋 맞춤 anchor 재계산
3. custom hyp + albumentations 증강으로 학습
4. TTA + 이미지-레벨 보정 추론 & CSV 저장
"""

from pathlib import Path
import os, sys, csv, yaml, subprocess, shutil
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch

# ---------- 경로 ----------
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT/'data.yaml'
HYP_FILE = ROOT/'night_hyp.yaml'
DATA_ROOT = Path('/Users/gimsang-yun/Downloads/bdd100k_yolo_night')
CACHE_DIR = Path('/Users/gimsang-yun/Downloads/bdd100k_yolo_night/images_aug')       # 밝기 보정 이미지 캐시
RESULT_DIR = ROOT/'runs'/'night_yolo'
IMG_SIZE = 1280
BATCH = 1
EPOCHS = 2
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = 'yolov8l.pt'

# ---------- 0. 유틸 ----------
def clahe_retinex(img):
    """ 간단 CLAHE+Retinex 결합 """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # MSR – 3 σ
    img_float = img_clahe.astype(np.float32)+1
    scales = [15, 80, 250]
    retinex = np.sum([cv2.GaussianBlur(img_float, (0,0), s) for s in scales], axis=0)/len(scales)
    img_ret = np.log10(img_float) - np.log10(retinex+1e-6)
    img_ret = cv2.normalize(img_ret, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_ret

import torch, kornia as K
import cv2, numpy as np
from functools import partial

# ─────────────────────────────────────────────────────────────
# 1. GPU 버전 CLAHE + Retinex
# ─────────────────────────────────────────────────────────────
def clahe_retinex_gpu(bgr_8u: np.ndarray, device="cuda"):
    """
    Args  : BGR uint8 image (H, W, 3)
    Return: BGR uint8 image after CLAHE + MSR (PyTorch/Kornia GPU)
    """
    # ① OpenCV BGR → RGB, 0-1 정규화 → (1,3,H,W) 텐서
    rgb = cv2.cvtColor(bgr_8u, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.

    # ② CLAHE (clipLimit=2.0, tileGrid=(8,8))
    t = K.enhance.equalize_clahe(t, clip_limit=2.0)

    # ③ Multi-Scale Retinex (σ=15,80,250)
    t = K.enhance.retinex_multi_scale(t, sigmas=[15, 80, 250])

    # ④ 0-1 → uint8 BGR로 되돌리기
    t = (t.clamp(0, 1) * 255).byte().squeeze(0).permute(1,2,0).cpu().numpy()
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────
# 2. 전처리 캐시 함수 수정
# ─────────────────────────────────────────────────────────────
def preprocess_and_cache():
    """train 이미지를 CLAHE+Retinex(GPU)로 보정하여 images_aug/train 저장,
       동일 상대경로에 labels/train 복사 → 새 data.yaml 반환"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- data.yaml 정보 파싱 ---
    ds       = yaml.safe_load(open(DATA_YAML))
    root     = Path(ds['path'])
    train_cfg= ds['train']
    img_src  = root / (train_cfg['images'] if isinstance(train_cfg, dict) else train_cfg)
    lbl_src  = root / (train_cfg['labels'] if isinstance(train_cfg, dict) else 'labels/train')

    img_dst  = DATA_ROOT / 'images_aug' / 'train'
    lbl_dst  = DATA_ROOT / 'labels'      / 'train'
    img_dst.mkdir(parents=True, exist_ok=True)

    # --- CPU·GPU 함수 선택 ---
    fn = clahe_retinex_gpu if torch.cuda.is_available() else clahe_retinex
    desc = 'CLAHE/Retinex GPU' if torch.cuda.is_available() else 'CLAHE/Retinex CPU'

    # --- 이미지 보정 + 라벨 복사 ---
    for img_p in tqdm(list(img_src.rglob('*.jpg')), desc=desc):
        rel = img_p.relative_to(img_src)
        out_img = img_dst / rel
        out_img.parent.mkdir(parents=True, exist_ok=True)

        if not out_img.exists():
            img = cv2.imread(str(img_p))
            cv2.imwrite(str(out_img), fn(img))   # ← GPU or CPU

        # 라벨
        lbl_src_p = lbl_src / rel.with_suffix('.txt')
        lbl_dst_p = lbl_dst / rel.with_suffix('.txt')
        lbl_dst_p.parent.mkdir(parents=True, exist_ok=True)
        if lbl_src_p.exists() and not lbl_dst_p.exists():
            shutil.copy(lbl_src_p, lbl_dst_p)

    # --- 새 data.yaml 생성 ---
    new_yaml = {
        'path' : str(DATA_ROOT),
        'train': {'images': 'images_aug/train', 'labels': 'labels/train'},
        'val'  : ds['val'],
        'test' : ds.get('test',''),
        'nc'   : ds['nc'],
        'names': ds['names']
    }
    out_yaml = DATA_ROOT / 'night_cache.yaml'
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(new_yaml, f)

    return out_yaml

# ---------- 1. anchor 재계산 ----------
def calc_anchors(data_yaml):
    cmd = ['yolo', 'utils', 'anchor_auto', f'data={data_yaml}', '--imgsz', str(IMG_SIZE)]
    LOGGER.info('> Re-estimating anchors...')
    subprocess.run(cmd, check=True)

# ---------- 2. 학습 ----------
def train(data_yaml):
    model = YOLO(MODEL_WEIGHTS)
    # custom Albumentations (γ / fog / motion-blur)
    from albumentations import RandomGamma, RandomFog, MotionBlur, Compose

    def alb_transform(batch):
        aug = Compose([
            RandomGamma(gamma_limit=(40, 200), p=0.5),
            RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            MotionBlur(p=0.3)
        ], bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})

        imgs = []
        for im in batch['img']:
            im_np = (im.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
            im_aug = aug(image=im_np)['image']
            imgs.append( torch.from_numpy(im_aug.transpose(2,0,1)/255.0) )
        batch['img'] = torch.stack(imgs).to(im.device)
        return batch

    model.add_callback("on_preprocess_batch", alb_transform)

    # ---------- HYP 파라미터 로드 ----------
    from ultralytics.cfg import get_cfg
    ALLOWED_KEYS = set(vars(get_cfg()).keys())

    with open(HYP_FILE) as f:
        raw_hyp = yaml.safe_load(f)

    hyp_dict = {k: v for k, v in raw_hyp.items() if k in ALLOWED_KEYS}
    missing  = set(raw_hyp) - ALLOWED_KEYS
    if missing:
        LOGGER.warning(f"⚠️  Ignoring unsupported top-level keys: {sorted(missing)}")
    # ---------- 학습 ----------
    model.train(
        data=str(data_yaml),
        device=DEVICE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RESULT_DIR.parent),
        name=RESULT_DIR.name,
        **hyp_dict          # hsv, mosaic, lr0 … 전부 dict 안에 있음
    )
    best_ckpt = model.trainer.best           # e.g. runs/night_yolo/weights/best.pt
    return best_ckpt

# ---------- 3. 추론 (TTA + 이미지-보정 ensemble) ----------
def predict(best_ckpt, data_yaml):
    model = YOLO(best_ckpt)
    save_csv = RESULT_DIR / 'inference_tta.csv'

    # ── ① val 이미지 디렉터리 추출 ──
    cfg  = yaml.safe_load(open(data_yaml))
    base = Path(cfg['path'])
    val  = cfg['val']
    img_dir = base / (val['images'] if isinstance(val, dict) else val)

    imgs = list(img_dir.rglob('*.jpg'))

    with open(save_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['image', 'cls', 'conf', 'xyxy'])

        for im_path in tqdm(imgs, desc='TTA inference'):
            raw     = cv2.imread(str(im_path))
            bright  = clahe_retinex(raw)

            preds = []
            for img in (raw, bright):
                preds.append(
                    model.predict(
                        img,
                        device=DEVICE,
                        imgsz=IMG_SIZE,
                        conf=0.15,
                        iou=0.55,
                        augment=True,   # v8.3+ TTA 플래그
                        stream=False,
                        verbose=False
                    )[0]
                )

            # ── 두 결과 concat ──
            boxes  = np.vstack([p.boxes.xyxy.cpu().numpy() for p in preds])
            confs  = np.hstack([p.boxes.conf.cpu().numpy() for p in preds])
            clses  = np.hstack([p.boxes.cls.cpu().numpy()  for p in preds])

            # xyxy → xywh (OpenCV 형식)
            xywh          = boxes.copy()
            xywh[:, 2]   -= boxes[:, 0]      # w = x2 - x1
            xywh[:, 3]   -= boxes[:, 1]      # h = y2 - y1

            idxs = cv2.dnn.NMSBoxes(
                bboxes=xywh.tolist(),
                scores=confs.tolist(),
                score_threshold=0.01,
                nms_threshold=0.55
            )
            for i in np.array(idxs).ravel():           # 빈 tuple → 안전 처리
                wr.writerow([im_path.name,
                             int(clses[i]),
                             float(confs[i]),
                             boxes[i].tolist()])
                

# ---------- main ----------
if __name__ == '__main__':        
    cached_yaml = preprocess_and_cache()          # step-1
    ckpt = train(cached_yaml)                     # step-2
    # predict(ckpt, cached_yaml)                    # step-3
    LOGGER.info(f'Done! Results → {RESULT_DIR}')
