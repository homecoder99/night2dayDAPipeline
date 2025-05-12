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

def preprocess_and_cache():
    """ 밝기 보정 & cache 저장 """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = Path(yaml.safe_load(open(DATA_YAML))['path'] + '/' + yaml.safe_load(open(DATA_YAML))['train'])
    out_dir = CACHE_DIR/img_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(img_dir)
    for img_path in tqdm(list(img_dir.rglob('*.jpg')), desc='CLAHE/Retinex'):
        rel = img_path.relative_to(img_dir)
        out_path = out_dir/rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(out_path), clahe_retinex(img))
    # ── 새 data.yaml 작성 ──
    ds = yaml.safe_load(open(DATA_YAML))
    new_yaml = {
        'path' : str(DATA_ROOT),
        'train': {'images': 'images_aug/train', 'labels': 'labels/train'},
        'val'  : {'images': 'images/val',   'labels': 'labels/val'},
        'test' : {'images': 'images/test',  'labels': 'labels/test'},
        'nc'   : ds['nc'],
        'names': ds['names']
    }
    new_yaml_path = DATA_ROOT / 'night_cache.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.safe_dump(new_yaml, f)

    return new_yaml_path

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
