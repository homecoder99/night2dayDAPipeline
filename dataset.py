import json, shutil, os, glob, cv2
from pathlib import Path
from tqdm import tqdm

    # BDD100K 2D box 카테고리 중 사용할 것만 선정
CLASSES = [
    "person", "rider", "car", "truck", "bus", "train",
    "motor", "bike", "traffic light", "traffic sign"
]
cls2id = {c: i for i, c in enumerate(CLASSES)}

DRIVE_ROOT = "/Users/gimsang-yun/Downloads"        # 필요시 수정
DATA_DIR   = "/Users/gimsang-yun/Downloads"
OUTPUT_DIR = f"{DRIVE_ROOT}/bdd100k_yolo"    # 변환 결과 저장
OUTPUT_DIR_NIGHT = f"{DRIVE_ROOT}/bdd100k_yolo_night"    # 변환 결과 저장

def convert_split(split: str):
    img_src = Path(DATA_DIR, "100k-images", split)
    lbl_src = Path(DATA_DIR, "100k-labels", split)
    img_dst = Path(OUTPUT_DIR, "images", split)
    lbl_dst = Path(OUTPUT_DIR, "labels", split)
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    json_files = glob.glob(str(lbl_src / "*.json"))
    for js_path in tqdm(json_files, desc=f"{split}"):
        uuid = Path(js_path).stem
        jpg_src = img_src / f"{uuid}.jpg"
        if not jpg_src.exists():         # 일부 test 세트는 라벨이 없을 수 있음
            continue

        # ---------- 이미지 복사 ----------
        shutil.copy(jpg_src, img_dst / jpg_src.name)

        # ---------- 라벨 변환 ----------
        img_h, img_w = cv2.imread(str(jpg_src)).shape[:2]
        yolo_lines = []
        with open(js_path, "r") as f:
            data = json.load(f)
        for obj in data["frames"][0]["objects"]:
            if "box2d" not in obj:              # poly2d는 여기서는 제외
                continue
            cat = obj["category"]
            if cat not in cls2id:               # 미사용 클래스를 스킵
                continue
            box = obj["box2d"]
            # BDD100K box는 좌상(x1,y1) ↔ 우하(x2,y2)
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            # YOLO 형식: xc, yc, w, h (정규화)
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            yolo_lines.append(f"{cls2id[cat]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # test split은 라벨이 없을 수도 있으므로 빈 파일도 만듦
        with open(lbl_dst / f"{uuid}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

# ▶ 체크할 루트 지정
ROOT = Path("/Users/gimsang-yun/Downloads/bdd100k_yolo_night")   # ← 수정!

def count_files(root: Path, splits=("train", "val", "test"),
                subdirs=("images", "labels"), exts=(".jpg", ".txt")):
    for sub in subdirs:
        print(f"\n── {sub.upper()} ──")
        for sp in splits:
            dir_ = root / sub / sp
            if not dir_.exists():
                print(f"{sp:>6}:  ❌ 디렉터리 없음")
                continue
            n = sum(1 for _ in dir_.rglob("*") if _.suffix.lower() in exts)
            print(f"{sp:>6}:  {n:,}")

# count_files(ROOT)

def create_yolo_yaml():
    yaml_text = f"""
    path: {OUTPUT_DIR}
    train: images/train
    val:   images/val
    test:  images/test
    nc: {len(CLASSES)}
    names: {CLASSES}
    """
    with open(Path(OUTPUT_DIR, "bdd100k_yolo.yaml"), "w") as f:
        f.write(yaml_text)
    print(Path(OUTPUT_DIR, 'bdd100k_yolo.yaml').read_text())

# --- 경로 설정 ----------------------------------------------------------
ROOT_RAW   = "/Users/gimsang-yun/Downloads"     # json·jpg 원본
ROOT_NIGHT = "/Users/gimsang-yun/Downloads/bdd100k_yolo_night"   # 새롭게 만들 폴더
from pathlib import Path
import json, shutil, cv2, glob, os
from tqdm import tqdm

def convert_night(split):
    src_img = Path(ROOT_RAW, "100k-images", split)           # *.jpg
    src_lbl = Path(ROOT_RAW, "100k-labels", split)           # *.json
    dst_img = Path(ROOT_NIGHT, "images", split)
    dst_lbl = Path(ROOT_NIGHT, "labels", split)
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    json_files = glob.glob(str(src_lbl / "*.json"))
    for js in tqdm(json_files, desc=f"{split} night"):
        with open(js) as f:
            data = json.load(f)
            
        if data["attributes"].get("timeofday") != "night":
            continue                           # 낮/저녁은 패스
        uuid   = Path(js).stem
        jpg_in = src_img / f"{uuid}.jpg"
        if not jpg_in.exists():                # test split 등 라벨만 있을 때 대비
            continue

        # ── 이미지 복사 ─────────────────────────
        shutil.copy(jpg_in, dst_img / jpg_in.name)

        # ── YOLO 라벨 변환 ─────────────────────
        h, w = cv2.imread(str(jpg_in)).shape[:2]
        lines = []
        for obj in data["frames"][0]["objects"]:
            if "box2d" not in obj: continue
            cat = obj["category"]
            if cat not in cls2id:  continue
            b = obj["box2d"]
            xc = (b["x1"]+b["x2"])/(2*w);  yc = (b["y1"]+b["y2"])/(2*h)
            bw = (b["x2"]-b["x1"])/w;      bh = (b["y2"]-b["y1"])/h
            lines.append(f"{cls2id[cat]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(dst_lbl / f"{uuid}.txt","w") as f:
            f.write("\n".join(lines))

def create_yolo_night_yaml():
    yaml_text = f"""
    path: {OUTPUT_DIR_NIGHT}
    train: images/train
    val:   images/val
    test:  images/test
    nc: {len(CLASSES)}
    names: {CLASSES}
    """
    with open(Path(OUTPUT_DIR_NIGHT, "bdd100k_yolo_night.yaml"), "w") as f:
        f.write(yaml_text)
    print(Path(OUTPUT_DIR_NIGHT, 'bdd100k_yolo_night.yaml').read_text())

if __name__ == "__main__":

    # for split in ["train","val","test"]:
    #     convert_split(split)

    for split in ["train","val","test"]:
        convert_night(split)

    # create_yolo_yaml()
    # create_yolo_night_yaml()

