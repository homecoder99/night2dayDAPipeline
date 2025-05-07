# night2day_da_pipeline.py – Ultralytics >= 8.3.127 compatible
"""
End‑to‑end Night→Day augmentation (CycleGAN) + YOLOv8 domain‑adaptation training
pipeline, updated for Ultralytics v8.3.127 (or newer).

Major fixes ▸
  • Removed deprecated/unused Ultralytics imports
  • Patched YAML‑lookup logic (works with the new pkg layout)
  • Re‑named domain‑classifier attributes → self.domP3/5 and used in forward()
  • Re‑placed trainer‑only helpers with public YOLO APIs
  • Build‑dataloader now leverages YOLO.build_dataloader (still available ≥8.3.x)
  • Added small guards for CPU‑only execution
"""

from __future__ import annotations

import argparse
import importlib
import importlib.resources as pkg
import os
import random
import tempfile
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

# -----------------------------
# ◼︎  Section 1: CycleGAN  ◼︎
# -----------------------------


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):  # type: ignore[override]
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2 ** n_down
        model += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_c, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_c: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw, padw = 4, 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_c, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


class ImageBuffer:  # History buffer (CycleGAN paper)
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.images: list[torch.Tensor] = []

    def push(self, data: torch.Tensor):
        res = []
        for img in data:
            img = img.unsqueeze(0)
            if len(self.images) < self.max_size:
                self.images.append(img)
                res.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    replaced = self.images[idx].clone()
                    self.images[idx] = img
                    res.append(replaced)
                else:
                    res.append(img)
        return torch.cat(res, 0)


# ───────── ImageDataset 수정 ─────────
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.files_A = sorted(Path(root, f"{mode}A").glob("*.jpg"))
        self.files_B = sorted(Path(root, f"{mode}B").glob("*.jpg"))

        # ⬇ 기본 transform: PIL → Tensor[0‒1]
        self.transform = transform or transforms.Compose([
            transforms.Resize(286, interpolation=Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),           # <─ 중요
        ])

    def __getitem__(self, idx):
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[idx % len(self.files_B)]).convert("RGB")
        return {
            "A": self.transform(img_A),   # Tensor[C,H,W]
            "B": self.transform(img_B),
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):  # type: ignore[override]
        return self.loss(pred, target)


# —— CycleGAN training ————————————————————————————————————————————

def train_cyclegan(data_dir: str, epochs: int = 100, batch_size: int = 1, lr: float = 2e-4, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    G_A2B, G_B2A = ResnetGenerator().to(device), ResnetGenerator().to(device)
    D_A, D_B = NLayerDiscriminator().to(device), NLayerDiscriminator().to(device)
    criterion_gan, criterion_cycle = GANLoss().to(device), nn.L1Loss().to(device)
    opt_G = optim.Adam((*G_A2B.parameters(), *G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D_A, opt_D_B = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999)), optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = lambda e: 1.0 - max(0, e - epochs // 2) / (epochs // 2)
    sched_G, sched_D_A, sched_D_B = (optim.lr_scheduler.LambdaLR(opt, sched) for opt in (opt_G, opt_D_A, opt_D_B))

    tf = transforms.Compose([transforms.Resize(286), transforms.RandomCrop(256), transforms.RandomHorizontalFlip()])
    dl = DataLoader(
            ImageDataset(data_dir, mode="train"),   # transform 내장
            batch_size=1,
            shuffle=True,
            num_workers=4,    # macOS MPS: 먼저 0으로 테스트
            pin_memory=False)
    buf_A, buf_B = ImageBuffer(), ImageBuffer()

    for epoch in range(epochs):
        for i, batch in enumerate(dl):
            real_A, real_B = batch["A"].to(device), batch["B"].to(device)
            # —— Generators ——
            opt_G.zero_grad()
            idt_A, idt_B = G_B2A(real_B), G_A2B(real_A)
            loss_idt = (criterion_cycle(idt_A, real_B) + criterion_cycle(idt_B, real_A)) * 5
            fake_B, fake_A = G_A2B(real_A), G_B2A(real_B)
            loss_gan = (
                criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
                criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            )
            rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
            loss_cycle = criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)
            loss_G = loss_idt + loss_gan + 10 * loss_cycle
            loss_G.backward(); opt_G.step()
            # —— Discriminators ——
            for D, real, fake, buf, opt in (
                (D_A, real_A, fake_A, buf_A, opt_D_A),
                (D_B, real_B, fake_B, buf_B, opt_D_B),
            ):
                opt.zero_grad()
                loss_D = (
                    criterion_gan(D(real), torch.ones_like(D(real))) +
                    criterion_gan(D(buf.push(fake.detach())), torch.zeros_like(D(real)))
                ) * 0.5
                loss_D.backward(); opt.step()
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dl)}] G:{loss_G.item():.3f} D_A:{loss_D.item():.3f}")
        sched_G.step(); sched_D_A.step(); sched_D_B.step()
        if (epoch + 1) % 5 == 0:
            save_image(fake_B[:4] * 0.5 + 0.5, f"runs/cyclegan/fake_B_e{epoch+1}.png", nrow=2)
            torch.save(G_A2B.state_dict(), f"runs/cyclegan/G_A_e{epoch+1}.ckpt")
            torch.save(G_B2A.state_dict(), f"runs/cyclegan/G_B_e{epoch+1}.ckpt")


# —— Image augmentation with the trained generator —————————————————

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch, os
from pathlib import Path

def augment_with_generator(
        generator_ckpt: str,
        input_dir: str,
        output_dir: str,
        device: str = "cuda"
):
    # ---------------- device ----------------
    device = torch.device("mps" if (device == "cuda" and torch.backends.mps.is_available()) else device)

    # ---------------- generator -------------
    gen = ResnetGenerator().to(device)
    gen.load_state_dict(torch.load(generator_ckpt, map_location=device))
    gen.eval()

    # ---------------- I/O & transform -------
    os.makedirs(output_dir, exist_ok=True)
    resize   = transforms.Resize((256, 256))   # PIL → PIL
    to_tensor = transforms.ToTensor()          # PIL → FloatTensor [0,1]

    for p in Path(input_dir).glob("*.jpg"):
        # ① 이미지 읽기(PIL) ② 크기 조정 ③ 텐서화
        img = to_tensor(resize(Image.open(p).convert("RGB"))).unsqueeze(0).to(device)

        with torch.no_grad():
            out = gen(img)[0].cpu()            # [C,H,W]  range(-1,1)

        # -1~1 → 0~1 로 스케일 후 저장
        save_image(out * 0.5 + 0.5, Path(output_dir, p.name))

    print(f"✅ saved augmented images to {output_dir}")

# ----------------------------------------------------------
# ◼︎  Section 2: YOLOv8 Domain‑Adaptation ◼︎
# ----------------------------------------------------------

from ultralytics import YOLO  # after pip install ultralytics>=8.3.127
import yaml

# — Utility: copy+patch base YAML to the system temp dir ———————————

def _fallback_yaml_search(base_yaml: str) -> Path | None:
    """Return a plausible Path for *base_yaml* inside Ultralytics, or None."""
    # 1) new structure `ultralytics/cfg/models/v8/…`
    try:
        with pkg.path("ultralytics.cfg.models.v8", base_yaml) as p:
            return Path(p)
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    # 2) bundled but not exposed as resource (older builds)
    try:
        res_root = importlib.resources.files("ultralytics")
        p = res_root.joinpath(f"cfg/models/v8/{base_yaml}")
        return p if p.is_file() else None
    except Exception:  # pragma: no cover
        return None


def custom_cfg(base_yaml: str = "yolov8n.yaml", nc: int = 8) -> str:
    """Locate *base_yaml*, patch `nc`, write to TMP, and return the new path."""
    # explicit path provided by caller
    p = Path(base_yaml)
    if not p.is_file():
        p = _fallback_yaml_search(base_yaml) or Path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Could not locate '{base_yaml}' inside the Ultralytics package. "
            "Make sure Ultralytics ≥ 8.3.127 is installed correctly or pass an "
            "explicit path to a YOLOv8 YAML file."
        )
    out = Path(tempfile.gettempdir()) / f"{p.stem}_{nc}cls.yaml"
    if not out.exists():
        cfg = yaml.safe_load(p.read_text())
        cfg["nc"] = nc
        out.write_text(yaml.safe_dump(cfg))
    return str(out)


# — Domain‑classifier helpers ————————————————————————————————


class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, α): ctx.α = α; return x
    @staticmethod
    def backward(ctx, g):   return -g * ctx.α, None
def grad_reverse(x, α=1): return GRLayer.apply(x, α)

class DomainClassifier(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, 256), nn.ReLU(True),
            nn.Linear(256, 2))
    def forward(self, x, α): return self.net(grad_reverse(x, α))


# — Custom YOLOv8 model with domain heads —————————————————————————


class YOLOv8_DA(YOLO):
    def __init__(self, num_classes=8, base_yaml="yolov8n.yaml"):
        # YAML 구성 파일을 기반으로 DetectionModel 생성
        super().__init__(model=custom_cfg(base_yaml, num_classes),
                         task="detect", verbose=False)

        # 손실 함수와 도메인 분류기 placeholder
        self.model.domain_loss = nn.CrossEntropyLoss()
        self.model.domP3 = None
        self.model.domP5 = None

    def _lazy_init(self, p3, p5):
        device = p3.device
        if self.model.domP3 is None:
            self.model.domP3 = DomainClassifier(p3.shape[1]).to(device)
            self.model.domP5 = DomainClassifier(p5.shape[1]).to(device)

    def forward(self, x, dom=None, α=0.1, **kw):
        p3, p4, p5 = self.model.model[:-1](x)              # backbone + neck
        self._lazy_init(p3, p5)                            # 최초 1회 초기화
        det = self.model.model[-1]((p3, p4, p5))           # head

        if self.training:
            d3 = self.model.domP3(p3, α)
            d5 = self.model.domP5(p5, α)
            return det, d3, d5
        return det

# — Training loop ——————————————————————————————————————————————

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
import torch, torch.nn as nn
from pathlib import Path

def train_yolov8_da(data_yaml, model_yaml, epochs=60, batch=8, device="cuda"):
    model = YOLOv8_DA(num_classes=10, base_yaml=model_yaml)

    trainer = DetectionTrainer(overrides=dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        workers=4,
    ))

    # 커스텀 모델 주입
    trainer.model = model.model  # <-- 여기만 사용
    # 커스텀 속성 주입
    trainer.model.domain_loss = model.model.domain_loss
    trainer.model.domP3 = model.model.domP3
    trainer.model.domP5 = model.model.domP5

    trainer.train()

# —————————————————————————————————————————————— CLI ————————————

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train_gan", "augment", "train_yolo"])
    p.add_argument("--data_dir")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--generator_ckpt")
    p.add_argument("--input_dir")
    p.add_argument("--output_dir")
    p.add_argument("--data_yaml")
    p.add_argument("--model_yaml", default="yolov8n.yaml")  # 추가됨
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.set_float32_matmul_precision("medium")

    if args.mode == "train_gan":
        assert args.data_dir, "--data_dir required for CycleGAN training"
        train_cyclegan(args.data_dir, args.epochs, device=args.device)
    elif args.mode == "augment":
        assert all((args.generator_ckpt, args.input_dir, args.output_dir))
        augment_with_generator(args.generator_ckpt, args.input_dir, args.output_dir, device=args.device)
    else:  # train_yolo
        assert args.data_yaml and args.model_yaml
        train_yolov8_da(args.data_yaml, args.model_yaml, args.epochs, device=args.device)
