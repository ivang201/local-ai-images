"""Удаление фона локально: PNG с прозрачным фоном, в том числе по текстовому запросу.

Модели (всё на видеокарте, без интернета):
  * BiRefNet (MIT)             — автоматически находит главный объект и даёт точные края (волосы, мех).
  * SAM 3.1 (Meta, SAM License) — находит объекты по тексту: "cat", "red car", "логотип".

Режимы:
  rmbg.py photo.jpg                       -> главный объект, фон прозрачный
  rmbg.py photo.jpg -p "собака"           -> оставить только собак (все найденные)
  rmbg.py photo.jpg -p "person" --one     -> только самого уверенного человека
  rmbg.py photo.jpg -p "провода" --remove -> вырезать найденное, остальное оставить
  rmbg.py logo.png --crop --bg "#ffffff"  -> обрезать по объекту, залить фон цветом

Запускать через окружение:  <папка>\\rmbg\\.venv\\Scripts\\python.exe <папка>\\rmbg\\rmbg.py ...
или ярлыком:               <папка>\\rmbg\\rmbg.bat ...
Последняя строка вывода — JSON с путями к результатам.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # без лишних предупреждений в выводе
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # всё лежит локально, в интернет не ходим
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
BIREFNET_DIR = ROOT / "models" / "BiRefNet"
SAM_DIR = ROOT / "models" / "sam3.1"
OPUS_DIR = ROOT / "models" / "opus-mt-ru-en"  # перевод русских запросов для SAM
OLLAMA = "http://127.0.0.1:11434"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_birefnet = None
_sam = None


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- модели

def birefnet():
    global _birefnet
    if _birefnet is None:
        from transformers import AutoModelForImageSegmentation
        log("Загружаю BiRefNet…")
        m = AutoModelForImageSegmentation.from_pretrained(str(BIREFNET_DIR), trust_remote_code=True)
        _birefnet = m.to(DEVICE).eval().half() if DEVICE == "cuda" else m.eval()
    return _birefnet


def sam():
    global _sam
    if _sam is None:
        from transformers import Sam3Model, Sam3Processor
        log("Загружаю SAM 3.1…")
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        model = Sam3Model.from_pretrained(str(SAM_DIR), dtype=dtype).to(DEVICE).eval()
        _sam = (model, Sam3Processor.from_pretrained(str(SAM_DIR)), dtype)
    return _sam


# ---------------------------------------------------------------- маски

def fg_alpha(img: Image.Image, size=1024) -> torch.Tensor:
    """BiRefNet: мягкая альфа главного объекта, float [H,W] 0..1 на DEVICE."""
    m = birefnet()
    x = torch.from_numpy(np.array(img.convert("RGB").resize((size, size), Image.BICUBIC))).float()
    x = ((x.permute(2, 0, 1)[None] / 255.0) - MEAN) / STD
    x = x.to(DEVICE, next(m.parameters()).dtype)
    with torch.inference_mode():
        pred = m(x)[-1].sigmoid().float()
    return F.interpolate(pred, size=(img.height, img.width), mode="bilinear", align_corners=False)[0, 0]


def sam_masks(img: Image.Image, text: str, threshold: float):
    """SAM 3.1: список (score, bool-маска [H,W]) для объектов по тексту."""
    model, proc, dtype = sam()
    inputs = proc(images=img.convert("RGB"), text=text, return_tensors="pt").to(DEVICE)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    return sorted(zip(res["scores"].float().tolist(), res["masks"].to(DEVICE).bool()), key=lambda t: -t[0])


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask.float()
    return F.max_pool2d(mask.float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def feather(mask: torch.Tensor, r: float) -> torch.Tensor:
    if r <= 0:
        return mask.float()
    k = int(r * 3) * 2 + 1
    xs = torch.arange(k, device=mask.device) - k // 2
    g = torch.exp(-(xs.float() ** 2) / (2 * r * r))
    g = (g / g.sum()).view(1, 1, 1, k)
    m = mask.float()[None, None]
    m = F.conv2d(F.pad(m, (k // 2, k // 2, 0, 0), mode="replicate"), g)
    m = F.conv2d(F.pad(m, (0, 0, k // 2, k // 2), mode="replicate"), g.transpose(2, 3))
    return m[0, 0]


def refine_with_birefnet(img: Image.Image, hard: torch.Tensor) -> torch.Tensor:
    """Уточнить края маски SAM: BiRefNet на вырезке вокруг объекта, ограниченный расширенной маской."""
    H, W = hard.shape
    ys, xs = torch.nonzero(hard, as_tuple=True)
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    pad = int(0.08 * max(y1 - y0, x1 - x0)) + 8
    y0, y1, x0, x1 = max(0, y0 - pad), min(H, y1 + pad + 1), max(0, x0 - pad), min(W, x1 + pad + 1)
    crop_alpha = fg_alpha(img.crop((x0, y0, x1, y1)))
    soft = torch.zeros_like(hard, dtype=torch.float32)
    soft[y0:y1, x0:x1] = crop_alpha
    r = max(2, int(0.005 * max(H, W)))
    # BiRefNet уточняет только узкую полосу вокруг края SAM и не «утекает» на соседние объекты (руки и т.п.)
    limited = soft * feather(dilate(hard, r), r / 2)
    # Насколько BiRefNet согласен с SAM. Если нет (объект не «главный» на вырезке) — берём SAM.
    inter = (limited > 0.5).logical_and(hard).sum().item()
    union = (limited > 0.5).logical_or(hard).sum().item()
    iou = inter / max(1, union)
    if iou >= 0.75:
        return limited, iou
    return feather(hard, 1.0), iou


def clean_phrase(s: str) -> str:
    s = s.strip().strip('."\'').strip()
    return re.sub(r"^(the|a|an)\s+", "", s, flags=re.I).lower()


def to_english(text: str) -> str:
    """SAM понимает только английский. Кириллицу переводим локально: Opus-MT ru-en, запасной вариант — Ollama."""
    if not re.search(r"[А-Яа-яЁё]", text):
        return text
    if OPUS_DIR.exists():
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(str(OPUS_DIR))
        model = MarianMTModel.from_pretrained(str(OPUS_DIR)).eval()
        with torch.inference_mode():
            ids = model.generate(**tok([text], return_tensors="pt"), max_new_tokens=32)
        out = clean_phrase(tok.decode(ids[0], skip_special_tokens=True))
        log(f'Запрос переведён: "{text}" -> "{out}"')
        return out
    try:
        tags = json.load(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3))["models"]
        # самая лёгкая текстовая модель — быстрее всего загрузится ради перевода пары слов
        names = [m["name"] for m in sorted(tags, key=lambda m: m.get("size", 1 << 62)) if "embed" not in m["name"]]
        if not names:
            raise RuntimeError("нет моделей")
        body = {
            "model": names[0], "stream": False, "think": False, "keep_alive": 0,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": "Translate the user's text into a short English noun phrase for an "
                 "object detector (e.g. 'red car', 'person', 'logo'). Output only the phrase."},
                {"role": "user", "content": text},
            ],
        }
        req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"]
        out = clean_phrase(re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().splitlines()[0])
        log(f'Запрос переведён: "{text}" -> "{out}"')
        return out
    except Exception as e:
        raise SystemExit(f'Запрос "{text}" на русском, а перевести не удалось ({e}). '
                         f"Запустите Ollama или напишите запрос по-английски.")


# ---------------------------------------------------------------- обработка

def save_preview(res: Image.Image, path: Path) -> Path:
    """Результат на шахматке (как в редакторах) — чтобы глазами проверить прозрачность."""
    im = res.convert("RGBA")
    im.thumbnail((768, 768))
    s = 16
    yy, xx = np.mgrid[0:im.height, 0:im.width]
    board = np.where(((xx // s + yy // s) % 2)[..., None] == 1, [200, 204, 212, 255], [244, 245, 248, 255])
    bg = Image.fromarray(board.astype(np.uint8), "RGBA")
    bg.alpha_composite(im)
    bg.convert("RGB").save(path, quality=88)
    return path


def process(path: Path, out: Path, args) -> dict:
    img = Image.open(path)
    img.load()
    rgb = img.convert("RGB")
    t0 = time.time()
    info = {"input": str(path)}
    if args.prompt:
        text = to_english(args.prompt)
        found = sam_masks(rgb, text, args.threshold)
        info.update(prompt=text, found=len(found), scores=[round(s, 3) for s, _ in found])
        if not found:
            raise SystemExit(f'"{text}" не найдено на {path.name}. Попробуйте другое слово или --threshold 0.25.')
        chosen = found[:1] if args.one else found
        hard = torch.stack([m for _, m in chosen]).any(0)
        if args.remove:
            alpha = 1.0 - feather(dilate(hard, max(2, int(0.004 * max(hard.shape)))), 1.5)
        elif args.no_refine:
            alpha = feather(hard, 1.0)
        else:
            alpha, iou = refine_with_birefnet(rgb, hard)
            info["refined"] = iou >= 0.75
    else:
        alpha = fg_alpha(rgb)
    a = (alpha.clamp(0, 1) * 255).round().byte().cpu().numpy()
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:  # сохранить исходную прозрачность
        a = np.minimum(a, np.asarray(img.convert("RGBA"))[..., 3])

    res = rgb.convert("RGBA")
    res.putalpha(Image.fromarray(a))
    if args.crop:
        bbox = Image.fromarray(a).point(lambda v: 255 if v > 8 else 0).getbbox()
        if bbox:
            p = int(args.padding * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            bbox = (max(0, bbox[0] - p), max(0, bbox[1] - p), min(res.width, bbox[2] + p), min(res.height, bbox[3] + p))
            res = res.crop(bbox)
    if args.bg:
        bg = Image.new("RGBA", res.size, args.bg)
        bg.alpha_composite(res)
        res = bg.convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out)
    if args.preview:
        info["preview"] = str(save_preview(res, out.with_name(out.stem + "_preview.jpg")))
    info.update(output=str(out), size=list(res.size), seconds=round(time.time() - t0, 1),
                coverage=round(float((a > 127).mean()), 3))
    log(f"{path.name} -> {out}  ({info['seconds']} с)")
    return info


def main():
    for s in (sys.stdout, sys.stderr):  # русский текст без «кракозябр» в любой консоли
        s.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Локальное удаление фона (BiRefNet + SAM 3.1)")
    p.add_argument("inputs", nargs="+", help="картинки (jpg/png/webp…)")
    p.add_argument("-o", "--out", help="файл .png (для одной картинки) или папка; по умолчанию рядом, *_nobg.png")
    p.add_argument("-p", "--prompt", help='что оставить: "cat", "red car", "человек" (без -p — главный объект)')
    p.add_argument("--remove", action="store_true", help="наоборот: сделать прозрачным то, что найдено по -p")
    p.add_argument("--one", action="store_true", help="только один, самый уверенный объект по -p")
    p.add_argument("--threshold", type=float, default=0.4, help="уверенность SAM для -p (меньше — найдёт больше)")
    p.add_argument("--no-refine", action="store_true", help="не уточнять края маски SAM через BiRefNet")
    p.add_argument("--crop", action="store_true", help="обрезать картинку по объекту")
    p.add_argument("--padding", type=float, default=0.03, help="поля при --crop, доля размера объекта")
    p.add_argument("--bg", help='залить фон цветом вместо прозрачности, напр. "#ffffff"')
    p.add_argument("--preview", action="store_true",
                   help="ещё сохранить <имя>_preview.jpg — результат на шахматке для проверки глазами")
    args = p.parse_args()
    if args.remove and not args.prompt:
        p.error("--remove работает только вместе с -p")

    outs = []
    files = [Path(f).resolve() for f in args.inputs]
    for f in files:
        if not f.exists():
            p.error(f"нет файла {f}")
        if args.out:
            o = Path(args.out).resolve()
            dst = o if (o.suffix.lower() == ".png" and len(files) == 1) else o / f"{f.stem}_nobg.png"
        else:
            dst = f.with_name(f"{f.stem}_nobg.png")
        outs.append(process(f, dst, args))
    print(json.dumps({"results": outs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
