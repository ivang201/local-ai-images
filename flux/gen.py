"""Генерация картинок из командной строки (для Claude и скриптов).

Модель: FLUX 3 Image Klein (stable-diffusion.cpp, CUDA) + uncensored Qwen3-VL энкодер,
короткий запрос расширяется апсемплером в Ollama. Всё локально, на видеокарте NVIDIA от 8 ГБ.

Примеры:
  python <папка>\\flux\\gen.py "минималистичный логотип кофейни, лиса и чашка"
  python <папка>\\flux\\gen.py "лес в тумане" --aspect 16:9 -o C:\\proj\\img\\hero.png
  python <папка>\\flux\\gen.py "иконка ракеты" -W 512 -H 512 -n 4
  python <папка>\\flux\\gen.py "Full-color photograph, ..." --no-upsample --seed 7
  python <папка>\\flux\\gen.py "перенеси этого человека в Париж" --ref C:\\photos\\me.jpg

В конце печатает строку JSON с путями к картинкам (последняя строка вывода).
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
import app as klein  # noqa: E402  (тот же код, что у веб-страницы)


def snap(v):
    return max(256, int(round(v / 16)) * 16)


def size_from(args):
    if args.width or args.height:
        w = args.width or args.height
        h = args.height or args.width
        return snap(w), snap(h)
    if args.ref and not args.aspect:
        return klein.size_like(args.ref[0], args.megapixels)
    a, b = (float(x) for x in (args.aspect or "1:1").split(":"))
    mp = args.megapixels * 1024 * 1024
    w = (mp * a / b) ** 0.5
    return snap(w), snap(w * b / a)


def main():
    p = argparse.ArgumentParser(description="Локальная генерация картинок FLUX 3 Klein")
    p.add_argument("prompt", help="что нарисовать (любой язык)")
    p.add_argument("-o", "--out", help="файл .png или папка (по умолчанию папка flux\\outputs)")
    p.add_argument("-W", "--width", type=int, help="ширина в px (кратно 16)")
    p.add_argument("-H", "--height", type=int, help="высота в px (кратно 16)")
    p.add_argument("--aspect", help="соотношение сторон, напр. 16:9, 9:16, 3:2 (по умолчанию 1:1, "
                                    "а с --ref — как у первого фото)")
    p.add_argument("-r", "--ref", action="append", default=[],
                   help="фото-образец для редактирования (JPG/PNG), можно до 3 раз: "
                        "\"перенеси человека с фото в Париж\", \"замени фон\"")
    p.add_argument("--megapixels", type=float, default=1.0,
                   help="площадь для --aspect в мегапикселях (1.0 ≈ 1024x1024; до ~2.0 на 8 ГБ VRAM)")
    p.add_argument("-n", "--count", type=int, default=1, help="сколько вариантов (разные seed)")
    p.add_argument("--seed", type=int, help="seed (по умолчанию случайный)")
    p.add_argument("--steps", type=int, default=16, help="шаги (модель обучена на 16)")
    p.add_argument("--no-upsample", action="store_true",
                   help="не расширять промпт (если уже даёте подробное английское описание)")
    args = p.parse_args()
    refs = [Path(r).resolve() for r in args.ref]
    for r in refs:
        if not r.exists():
            p.error(f"нет файла {r}")
    if len(refs) > klein.MAX_REFS:
        p.error(f"не больше {klein.MAX_REFS} фото (на 8 ГБ видеопамяти)")

    w, h = size_from(args)
    if w * h > 2.3 * 1024 * 1024:
        print(f"Внимание: {w}x{h} больше ~2 МП, может не хватить видеопамяти или выйти хуже.", file=sys.stderr)

    caption = args.prompt
    if not args.no_upsample:
        print("Расширяю промпт через Ollama…", file=sys.stderr)
        caption = klein.upsample(args.prompt, w, h, refs=refs)
        print("Промпт для модели:\n" + caption + "\n", file=sys.stderr)

    out_arg = Path(args.out) if args.out else None
    results = []
    seed0 = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    for i in range(args.count):
        seed = seed0 + i
        t0 = time.time()
        print(f"[{i + 1}/{args.count}] Рисую {w}x{h}, seed {seed}…", file=sys.stderr)
        tmp = klein.OUTPUTS / (time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{seed}.png")
        klein.render(caption, w, h, seed, tmp, steps=args.steps, refs=refs)
        tmp.with_suffix(".txt").write_text(
            f"Запрос: {args.prompt}\nSeed: {seed}\nРазмер: {w}x{h}\n\nПромпт для модели:\n{caption}\n",
            encoding="utf-8")
        final = tmp
        if out_arg:
            if out_arg.suffix.lower() == ".png":
                final = out_arg if args.count == 1 else out_arg.with_name(f"{out_arg.stem}_{i + 1}.png")
            else:
                final = out_arg / tmp.name
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp, final)
        print(f"   готово за {time.time() - t0:.0f} с: {final}", file=sys.stderr)
        results.append(str(final))

    print(json.dumps({"images": results, "width": w, "height": h, "seed": seed0, "caption": caption},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
