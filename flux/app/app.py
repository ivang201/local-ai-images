"""Локальный генератор картинок: FLUX 3 Image Klein через stable-diffusion.cpp.

Открывает страницу в браузере: пишете текст -> получаете картинку.
Короткий запрос (на любом языке) сначала расширяет «апсемплер» промптов в Ollama,
потом sd-cli рисует картинку на видеокарте.
"""

import base64
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # <папка>\flux
APP = ROOT / "app"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
SD_CLI = ROOT / "sd" / "sd-cli.exe"

DIFFUSION_MODEL = MODELS / "FLUX-3-Image-Klein-AD-Q5_K.gguf"
TEXT_ENCODER = MODELS / "Qwen3-VL-4B-Instruct-Uncensored-AD-Q6_K.gguf"
VAE = MODELS / "ae.safetensors"

OLLAMA = "http://127.0.0.1:11434"
UPSAMPLER = "hf.co/AtomicChat/FLUX-3-Image-Klein-pup-Uncensored-GGUF:IQ4_XS"

PORT = 7860
SIZES = {"square": (1024, 1024), "portrait": (768, 1344), "landscape": (1344, 768)}

OUTPUTS.mkdir(exist_ok=True)
_PROMPTS = json.loads((APP / "system_prompts.json").read_text(encoding="utf-8"))
T2I_SYSTEM, I2I_SYSTEM = _PROMPTS["t2i"], _PROMPTS["i2i"]
REFS = OUTPUTS / "refs"
REFS.mkdir(exist_ok=True)
MAX_REFS = 3  # модель берёт до 10, но на 8 ГБ VRAM каждое фото заметно замедляет работу
REF_BUDGET_MP = 0.56  # суммарная площадь фото-образцов в мегапикселях (0.56 ≈ 768x768)
REF_MIN_MP = 0.25  # меньше — лица начинают «плыть»


def image_size(path):
    """(ширина, высота) для PNG/JPEG без Pillow; None, если формат не распознан."""
    data = Path(path).read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def size_like(path, megapixels=1.0):
    """Размер холста с пропорциями фото, ~1 МП, кратно 16."""
    wh = image_size(path)
    if not wh:
        return 1024, 1024
    w0, h0 = wh
    k = (megapixels * 1024 * 1024 / (w0 * h0)) ** 0.5
    return max(256, round(w0 * k / 16) * 16), max(256, round(h0 * k / 16) * 16)

lock = threading.Lock()
job = {"state": "idle"}  # idle | running | done | error


def set_job(**kw):
    job.update(kw)


def ollama_alive():
    try:
        urllib.request.urlopen(OLLAMA + "/api/version", timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama():
    if ollama_alive():
        return
    subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(30):
        time.sleep(1)
        if ollama_alive():
            return
    raise RuntimeError("Ollama не запустилась. Откройте приложение Ollama и попробуйте снова.")


def upsample(prompt, w, h, refs=()):
    """Короткий запрос -> подробный английский промпт. refs: пути к фото для режима редактирования."""
    ensure_ollama()
    if refs:
        user = {"role": "user", "content": prompt,
                "images": [base64.b64encode(Path(p).read_bytes()).decode() for p in refs]}
        system, penalty = I2I_SYSTEM, 0.0  # настройки BFL для редактирования
    else:
        g = math.gcd(w, h)
        user = {"role": "user", "content": f"TARGET CANVAS: {w // g}:{h // g}\n{prompt}"}
        system, penalty = T2I_SYSTEM, 1.5
    body = {
        "model": UPSAMPLER,
        "messages": [{"role": "system", "content": system}, user],
        "think": False,
        "stream": False,
        "keep_alive": 0,  # сразу выгрузить из видеопамяти, она нужна для рисования
        "options": {"temperature": 0, "presence_penalty": penalty, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.loads(r.read())["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text or prompt


def render(caption, w, h, seed, out_path, steps=16, refs=()):
    # Каждое фото-образец добавляет модели столько же работы, сколько сама картинка.
    # На 8 ГБ VRAM уменьшаем образцы (общий бюджет REF_BUDGET_MP), иначе не хватает памяти.
    ref_args = []
    per_ref = max(REF_MIN_MP, REF_BUDGET_MP / max(1, len(refs)))
    for i, p in enumerate(refs):
        ref_args += ["-r", str(p)]
        if image_size(p):
            rw, rh = size_like(p, per_ref)
            ref_args += ["--image-preprocess",
                         f"target=ref,index={i},mode=crop-resize,width={rw},height={rh}"]
    sampling_started = False
    cmd = [
        str(SD_CLI),
        "--diffusion-model", str(DIFFUSION_MODEL),
        "--llm", str(TEXT_ENCODER),
        "--vae", str(VAE),
        *ref_args,
        "-p", caption,
        "--steps", str(steps), "--cfg-scale", "1.0", "--sampling-method", "euler",
        "-W", str(w), "-H", str(h), "-s", str(seed),
        "--offload-to-cpu", "--vae-tiling",
        "-o", str(out_path),
    ]
    log_path = OUTPUTS / "last_sd_log.txt"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    buf, tail = b"", []
    with open(log_path, "wb") as log:
        while True:
            ch = proc.stdout.read(256)
            if not ch:
                break
            log.write(ch)
            buf += ch
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                line = p.decode("utf-8", "replace")
                if line.strip():
                    tail = (tail + [line])[-30:]
                m = re.search(r"\|\s*(\d+)/(\d+)", line)
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                    if "#" in line:  # прогресс-бар загрузки весов
                        set_job(stage="Загружаю модели…", progress=step / total)
                    elif total == steps:  # шаги сэмплирования
                        sampling_started = True
                        set_job(stage=f"Рисую: шаг {step} из {total}", progress=step / total)
                    elif sampling_started:  # тайлы VAE при декодировании
                        set_job(stage="Проявляю картинку…", progress=step / total)
                    else:  # тайлы VAE при кодировании фото-образцов
                        set_job(stage="Изучаю фото…", progress=step / total)
    proc.wait()
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError("sd-cli завершился с ошибкой:\n" + "\n".join(tail[-12:]))


def save_refs(images):
    """dataURL-ы со страницы -> файлы в outputs/refs."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = []
    for i, url in enumerate(images[:MAX_REFS]):
        head, _, b64 = url.partition(",")
        ext = ".png" if "image/png" in head else ".jpg"
        p = REFS / f"{stamp}_{i + 1}{ext}"
        p.write_bytes(base64.b64decode(b64))
        paths.append(p)
    return paths


def worker(prompt, size, use_upsampler, seed, refs):
    try:
        if refs and size == "auto":
            w, h = size_like(refs[0])
        else:
            w, h = SIZES.get(size, SIZES["square"])
        t0 = time.time()
        caption = prompt
        if use_upsampler:
            set_job(stage="Смотрю на фото и улучшаю промпт…" if refs else "Улучшаю промпт…", progress=0)
            caption = upsample(prompt, w, h, refs=refs)
        set_job(stage="Загружаю модели…", progress=0, caption=caption)
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{seed}.png"
        out = OUTPUTS / name
        render(caption, w, h, seed, out, refs=refs)
        ref_note = "".join(f"\nФото: {p}" for p in refs)
        out.with_suffix(".txt").write_text(
            f"Запрос: {prompt}\nSeed: {seed}\nРазмер: {w}x{h}{ref_note}\n\nПромпт для модели:\n{caption}\n",
            encoding="utf-8",
        )
        set_job(state="done", stage="Готово", progress=1, image="/outputs/" + name,
                caption=caption, seed=seed, seconds=round(time.time() - t0))
    except Exception as e:
        set_job(state="error", stage="Ошибка", error=str(e))
    finally:
        lock.release()


RMBG_PY = ROOT.parent / "rmbg" / ".venv" / "Scripts" / "python.exe"
RMBG_SCRIPT = ROOT.parent / "rmbg" / "rmbg.py"


def rmbg_worker(src, prompt, remove):
    """Убрать фон у картинки src (путь) через <папка>\\rmbg (BiRefNet + SAM 3.1)."""
    try:
        t0 = time.time()
        set_job(stage="Убираю фон…" if not prompt else f"Ищу «{prompt}» и вырезаю…", progress=0.3)
        out = OUTPUTS / f"{src.stem}_nobg_{datetime.now().strftime('%H%M%S')}.png"
        cmd = [str(RMBG_PY), str(RMBG_SCRIPT), str(src), "-o", str(out)]
        if prompt:
            cmd += ["-p", prompt] + (["--remove"] if remove else [])
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0 or not out.exists():
            msg = (r.stderr or r.stdout).strip().splitlines()
            raise RuntimeError("\n".join(msg[-6:]) or "rmbg завершился с ошибкой")
        set_job(state="done", stage="Готово", progress=1, image="/outputs/" + out.name,
                seconds=round(time.time() - t0), transparent=True)
    except Exception as e:
        set_job(state="error", stage="Ошибка", error=str(e))
    finally:
        lock.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, (APP / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self.send(200, job)
        if self.path == "/api/history":
            files = sorted(OUTPUTS.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)[:24]
            return self.send(200, ["/outputs/" + f.name for f in files])
        if self.path.startswith("/outputs/"):
            f = (OUTPUTS / Path(self.path[9:]).name)
            if f.suffix == ".png" and f.exists():
                return self.send(200, f.read_bytes(), "image/png")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/rmbg":
            return self.post_rmbg()
        if self.path != "/api/generate":
            return self.send(404, {"error": "not found"})
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return self.send(400, {"error": "Напишите, что нарисовать."})
        if not lock.acquire(blocking=False):
            return self.send(409, {"error": "Уже рисую предыдущую картинку, подождите."})
        try:
            seed = int(data.get("seed")) if str(data.get("seed", "")).strip() else random.randint(1, 2**31 - 1)
        except ValueError:
            seed = random.randint(1, 2**31 - 1)
        try:
            refs = save_refs(data.get("images") or [])
        except Exception:
            lock.release()
            return self.send(400, {"error": "Не удалось прочитать фото. Попробуйте JPG или PNG."})
        job.clear()
        set_job(state="running", stage="Начинаю…", progress=0, prompt=prompt)
        threading.Thread(
            target=worker,
            args=(prompt, data.get("size", "square"), bool(data.get("upsample", True)), seed, refs),
            daemon=True,
        ).start()
        self.send(200, {"ok": True})

    def post_rmbg(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not RMBG_PY.exists():
            return self.send(500, {"error": "Не установлено удаление фона (папка rmbg). Запустите установщик ещё раз."})
        src = data.get("src") or ""
        if src.startswith("/outputs/"):
            path = OUTPUTS / Path(src[9:]).name
        elif src.startswith("data:image"):
            path = save_refs([src])[0]
        else:
            return self.send(400, {"error": "Сначала выберите картинку."})
        if not path.exists():
            return self.send(404, {"error": "Картинка не найдена."})
        if not lock.acquire(blocking=False):
            return self.send(409, {"error": "Видеокарта занята предыдущей задачей, подождите."})
        job.clear()
        set_job(state="running", stage="Начинаю…", progress=0)
        prompt = (data.get("prompt") or "").strip()
        threading.Thread(target=rmbg_worker, args=(path, prompt, bool(data.get("remove"))), daemon=True).start()
        self.send(200, {"ok": True})


def main():
    missing = [p for p in (SD_CLI, DIFFUSION_MODEL, TEXT_ENCODER, VAE) if not p.exists()]
    if missing:
        print("Не хватает файлов:\n  " + "\n  ".join(map(str, missing)))
        input("Нажмите Enter для выхода")
        return
    url = f"http://127.0.0.1:{PORT}"
    # На Windows без этого второй запуск молча занимает тот же порт, и запросы уходят в старый процесс.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"Генератор уже запущен в другом окне — открываю {url}")
        webbrowser.open(url)
        time.sleep(3)
        return
    print(f"Генератор картинок запущен: {url}\nНе закрывайте это окно, пока пользуетесь генератором.")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
