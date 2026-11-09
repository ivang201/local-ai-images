<!-- local-ai-images:start -->
## Pictures: I CAN make them locally — never say "I can't draw"

This PC has two local tools (NVIDIA GPU, offline). Use them whenever a task needs an image: logo, icon,
sticker, illustration, hero/banner, product shot, avatar, texture, mockup, placeholder art, or edits to
the user's own photos. Nothing is uploaded anywhere.

| Tool | What it does | Command |
|---|---|---|
| **gen.py** (FLUX 3 Klein) | text → image; photo + instruction → edited photo | `python "{{ROOT}}/flux/gen.py" "<prompt>" -o "<out.png>" [opts]` |
| **rmbg** (BiRefNet + SAM 3.1) | image → transparent PNG; cut out / erase things by text | `"{{ROOT}}/rmbg/.venv/Scripts/python.exe" "{{ROOT}}/rmbg/rmbg.py" "<img>" [opts]` |

- **Always quote paths and use forward slashes** — in Bash an unquoted `C:\x\gen.py` becomes `C:xgen.py`.
  The commands above work unchanged in both Bash and PowerShell.
- Both print progress to stderr and a JSON line as the **last stdout line** — parse paths from it.
  Run them in the background or with a 400000 ms timeout.
- Do the whole job myself (generate → cut out → put the file where it belongs); don't hand
  the user commands to run.

### Recipes
(`gen.py …` / `rmbg.py …` below are shorthand for the full quoted commands from the table.)

1. **Logo / icon / sticker with a transparent background** (FLUX can't output transparency):
   ```
   python "{{ROOT}}/flux/gen.py" "flat vector logo of <idea>, text 'BRAND', on a plain white background" -o "C:/proj/logo_raw.png"
   "{{ROOT}}/rmbg/.venv/Scripts/python.exe" "{{ROOT}}/rmbg/rmbg.py" "C:/proj/logo_raw.png" -o "C:/proj/logo.png" --crop
   ```
2. **Picture for a site / post / banner**: `gen.py "<scene>" --aspect 16:9 -o hero.png` (or `-W 1344 -H 768`).
3. **Edit the user's photo** ("move me to Paris", "replace the background with a beach", "make it anime"):
   `gen.py "<instruction>" --ref "C:/path/photo.jpg" -o edited.png` (up to 3 `--ref`; keeps the photo's aspect).
4. **Remove the background of an existing image** (product, person, pet): `rmbg.py photo.jpg`
   → `photo_nobg.png` next to it.
5. **Keep only one thing**: `rmbg.py img -p "red car"` (Russian is fine: `-p "кот"`); add `--one` for just the
   most confident match. **Make one thing transparent**: `rmbg.py img -p "wires" --remove`.
6. **Erase an object and fill the gap with background** (not transparency): `gen.py "remove the wires"
   --ref img.png` — that's an edit, rmbg only cuts.

### gen.py details
- Options: `-o file.png|dir`, `-n N` variants (seeds +1), `--seed N`, `-W/-H` (multiples of 16) or
  `--aspect 3:2 --megapixels 1.0`, `--ref photo` (edit mode), `--no-upsample` (if the prompt is already a detailed
  English caption; otherwise the Ollama upsampler expands short/Russian prompts — keep it on).
- Time on an 8 GB card: ~35–45 s per 1024² image, ~60–80 s with `--ref`; first run is slower. Max ~2 MP.
- Text in images: English only, in quotes. Cyrillic renders as Latin look-alikes ("РАМЕН" → "PAMEH").
- Edit mode sometimes "ghosts" the old background through the new one → rerun with another `--seed`.

### rmbg details
- Options: `-o out.png|dir`, `-p "what"`, `--remove`, `--one`, `--threshold 0.25` (finds more), `--no-refine`
  (if refined edges pull in neighbours), `--crop [--padding 0.03]`, `--bg "#ffffff"` (fill instead of transparent),
  `--preview` (optional: also writes `<out>_preview.jpg` on a checkerboard).
- ~10–15 s per image incl. model load; several inputs in one call are faster. JSON has `found`/`scores`:
  `found: 0` means the prompt matched nothing — try a simpler English noun.

### Rules
- One GPU job at a time: never run gen.py and rmbg in parallel, and they share the GPU with the user's web UI.
  If a run fails with an out-of-memory / "cannot make enough memory available" error, ask whether the web UI
  is generating something, wait, retry.
- Save results into the user's project (not into the generator's outputs folder) unless they just want to look.
- Web UI for the user: `{{ROOT}}/Запустить генератор.bat` → http://127.0.0.1:7860 (generate, edit with
  photos, «Убрать фон» button).
<!-- local-ai-images:end -->
