# Установщик «под ключ»: генерация картинок (FLUX 3 Image Klein) + удаление фона (BiRefNet + SAM 3.1).
# Запуск: двойной клик по «Установить.bat» (или: powershell -ExecutionPolicy Bypass -File install.ps1)
# Повторный запуск безопасен: уже скачанное пропускается, оборванные загрузки докачиваются.

param(
    # Для какого ИИ-помощника записать инструкции: ask (спросить в меню), claude, codex, both, none
    [ValidateSet('ask', 'claude', 'codex', 'both', 'none')]
    [string]$Agent = 'ask',
    [switch]$Yes,               # не задавать вопросов; без явного -Agent файлы помощников НЕ меняются
    [switch]$OnlyInstructions,  # ничего не скачивать — только записать инструкции для помощника
    [switch]$SkipRmbg           # не ставить удаление фона (экономит ~9 ГБ)
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root  = $PSScriptRoot
$Flux  = Join-Path $Root 'flux'
$Rmbg  = Join-Path $Root 'rmbg'

# --- Точные версии, на которых всё проверено ---------------------------------------------------
$SdTag   = 'master-1060-0b3b98d'
$SdZip   = 'sd-master-0b3b98d-bin-win-cuda12-x64.zip'
$CudaZip = 'cudart-sd-bin-win-cu12-x64.zip'
$HF = @{
    Klein = 'AtomicChat/FLUX-3-Image-Klein-GGUF@b88bc03edf40e24846eaf7270b2ba9c2a9a1637f'
    TE    = 'AtomicChat/FLUX-3-Image-Klein-TE-Uncensored-GGUF@84746e03e397cb042bd264c0cb9e40db9398c001'
    Pup   = 'AtomicChat/FLUX-3-Image-Klein-pup-Uncensored-GGUF@8671d09367bc71d99a1c5e359575eb35b151156f'
    Sam   = 'unsloth/sam3.1@cd3538905f98152ba184d2791fc5aa5d06e2ae45'
    Bi    = 'ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
    Opus  = 'Helsinki-NLP/opus-mt-ru-en@fbd6dc73284f95536648512cc21d57f19191961a'
}
$Upsampler = 'hf.co/AtomicChat/FLUX-3-Image-Klein-pup-Uncensored-GGUF:IQ4_XS'

# --- Помощники ------------------------------------------------------------------------------------
function Step($t) { Write-Host ''; Write-Host "==> $t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "    OK  $t" -ForegroundColor Green }
function Warn($t) { Write-Host "    !!  $t" -ForegroundColor Yellow }
function Fail($t) { Write-Host ''; Write-Host "ОШИБКА: $t" -ForegroundColor Red; exit 1 }

function Get-File($url, $dst) {
    # Скачать с докачкой; файл появляется под своим именем только после полной загрузки.
    if (Test-Path $dst) { Ok "$(Split-Path $dst -Leaf) уже есть"; return }
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    $part = "$dst.part"
    Write-Host "    ... $(Split-Path $dst -Leaf)"
    for ($i = 1; $i -le 5; $i++) {
        & curl.exe -fL --retry 5 --retry-delay 5 -C - --progress-bar -o $part $url
        if ($LASTEXITCODE -eq 0) { Move-Item $part $dst -Force; Ok (Split-Path $dst -Leaf); return }
        Warn "загрузка прервалась (код $LASTEXITCODE), попытка $i из 5"
        Start-Sleep 10
    }
    Fail "не удалось скачать $url. Проверьте интернет (или VPN/прокси) и запустите установщик ещё раз — он докачает."
}

function Get-HF($spec, $file, $dstDir, $dstName = $null) {
    $repo, $sha = $spec -split '@'
    if (-not $dstName) { $dstName = Split-Path $file -Leaf }
    Get-File "https://huggingface.co/$repo/resolve/$sha/$file" (Join-Path $dstDir $dstName)
}

function Find-Exe($name, $extra) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in $extra) { if (Test-Path $p) { return $p } }
    return $null
}

function Get-AgentTargets($choice) {
    $codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
    $t = @()
    if ($choice -in 'claude', 'both') { $t += [pscustomobject]@{ Name = 'Claude Code'; File = Join-Path $env:USERPROFILE '.claude\CLAUDE.md' } }
    if ($choice -in 'codex', 'both')  { $t += [pscustomobject]@{ Name = 'Codex'; File = Join-Path $codexHome 'AGENTS.md' } }
    return $t
}

function Select-Agent {
    if ($Agent -ne 'ask') { return $Agent }
    if ($Yes) { return 'none' }   # молча чужие файлы не трогаем
    Write-Host ''
    Write-Host 'Научить ИИ-помощника самому рисовать и вырезать фон?' -ForegroundColor White
    Write-Host '  В его файл инструкций допишется отдельный блок; ваши записи в файле не меняются, копия сохранится как .bak'
    $t = Get-AgentTargets 'both'
    Write-Host "  1 — Claude Code   ($($t[0].File))"
    Write-Host "  2 — Codex         ($($t[1].File))"
    Write-Host '  3 — оба'
    Write-Host '  4 — не нужно'
    while ($true) {
        switch ((Read-Host 'Ваш выбор [1-4]').Trim()) {
            '1' { return 'claude' } '2' { return 'codex' } '3' { return 'both' } '4' { return 'none' }
            default { Write-Host '    введите цифру от 1 до 4' }
        }
    }
}

function Write-Instructions($target, $block) {
    # Меняется только наш блок между метками (или он дописывается в конец). Остальной текст — как был.
    $file = $target.File
    New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
    $start = '<!-- local-ai-images:start -->'; $end = '<!-- local-ai-images:end -->'
    $bom = $false; $old = ''
    if (Test-Path $file) {
        $bytes = [IO.File]::ReadAllBytes($file)
        $bom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
        $old = [IO.File]::ReadAllText($file)
    }
    $nl = if ($old.Contains("`r`n") -or -not $old.Contains("`n")) { "`r`n" } else { "`n" }  # стиль переносов как у файла
    $blk = ($block -replace "`r`n", "`n") -replace "`n", $nl
    $pattern = [regex]::Escape($start) + '[\s\S]*?' + [regex]::Escape($end)
    if ($old -match $pattern) {
        $new = [regex]::Replace($old, $pattern, [Text.RegularExpressions.MatchEvaluator]{ param($m) $blk })
        $how = 'блок обновлён'
    } elseif ($old.Trim()) {
        $sep = if ($old.EndsWith("`n")) { $nl } else { $nl + $nl }
        $new = $old + $sep + $blk + $nl
        $how = 'блок дописан в конец'
    } else {
        $new = $blk + $nl
        $how = 'файл создан'
    }
    if ($new -ceq $old) { Ok "$($target.Name): уже актуально ($file)"; return }
    if (Test-Path $file) { Copy-Item $file "$file.bak-$(Get-Date -Format yyyyMMdd-HHmmss)" }
    [IO.File]::WriteAllText($file, $new, (New-Object Text.UTF8Encoding $bom))
    Ok "$($target.Name): $how ($file)"
}

Write-Host 'Установка локальной генерации картинок и удаления фона' -ForegroundColor White
Write-Host "Папка: $Root"
$AgentChoice = Select-Agent

if ($OnlyInstructions) { Write-Host 'Режим -OnlyInstructions: ничего не скачиваю, только инструкции для помощника.' } else {

# --- 1. Видеокарта и место ----------------------------------------------------------------------
Step '1/7  Проверяю видеокарту и место на диске'
$smi = Find-Exe 'nvidia-smi' @("$env:SystemRoot\System32\nvidia-smi.exe")
if (-not $smi) { Fail 'не найдена видеокарта NVIDIA (nvidia-smi). Нужна NVIDIA с 8 ГБ памяти и свежим драйвером.' }
$gpu = (& $smi --query-gpu=name,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(',')
$vram = [int]$gpu[1].Trim()
Ok "$($gpu[0].Trim()), $vram МБ видеопамяти"
if ($vram -lt 7900) { Warn 'меньше 8 ГБ видеопамяти: генерация может не поместиться или будет очень медленной.' }
$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB)
$needGB = if ($SkipRmbg) { 18 } else { 27 }
if ($freeGB -lt $needGB) { Fail "на диске $($drive.Name): свободно $freeGB ГБ, нужно около $needGB ГБ." }
Ok "свободно $freeGB ГБ (нужно около $needGB)"

# --- 2. Python и Ollama ---------------------------------------------------------------------------
Step '2/7  Проверяю Python и Ollama'
$pyLauncher = Find-Exe 'py' @("$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe", "$env:SystemRoot\py.exe")
$pyVer = $null
if ($pyLauncher) {
    foreach ($v in '3.12', '3.11', '3.10', '3.13') {
        $ErrorActionPreference = 'Continue'   # py пишет «не найдено» в stderr — это не ошибка установки
        & $pyLauncher "-$v" -c 'import sys' 2>$null | Out-Null
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -eq 0) { $pyVer = $v; break }
    }
}
if (-not $pyVer) {
    Warn 'подходящего Python (3.10–3.13) нет — ставлю Python 3.12 через winget'
    & winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    $pyLauncher = Find-Exe 'py' @("$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe", "$env:SystemRoot\py.exe")
    if (-not $pyLauncher) { Fail 'Python не установился. Поставьте Python 3.12 с python.org (галочка «Add to PATH») и запустите снова.' }
    $pyVer = '3.12'
}
$Py = @($pyLauncher, "-$pyVer")
Ok "Python $pyVer"

$ollama = Find-Exe 'ollama' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")
if (-not $ollama) {
    Warn 'Ollama не найдена — ставлю через winget'
    & winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
    $ollama = Find-Exe 'ollama' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")
    if (-not $ollama) { Fail 'Ollama не установилась. Скачайте её с ollama.com и запустите установщик снова.' }
}
Ok 'Ollama'

# --- 3. stable-diffusion.cpp ------------------------------------------------------------------------
Step '3/7  stable-diffusion.cpp (программа, которая рисует на видеокарте)'
$sdDir = Join-Path $Flux 'sd'
if (Test-Path (Join-Path $sdDir 'sd-cli.exe')) { Ok 'sd-cli уже установлен' } else {
    $base = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$SdTag"
    Get-File "$base/$SdZip" (Join-Path $sdDir 'sd-cuda.zip')
    Get-File "$base/$CudaZip" (Join-Path $sdDir 'cudart.zip')
    Expand-Archive (Join-Path $sdDir 'sd-cuda.zip') $sdDir -Force
    Expand-Archive (Join-Path $sdDir 'cudart.zip') $sdDir -Force
    Remove-Item (Join-Path $sdDir '*.zip')
    Ok 'sd-cli распакован'
}

# --- 4. Модели для картинок ----------------------------------------------------------------------
Step '4/7  Модели FLUX 3 Image Klein (~8.5 ГБ)'
$fm = Join-Path $Flux 'models'
Get-HF $HF.Klein 'FLUX-3-Image-Klein-AD-Q5_K.gguf' $fm
Get-HF $HF.TE 'Qwen3-VL-4B-Instruct-Uncensored-AD-Q6_K.gguf' $fm
Get-HF $HF.Klein 'vae/ae.safetensors' $fm
Get-HF $HF.Pup 'system_prompts.json' (Join-Path $Flux 'app')
New-Item -ItemType Directory -Force (Join-Path $Flux 'outputs') | Out-Null

# --- 5. Расширитель промптов в Ollama ------------------------------------------------------------
Step '5/7  Расширитель промптов в Ollama (~6.4 ГБ)'
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 3 | Out-Null }
catch {
    Start-Process $ollama 'serve' -WindowStyle Hidden
    Start-Sleep 5
}
& $ollama pull $Upsampler
if ($LASTEXITCODE -ne 0) { Fail 'ollama pull не удался. Проверьте интернет и запустите установщик снова.' }
Ok 'модель загружена в Ollama'

# --- 6. Удаление фона ----------------------------------------------------------------------------
if ($SkipRmbg) { Step '6/7  Удаление фона — пропущено (-SkipRmbg)' } else {
    Step '6/7  Удаление фона: PyTorch + модели (~9 ГБ, самый долгий шаг)'
    $venvPy = Join-Path $Rmbg '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { & $Py[0] $Py[1] -m venv (Join-Path $Rmbg '.venv') }
    & $venvPy -m pip install -q --upgrade pip
    & $venvPy -m pip install -q --only-binary ':all:' torch torchvision --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { Fail 'не удалось установить PyTorch.' }
    & $venvPy -m pip install -q --only-binary ':all:' transformers timm einops kornia pillow numpy safetensors accelerate sentencepiece sacremoses
    if ($LASTEXITCODE -ne 0) { Fail 'не удалось установить библиотеки Python.' }
    Ok 'Python-окружение готово'

    $rm = Join-Path $Rmbg 'models'
    foreach ($f in 'config.json', 'merges.txt', 'model.safetensors', 'processor_config.json', 'special_tokens_map.json',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'LICENSE') {
        Get-HF $HF.Sam $f (Join-Path $rm 'sam3.1')
    }
    foreach ($f in 'BiRefNet_config.py', 'birefnet.py', 'config.json', 'model.safetensors') {
        Get-HF $HF.Bi $f (Join-Path $rm 'BiRefNet')
    }
    foreach ($f in 'config.json', 'generation_config.json', 'pytorch_model.bin', 'source.spm', 'target.spm',
                   'tokenizer_config.json', 'vocab.json') {
        Get-HF $HF.Opus $f (Join-Path $rm 'opus-mt-ru-en')
    }
    & $venvPy -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'; print('    OK  PyTorch видит', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { Fail 'PyTorch не видит видеокарту. Обновите драйвер NVIDIA и запустите снова.' }
}

}  # конец блока загрузок (пропускается с -OnlyInstructions)

# --- 7. Инструкции для ИИ-помощника --------------------------------------------------------------
Step '7/7  Инструкции для ИИ-помощника (Claude Code / Codex)'
$rootFwd = ($Root -replace '\\', '/')
$tpl = [IO.File]::ReadAllText((Join-Path $Root 'agents\instructions.md'))
$block = $tpl.Replace('{{ROOT}}', $rootFwd).Trim()
$generated = Join-Path $Root 'agents\instructions.generated.md'
[IO.File]::WriteAllText($generated, $block + "`r`n", (New-Object Text.UTF8Encoding $false))
Ok "текст с вашими путями (для ручной вставки): $generated"
$targets = Get-AgentTargets $AgentChoice
if (-not $targets) { Warn 'файлы помощников не трогаю (выбрано «не нужно»)' }
foreach ($t in $targets) { Write-Instructions $t $block }

Write-Host ''
Write-Host 'Готово!' -ForegroundColor Green
Write-Host '  * Генератор в браузере: двойной клик по «Запустить генератор.bat»'
Write-Host "  * Из командной строки:  python `"$rootFwd/flux/gen.py`" `"рыжий кот на подоконнике`""
if (-not $SkipRmbg) { Write-Host "  * Убрать фон:           $rootFwd/rmbg/rmbg.bat картинка.png" }
if ($targets) { Write-Host "  * В $(($targets | ForEach-Object Name) -join ' / '): откройте НОВЫЙ чат и попросите «сделай логотип с прозрачным фоном»." }
