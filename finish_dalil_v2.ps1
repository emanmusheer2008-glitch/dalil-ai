# Dalil AI - finish V2 install (run from PowerShell inside Desktop\dalil-ai)
#   cd $HOME\Desktop\dalil-ai ; powershell -ExecutionPolicy Bypass -File .\finish_dalil_v2.ps1
$ErrorActionPreference = "Stop"
$proj = "$HOME\Desktop\dalil-ai"
Set-Location $proj
# 1) keep V1 reachable forever
git tag -f v1 457da21
# 2) unpack V2 over the working tree
Expand-Archive -Path "$proj\dalil_v2_final.zip" -DestinationPath $proj -Force
if (Test-Path "$proj\HANDOVER_V2.md") { Remove-Item "$proj\HANDOVER_V2.md" }   # moved to docs\archive
# 3) organise raw captures (kept out of git by .gitignore)
New-Item -ItemType Directory -Force "$proj\data\raw\official\v2" | Out-Null
Get-ChildItem "$HOME\Downloads\dalil_v2_*.json" | Copy-Item -Destination "$proj\data\raw\official\v2\" -Force
# 4) commit V2
git add -A
git commit -m "Dalil AI V2: 799 services from 10 agencies, hybrid bilingual retrieval, calibrated answer/possible-match/related/decline, leakage-checked evaluation, 90 tests, full docs"
git tag -f v2
git log --oneline -3
# 5) environment + tests
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
Write-Host "Done. Run:  .\.venv\Scripts\streamlit run app.py"
