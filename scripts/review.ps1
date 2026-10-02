$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONPATH = "${PWD}/src;${PWD}/vendor"
$env:OMP_NUM_THREADS = '2'
python -m pytest -q --basetemp=review/pytest_local
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
python -m aha audit
if ($LASTEXITCODE -ne 0) { throw 'Dataset audit failed' }
python -m aha verify
if ($LASTEXITCODE -ne 0) { throw 'Frozen hashes differ' }
