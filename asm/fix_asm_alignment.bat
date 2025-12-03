@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM fix_asm_alignment.bat
REM Fixes vmovapd alignment issues by changing RIP-relative vmovapd to vmovupd
REM This prevents random startup crashes on Windows where .rodata section
REM alignment is not guaranteed to be 32-byte aligned.
REM ============================================================================

set "ASM_FILE=bocpd_avx2_kernel_intel.asm"
set "BACKUP_FILE=bocpd_avx2_kernel_intel.asm.bak"

if not exist "%ASM_FILE%" (
    echo ERROR: %ASM_FILE% not found in current directory
    echo Please run this script from the directory containing the ASM file
    exit /b 1
)

echo.
echo ============================================================================
echo  ASM Alignment Fix - vmovapd to vmovupd for [rel ...] addresses
echo ============================================================================
echo.

REM Create backup
echo Creating backup: %BACKUP_FILE%
copy /Y "%ASM_FILE%" "%BACKUP_FILE%" >nul

REM Use PowerShell to do the replacement
echo Fixing RIP-relative vmovapd instructions...

powershell -Command ^
    "$content = Get-Content '%ASM_FILE%' -Raw; " ^
    "$count = ([regex]::Matches($content, 'vmovapd(\s+ymm\d+,\s*\[rel)')).Count; " ^
    "$content = $content -replace 'vmovapd(\s+ymm\d+,\s*\[rel)', 'vmovupd$1'; " ^
    "$content | Set-Content '%ASM_FILE%' -NoNewline; " ^
    "Write-Host \"  Replaced $count occurrences\""

if errorlevel 1 (
    echo ERROR: PowerShell replacement failed
    echo Restoring backup...
    copy /Y "%BACKUP_FILE%" "%ASM_FILE%" >nul
    exit /b 1
)

echo.
echo Done! Changes made:
echo   - vmovapd ymm*, [rel ...] -^> vmovupd ymm*, [rel ...]
echo.
echo Stack and buffer vmovapd instructions (aligned memory) are unchanged.
echo.
echo To revert: copy %BACKUP_FILE% %ASM_FILE%
echo.

exit /b 0
