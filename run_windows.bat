@echo off
REM Double-click to run the whole pipeline on Windows, inside WSL (Linux in Windows).
REM The code and all results live in the WSL home folder: \\wsl$\Ubuntu\home\<user>\epilepsy
REM If it stops (reboot, update), double-click again: finished runs are skipped, long runs resume from checkpoints.
REM
REM N_JOBS = number of simulations run in parallel. Default 28: a 32-thread CPU minus 4 left for other work.
set N_JOBS=28

wsl.exe --status >nul 2>&1
if errorlevel 1 (
  echo WSL is not installed. Open PowerShell as administrator and run:   wsl --install
  echo Reboot, open "Ubuntu" once from the Start menu to create a user, then double-click this file again.
  pause
  exit /b 1
)
wsl.exe bash -lc "cd ~ && ( [ -d epilepsy ] || git clone https://github.com/DumoulinVincent/epilepsy.git ) && cd epilepsy && git pull --ff-only; ./setup.sh && N_JOBS=%N_JOBS% N_HEAVY=%N_JOBS% ./reproduce_all.sh 2>&1 | tee -a reproduce.log"
echo.
echo Finished (or stopped). Log: reproduce.log in the epilepsy folder.
pause
