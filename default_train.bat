@echo off
setlocal

set N_EPOCHS=90
set MODEL_NAME=lucidrain_vit
set BATCH=256
set DATA_FOLDER=C:\data\imagenet1k\ILSVRC
set MODEL_SAVE_DIR=C:\temp\trained_models

python "C:\easy_train\train.py" ^
  --arch "%MODEL_NAME%" ^
  --batch-size %BATCH% ^
  --data "%DATA_FOLDER%" ^
  --epochs %N_EPOCHS% ^
  --output_parent_dir "%MODEL_SAVE_DIR%" ^
  --mixup


set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
