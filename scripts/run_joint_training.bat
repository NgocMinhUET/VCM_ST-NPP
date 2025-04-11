@echo off
REM This script runs joint training with specific model paths

REM Navigate to project root directory
cd %~dp0\..

REM Find the latest model versions (simplified approach for Windows)
echo Looking for latest model versions...

REM You'll need to manually specify the model paths on Windows
REM Adjust these paths to your specific model versions
set STNPP_MODEL=trained_models\stnpp\stnpp_best_v20250409_0726.pt
set QAL_MODEL=trained_models\qal\qal_best_v20250409_0804.pt
set PROXY_MODEL=trained_models\proxy\proxy_network_best_v20250407_1855.pt

echo Using models:
echo STNPP: %STNPP_MODEL%
echo QAL: %QAL_MODEL%
echo PROXY: %PROXY_MODEL%

REM Run joint training with specific model paths
python scripts/train_joint.py ^
  --stnpp_model "%STNPP_MODEL%" ^
  --qal_model "%QAL_MODEL%" ^
  --proxy_model "%PROXY_MODEL%" ^
  --dataset datasets/MOTChallenge/processed ^
  --batch_size 8 ^
  --epochs 20 ^
  --lr 1e-5 ^
  --lambda_distortion 1.0 ^
  --lambda_rate 0.1 ^
  --lambda_perception 0.01 ^
  --output_dir trained_models/joint ^
  --log_dir logs/joint 