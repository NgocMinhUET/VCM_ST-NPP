
# Find the latest model versions
STNPP_MODEL=$(ls -t trained_models/joint/stnpp_joint_best_v*.pt | head -1)
QAL_MODEL=$(ls -t trained_models/joint/qal_joint_best_v*.pt | head -1)

echo "Using models:"
echo "STNPP: $STNPP_MODEL"
echo "QAL: $QAL_MODEL"
    
# Sử dụng model_path để trỏ đến STNPP_MODEL, vì script chỉ chấp nhận một model
python run_comprehensive_evaluation.py \
  --model_path "$STNPP_MODEL" \
  --sequence_path datasets/MOTChallenge/MOT16/test/MOT16-04 \
  --max_frames 100 \
  --h264_crf "22,27,32,37" \
  --h265_crf "22,27,32,37" \
  --vp9_crf "22,27,32,37" \
  --output_dir results/comprehensive