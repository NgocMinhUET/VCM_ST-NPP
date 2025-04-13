
# Find the latest model versions
STNPP_MODEL=$(ls -t trained_models/stnpp/stnpp_joint_best_v*.pt | head -1)
QAL_MODEL=$(ls -t trained_models/qal/qal_joint_best_v*.pt | head -1)

echo "Using models:"
echo "STNPP: $STNPP_MODEL"
echo "QAL: $QAL_MODEL"
    
python run_comprehensive_evaluation.py \
  --stnpp_model "$STNPP_MODEL" \
  --qal_model "$QAL_MODEL" \
  --dataset_path datasets/MOTChallenge/MOT16 \
  --sequence MOT16-04 \
  --qp_range 22,27,32,37 \
  --output_dir results/comprehensive \
  --tasks detection,tracking,segmentation \
  --save_videos \
  --plot_curves