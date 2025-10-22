export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="/home/elicer/dataset/Dresscode/DressCode"
export OUTPUT_DIR="./output_512_densepose_dc_w_seq_concat"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1


export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export PROJECT_NAME="output_512_densepose_dc_w_seq_concat"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #,max_split_size_mb:128,garbage_collection_threshold:0.8
#--pretrained_inpaint_model_name_or_path="xiaozaa/flux1-fill-dev-diffusers" \
accelerate launch --config_file accelerate_config.yaml train_flux_inpaint_dc_pose_2.py \
  --project_name=$PROJECT_NAME \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --validation_epochs=2500 \
  --validation_steps=1000 \
  --seed="42" \
  --dataroot="/home/elicer/dataset/Dresscode/DressCode"  \
  --train_data_list="/home/elicer/dataset/Dresscode/DressCode/train_pairs.txt"  \
  --train_verification_list="/home/elicer/dataset/Dresscode/DressCode/test_pairs_1020.txt"  \
  --validation_data_list="/home/elicer/dataset/Dresscode/DressCode/test_pairs_1020.txt"  \
  --height=512 \
  --width=384 \
  --max_sequence_length=512  \
  --checkpointing_steps=500  \
  --report_to="wandb" \
  --train_base_model \
  --gradient_checkpointing \
  --allow_tf32 \
  --head_layer_select all \
  --avg_heads \
  --resume_from_checkpoint="latest" \
  # --height=768 \
  # --width=576 \  # --resume_from_checkpoint="latest"  \
  #--flow_gt_dir="/home/elicer/projects/syj/sd-dino/experiment/results_np/train/2_images_original_v3_only_dino_cycle_1024_768" \