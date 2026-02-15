
EPOCHS=250
BATCH_SIZE=640
LR=0.1
WORKERS=32
SEED=1000
ALPHA=0
BETA=0
NOISE_PROB=1
NOISE_TYPES=${1:-"color"} 
CMD="python main.py \
  --wandb_project $PROJECT_NAME \
  --seed $SEED \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --workers $WORKERS \
  --noise_prob $NOISE_PROB \
  --alpha $ALPHA \
  --beta $BETA"

if [ -n "$NOISE_TYPES" ]; then
  CMD="$CMD --noise_types $NOISE_TYPES"
fi

echo "----------------------------------------------------------------"
echo "开始训练任务: $PROJECT_NAME"
echo "噪声类型: ${NOISE_TYPES:-"None"}"
echo "正则化权重: Alpha(GMS)=$ALPHA, Beta(TCS)=$BETA"
echo "执行命令: $CMD"
echo "----------------------------------------------------------------"

eval $CMD