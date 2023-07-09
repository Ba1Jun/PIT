export  CUDA_VISIBLE_DEVICES=0
DATASET=BQ_corpus
ENCODER_TYPE=cbert
BATCH_SIZE=64
METHOD=pit
EPOCH=10
LR=3e-5
PIT_WEIGHT=15

SAVE_PATH=./output/${DATASET}/${ENCODER_TYPE}_${METHOD}/pit${PIT_WEIGHT}_e${EPOCH}_bs${BATCH_SIZE}_lr${LR}
mkdir -p ${SAVE_PATH}
python3 train.py \
  --dataset ${DATASET} \
  --max_len_1 36 \
  --max_len_2 36 \
  --epoch ${EPOCH} \
  --batch_size ${BATCH_SIZE} \
  --main_metric accuracy \
  --method ${METHOD} \
  --encoder_type ${ENCODER_TYPE} \
  --pit_weight ${PIT_WEIGHT} \
  --save_model_path ${SAVE_PATH} \
  --learning_rate ${LR} \
  --warmup_proportion 0.1 \
  --max_grad_norm 10.0 \
  --weight_decay 0.01 \
  --seed 42