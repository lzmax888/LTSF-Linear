# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear
num_workers=35
itr=1
train_epochs=5
enc_in=1

python -u run_longExp.py \
  --is_training 1 \
  --root_path /kaggle/input/btc2021/ \
  --data_path Exness_BTCUSD_2021.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in $enc_in \
  --des 'Exp' \
  --itr $itr --batch_size 512 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log \
  --num_workers $num_workers \
  --train_epochs $train_epochs

python -u run_longExp.py \
  --is_training 1 \
  --root_path /kaggle/input/btc2021/ \
  --data_path Exness_BTCUSD_2021.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in $enc_in \
  --des 'Exp' \
  --itr $itr --batch_size 64 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'192.log \
  --num_workers $num_workers \
  --train_epochs $train_epochs

python -u run_longExp.py \
  --is_training 1 \
  --root_path /kaggle/input/btc2021/ \
  --data_path Exness_BTCUSD_2021.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in $enc_in \
  --des 'Exp' \
  --itr $itr --batch_size 64  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'336.log   \
  --num_workers $num_workers \
  --train_epochs $train_epochs

python -u run_longExp.py \
  --is_training 1 \
  --root_path /kaggle/input/btc2021/ \
  --data_path Exness_BTCUSD_2021.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in $enc_in \
  --des 'Exp' \
  --itr $itr --batch_size 64 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'720.log  \
  --num_workers $num_workers \
  --train_epochs $train_epochs