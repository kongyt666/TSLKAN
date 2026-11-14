python3 -u run.py    --task_name    long_term_forecast    --is_training 1    --root_path ./dataset/weather/    --data_path weather.csv    --model_id weather_96_96    --model TSLKAN    --data custom    --features M    --seq_len 96    --label_len 48    --pred_len 96    --e_layers 2  --d_layers 2  --factor 3    --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'    --batch_size 32   --itr 1    --d_model 16    --d_ff 8    --dropout 0.1    --learning_rate 0.004

python3 -u run.py    --task_name    long_term_forecast    --is_training 1    --root_path ./dataset/weather/    --data_path weather.csv    --model_id weather_96_192   --model TSLKAN    --data custom    --features M    --seq_len 96    --label_len 48    --pred_len 192    --e_layers 2  --d_layers 2  --factor 3    --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'    --batch_size 32   --itr 1    --d_model 16    --d_ff 4    --dropout 0.1    --learning_rate 0.004

python3 -u run.py    --task_name    long_term_forecast    --is_training 1    --root_path ./dataset/weather/    --data_path weather.csv    --model_id weather_96_336   --model TSLKAN    --data custom    --features M    --seq_len 96    --label_len 48    --pred_len 336    --e_layers 2  --d_layers 2  --factor 3    --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'    --batch_size 32   --itr 1    --d_model 64    --d_ff 16    --dropout 0.1    --learning_rate 0.0017

python3 -u run.py    --task_name    long_term_forecast    --is_training 1    --root_path ./dataset/weather/    --data_path weather.csv    --model_id weather_96_720   --model TSLKAN    --data custom    --features M    --seq_len 96    --label_len 48    --pred_len 720    --e_layers 1  --d_layers 2  --factor 3    --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'    --batch_size 64   --itr 1    --d_model 16    --d_ff 4    --dropout 0.1    --learning_rate 0.0035




