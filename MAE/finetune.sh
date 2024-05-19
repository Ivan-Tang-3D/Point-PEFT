NAME=exp_finetune

python -u main.py --config ./cfgs/finetune_scan_hardest.yaml --exp_name ${NAME} --cache_prompt --ckpts ./ckpts/pretrain.pth 