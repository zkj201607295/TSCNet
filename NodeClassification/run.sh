#!/bin/sh
#JSUB -q normal
#JSUB -n 4
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_jobtrain_rate
#source /apps/software/anaconda3/bin/activate zkj
conda activate pyg
python training_batch.py  --dataset Cora --lr 0.01 --dprate 0.9 --dropout 0.6  --train_rate 0.6 --val_rate 0.2 --early_stopping 70 > result/Ablation/Direct_c/Cora.txt
python training_batch.py  --dataset Citeseer --lr 0.01 --dprate 0.8 --dropout 0.5  --train_rate 0.6 --val_rate 0.2 --early_stopping 10 > result/Ablation/Direct_c/Citeseer.txt
python training_batch.py  --dataset Computers --lr 0.01 --dprate 0.5 --dropout 0.0 --train_rate 0.6 --val_rate 0.2 --early_stopping 200 > result/Ablation/Direct_c/Computers.txt
python training_batch.py  --dataset Pubmed --lr 0.01 --dprate 0.5 --dropout 0.0  --train_rate 0.6 --val_rate 0.2 --early_stopping 200 > result/Ablation/Direct_c/Pubmed.txt
python training_batch.py  --dataset Photo --lr 0.01 --dprate 0.5 --dropout 0.0  --train_rate 0.6 --val_rate 0.2 --early_stopping 200  > result/Ablation/Direct_c/Photo.txt
python training_batch.py  --dataset CS --lr 0.1 --dprate 0.5 --dropout 0.5  --train_rate 0.6 --val_rate 0.2 --early_stopping 200 > result/Ablation/Direct_c/CS.txt
python training_batch.py  --dataset Physics --lr 0.1 --dprate 0.5 --dropout 0.8  --train_rate 0.6 --val_rate 0.2 --early_stopping 200 > result/Ablation/Direct_c/Physics.txt
python training_batch.py  --dataset Actor --lr 0.01 --dprate 0.0 --dropout 0.0  --train_rate 0.6 --val_rate 0.2 --early_stopping 70 > result/Ablation/Direct_c/Actor.txt
python training_batch.py  --dataset Texas --lr 0.03 --dprate 0.6 --dropout 0.9  --train_rate 0.6 --val_rate 0.2 --early_stopping 100  > result/Ablation/Direct_c/Texas.txt
python training_batch.py  --dataset Cornell --lr 0.03 --dprate 0.6 --dropout 0.9  --train_rate 0.6 --val_rate 0.2 --early_stopping 100  > result/Ablation/Direct_c/Cornell.txt