#!/bin/sh

#SBATCH --account             sxq@v100
#SBATCH --constraint          v100-32g
#SBATCH --cpus-per-task       10
#SBATCH --error               log/%x/%j/errors.err
#SBATCH --gres                gpu:4
#SBATCH --hint                nomultithread
#SBATCH --job-name            3_ResNet
#SBATCH --nodes               1
#SBATCH --ntasks              4
#SBATCH --output              log/%x/%j/logs.out
#SBATCH --qos                 qos_gpu-t3
#SBATCH --time                20:00:00
#SBATCH --signal              USR1@40

module purge
conda deactivate
module load anaconda-py3/2023.09
conda activate \$WORK/venvs/venvResolution
export WANDB_DIR=\$WORK/wandb/
export WANDB_MODE=offline


srun python3 example_train_resnet50.py --data_path $DSDIR/imagenet/ --model resnet50 --device cuda --epochs 600 --batch_size 256 --workers 8 --lr 0.5 --momentum 0.9 --weight_decay 2e-05 --lr_warmup_decay 0.01 --lr_warmup_method linear --lr_warmup_epochs 5 --print_freq 10 --output_dir $WORK/results_resnet/ --start_epoch 0 --logger txt --name resnet50_resize_128_162_167_0_64_64_128_256_512 --opt sgd --norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 --lr_scheduler cosineannealinglr --lr_step_size 30 --lr_gamma 0.1 --lr_min 0.0 --auto_augment ta_wide --ra_magnitude 9 --augmix_severity 3 --random_erase 0.1 --model_ema --model_ema_steps 32 --model_ema_decay 0.99998 --interpolation bilinear --val_resize_size 167 --val_crop_size 162 --train_crop_size 128 --ra_sampler --ra_reps 4