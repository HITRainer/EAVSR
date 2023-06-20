echo "Start to train the model...."

name="xxx" # replace 'xxx' with the training name

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
		mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train_basic.py \
	--model eavsrp        --niter 400              --lr_decay_iters 100   --dataroot /hdd2/wrh/dataset/MVSR4x      \
	--name $name          --dataset_name p50       --print_freq 100       --predict  False     --n_seq 100          \
	--save_imgs True      --batch_size 8           --patch_size 64        --scale 4                                 \
	--calc_psnr True      --lr 1e-4      -j 4      --gpu_ids 0            --n_frame 7 | tee $LOG

