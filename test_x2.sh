echo "Start to test the model...."

name="realvsr" # replace 'xxx' with the training name
dataroot='/hdd2/wrh/dataset/RealVSR'
model='eavsrpx2'
device="0" # GPU index
iter="400" # epoch
full='True'
frame='10'

scale='2'
data='realvsr' # name of the dataset
chop='False'


python test_basic.py \
    --model $model  --name $name  --dataset_name $data  --chop $chop --full_res $full --n_frame $frame --n_seq 50 \
    --load_iter $iter    --save_imgs True  --calc_psnr True  --gpu_ids $device  --scale $scale --dataroot $dataroot

python psnr_total.py  --device $device --name $name --load_iter $iter  --full_res $full --dataroot $dataroot
