import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model, networks as N
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy



if __name__ == '__main__':
	opt = TestOptions().parse()
	# log_dir = '%s/%s/psnr_x%s.txt' % (opt.checkpoints_dir, opt.name, opt.scale)
	# f = open(log_dir, 'a')
	if not isinstance(opt.load_iter, list):
		load_iters = [opt.load_iter]
	else:
		load_iters = deepcopy(opt.load_iter)

	# load_iters = [ i for i in range(600, 801)]

	if not isinstance(opt.dataset_name, list):
		dataset_names = [opt.dataset_name]
	else:
		dataset_names = deepcopy(opt.dataset_name)
	datasets = odict()
	for dataset_name in dataset_names:
		dataset = create_dataset(dataset_name, 'test', opt)
		datasets[dataset_name] = tqdm(dataset)

	log_dir = '%s/%s/test_log/log_%s.txt' % (opt.checkpoints_dir, \
			   opt.name, str(load_iters[0]) + '_' + str(load_iters[-1]))
	os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
	f = open(log_dir, 'a')
	
	visualizer = Visualizer(opt)
	model = create_model(opt)

	for load_iter in load_iters:
		opt.load_iter = load_iter
		model.setup(opt)
		model.eval()

		for dataset_name in dataset_names:
			opt.dataset_name = dataset_name
			tqdm_val = datasets[dataset_name]
			dataset_test = tqdm_val.iterable
			dataset_size_test = len(dataset_test)

			print('='*80)
			print(dataset_name)
			tqdm_val.reset()

			psnr = [0.0] * dataset_size_test
			ssim = [0.0] * dataset_size_test

			time_val = 0
			print(tqdm_val)
			for i, data in enumerate(tqdm_val):
				torch.cuda.empty_cache()
				model.set_input(data, 0)
				torch.cuda.synchronize()
				time_val_start = time.time()
				model.test()
				torch.cuda.synchronize()
				time_val += time.time() - time_val_start
				res = model.get_current_visuals()

				if opt.calc_psnr:
					psnr_curr = calc_psnr(res['data_sr_seq'], res['data_hr_seq'])
					f.write('folder: %s, PSNR: %s\n'
					% (data['fname'][0][0][:3], psnr_curr))
					psnr[i] = psnr_curr

				if opt.save_imgs:
					# folder_dir = './ckpt/%s/hr_%s' % (opt.name, camera)  # visual_test_1204, visual_train_1827warp
					# os.makedirs(folder_dir, exist_ok=True)
					# save_dir = '%s/%s' % (folder_dir, data['fname'][0])
					# dataset_test.imio.write(np.array(res['data_hr'][0].cpu()).astype(np.uint8), save_dir)
					for i in range(res['data_sr_seq'].shape[1]):
						if opt.full_res:
							folder_dir = './ckpt/%s/sr_full_%s/%s' % (opt.name, opt.load_iter, data['fname'][i][0][:3])  
						else:
							folder_dir = './ckpt/%s/sr_patch_%s/%s' % (opt.name, opt.load_iter, data['fname'][i][0][:3])  
						os.makedirs(folder_dir, exist_ok=True)
						save_dir = '%s/%s' % (folder_dir, data['fname'][i][0][-9:])
						dataset_test.imio.write(np.array(res['data_sr_seq'][0][i,...].cpu()).astype(np.uint8), save_dir)
				print(model.num, '----', model.time, '----')

					# folder_dir = './ckpt/%s/lr_%s' % (opt.name, camera)  # visual_test_1204, visual_train_1827warp
					# os.makedirs(folder_dir, exist_ok=True)
					# save_dir = '%s/%s' % (folder_dir, data['fname'][0])
					# dataset_test.imio.write(np.array(res['data_lr_up'][0].cpu()).astype(np.uint8), save_dir)

					# folder_dir = './ckpt/%s/hr400x400_%s' % (opt.name, camera)  # visual_test_1204, visual_train_1827warp
					# os.makedirs(folder_dir, exist_ok=True)
					# save_dir = '%s/%s' % (folder_dir, data['fname'][0])
					# dataset_test.imio.write(np.array(res['data_hr'][0].cpu()).astype(np.uint8), save_dir)

					# folder_dir = './ckpt/%s/visual_gcm_warp_%s' % (opt.name, camera)  # visual_test_1204, visual_train_1827warp
					# os.makedirs(folder_dir, exist_ok=True)
					# save_dir = '%s/%s' % (folder_dir, data['fname'][0])
					# dataset_test.imio.write(np.array(res['hr_warp'][0].cpu()).astype(np.uint8), save_dir)

					# folder_dir = './ckpt/%s/visual_mask_%s' % (opt.name, camera)  # visual_test_1204, visual_train_1827warp
					# os.makedirs(folder_dir, exist_ok=True)
					# save_dir = '%s/%s' % (folder_dir, data['fname'][0])
					# dataset_test.imio.write(np.array(res['mask'][0].cpu()).astype(np.uint8), save_dir)

			visualizer.print_psnr(load_iter, '/' , '/' , np.mean(psnr), print_psnr=False)
			avg_psnr = '%.6f'%np.mean(psnr)
			avg_ssim = '%.6f'%np.mean(ssim)
			
			f.write('dataset: %s, PSNR: %s, Epoch: %s, Time: %.3f sec.\n'
					% (dataset_name, avg_psnr, load_iter, time_val))
			print('Time: %.3f s AVG Time: %.3f ms PSNR: %s Epoch: %s\n'
				   % (time_val, time_val/dataset_size_test*1000, avg_psnr, load_iter))
			f.flush()
			f.write('\n')
	f.close()
	for dataset in datasets:
		datasets[dataset].close()
