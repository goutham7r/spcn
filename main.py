import glob
import sys
import os

from Run_StainSep import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm

#setting tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#to use cpu instead of gpu, uncomment the below line
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #use only GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '' #use only CPU

import tensorflow as tf
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
# config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)


#Parameters
nstains=2    #number of stains
lamb=0.1     #default value sparsity regularization parameter
# lamb=0 equivalent to NMF



# 0= stain separation of all images in a folder
# 1= stain separation of single image 
# 2= color normalization of one image with one target image
# 3= color normalization of all images in a folder with one target image
# 4= color normalization of one image with multiple target images
# 5= color normalization of all images in a folder with multiple target images individually
op=3

if op==0:
	filename="./source.png"
	print filename
	run_stainsep(filename,nstains,lamb)


elif op==1:
	input_direc="./input/"
	output_direc="./stain_separated/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	file_type="*"
	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print "No source files found"
		sys.exit()
	filenames=sorted(glob.glob(input_direc+file_type))
	print filenames
	for filename in filenames:
		run_stainsep(filename,nstains,lamb,output_direc=output_direc)


elif op==2:
	level=0
	output_direc="./output_direc/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)

	source_filename="./source.png"
	target_filename="./target.png"

	if not os.path.exists(source_filename):
		print "Source file does not exist"
		sys.exit()
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()
	background_correction = True	
	run_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level,background_correction,config=config)


elif op==3:
	level=0

	input_direc="./source_files/"
	output_direc="./normalized/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	file_type="*.png" #all of these file types from input_direc will be normalized
	
	target_filename="./target.png"
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()
	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print "No source files found"
		sys.exit()
	#filename format of normalized images can be changed in run_batch_colornorm
	filenames=[target_filename]+sorted(glob.glob(input_direc+file_type))
	
	background_correction = True
	# background_correction = False	
	run_batch_colornorm(filenames,nstains,lamb,output_direc,level,background_correction,config)
