import glob
import sys
import os

from Run_StainSep import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm

#setting tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #use only GPU-0
#to use cpu instead of gpu, uncomment the below line
# os.environ['CUDA_VISIBLE_DEVICES'] = '' #use only CPU

import tensorflow as tf
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)


#Parameters
nstains=2    #number of stains
lamb=0.1     #default value sparsity regularization parameter
# lamb=0 is equivalent to NMF


# 1= stain separation of a single image 
# 2= color normalization of one image with one target image
# 3= color normalization of all images in a folder with one target image
op=3

if op==1:
	#individual stains will be stored in this directory
	output_direc="./output_direc/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)

	#path to image to be stain separated
	filename="./img.png"

	#WSIs read by openslide can be read at different levels of resolution
	#level=0 is the highest resolution
	#for non-WSI images, keep level=0
	level=0

	#set to false for not using background whitespace correction (using 255 in BL-transform)
	#highly recommended to keep it True
	background_correction = True

	run_stainsep(filename,nstains,lamb,level,output_direc,background_correction)

elif op==2:
	#normalized source image will be stored in this directory
	output_direc="./output_direc/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	#filename format of normalized image can be changed in run_batch_colornorm


	#path to source image	
	source_filename="./source.png"
	if not os.path.exists(source_filename):
		print "Source file does not exist"
		sys.exit()
	
	#path to target image
	target_filename="./target.png"
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()

	#WSIs read by openslide can be read at different levels of resolution
	#level=0 is the highest resolution
	#for non-WSI images, keep level=0
	level=0

	#set to false for not using background whitespace correction (using 255 in BL-transform)
	#highly recommended to keep it True
	background_correction = True	
	
	run_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level,background_correction,config=config)


elif op==3:

	#normalized source images will be stored in this directory
	output_direc="./normalized/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	#filename format of normalized images can be changed in run_batch_colornorm

	#path to target image
	target_filename="./target.png"
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()

	#directory containing source images
	input_direc="./source_files/"
	
	#images from input_direc matching below file type will be normalized
	file_type="*.png"
	#can also use this as regex to extract relevant images

	#WSIs read by openslide can be read at different levels of resolution
	#level=0 is the highest resolution
	#for non-WSI images, keep level=0
	level=0

	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print "No source files found"
		sys.exit()
	filenames=[target_filename]+sorted(glob.glob(input_direc+file_type))
	
	#set to false for not using background whitespace correction (using 255 in BL-transform)
	#highly recommended to keep it True
	background_correction = True

	run_batch_colornorm(filenames,nstains,lamb,output_direc,level,background_correction,config)
