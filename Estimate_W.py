import sys
import os
import spams
import numpy as np
import math
from sklearn import preprocessing
from multiprocessing import Pool
from functools import partial
import signal

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)



def Wfast(img,nstains,lamb,num_patches,patchsize,level,background_correction=False):
	
	param=definePar(nstains,lamb)
	_max=3000
	max_size=_max*_max
	xdim,ydim=img.level_dimensions[0]
	patchsize=int(min(patchsize,xdim/3,ydim/3))
	patchsize_original=patchsize
	nstains=param['K']
	valid_inp=[]
	
	white_pixels=[]

	#100,000 pixels or 20% of total pixels is maximum number of white pixels sampled
	max_num_white=min(100000,(xdim*ydim)/5)
	min_num_white=10000

	white_cutoff=220
	I_percentile=80

	if ydim*xdim>max_size:
		print "Finding patches for W estimation:"
		for j in range(20):
			#print "Patch Sampling Attempt:",i+1
			initBias=int(math.ceil(patchsize/2)+1) 
			xx=np.array(range(initBias,xdim-initBias,patchsize))
			yy=np.array(range(initBias,ydim-initBias,patchsize))
			xx_yy=np.transpose([np.tile(xx, len(yy)), np.repeat(yy, len(xx))])
			np.random.shuffle(xx_yy)

			threshold=0.1 #maximum percentage of white pixels in patch
			for i in range(len(xx_yy)):
				patch=np.asarray(img.read_region((xx_yy[i][0],xx_yy[i][1]),level,(patchsize,patchsize)))
				patch=patch[:,:,:3]
				if len(white_pixels)<max_num_white:
					white_pixels.extend(patch[np.sum((patch>white_cutoff),axis=2)==3])

				if patch_Valid(patch,threshold):
					valid_inp.append(patch)
					if len(valid_inp)==num_patches:
						break

			if len(valid_inp)==num_patches:
				white_pixels=np.array(white_pixels[:max_num_white])
				break																																																																																																																																	
			patchsize=int(patchsize*0.95)
		valid_inp=np.array(valid_inp)
		print "Number of patches sampled for W estimation:", len(valid_inp)
	else:
		patch=np.asarray(img.read_region((0,0),level,(xdim,ydim)))
		patch=patch[:,:,:3]
		valid_inp=[]
		valid_inp.append(patch)
		white_pixels= patch[np.sum((patch>white_cutoff),axis=2)==3]
		print "Image small enough...W estimation done using whole image"

	if background_correction:
		print "Number of white pixels sampled",len(white_pixels)
		if len(white_pixels)<min_num_white:
			i0=np.array([255.0,255.0,255.0])
			print "Not enough white pixels found, default background intensity assumed"
		elif len(white_pixels)>0:
			i0 = np.percentile(white_pixels,I_percentile,axis=0)[:3]
		else:
			i0 = None
	else:
		i0 = np.array([255.0,255.0,255.0])

	if len(valid_inp)>0:
		out = suppress_stdout()
		pool = Pool(initializer=initializer)
		try:
		    WS = pool.map(partial(getstainMat,param=param,i_0=i0),valid_inp)
		except KeyboardInterrupt:
			pool.terminate()
			pool.join()
		pool.terminate()
		pool.join()
		suppress_stdout(out)

		WS=np.array(WS)

		if WS.shape[0]==1:
			Wsource=WS[0,:3,:]
		else:
			print "Median color basis of",len(WS),"patches used as actual color basis"
			Wsource=np.zeros((3,nstains))
			for k in range(nstains):
			    Wsource[:,k]=[np.median(WS[:,0,k]),np.median(WS[:,1,k]),np.median(WS[:,2,k])]
		
		Wsource = W_sort(normalize_W(Wsource))

		if Wsource.sum()==0:
			if patchsize*0.95<100:
				print "No suitable patches found for learning W. Please relax constraints"
				return None			#to prevent infinite recursion
			else:
				print "W estimation failed, matrix of all zeros found. Trying again..."				
				return Wfast(img,nstains,lamb,min(100,num_patches*1.5),int(patchsize_original*0.95),level)
		else:
			return Wsource,i0
	else:
		print "No suitable patches found for learning W. Please relax constraints"
		return None,None

#defines validity of patch for W-estimation
#valid if percentage of white pixels is less than threshold
#can be customized as per needs
def patch_Valid(patch,threshold):
	r_th=220 #red channel threhold for white pixels
	g_th=220 #green channel threhold for white pixels
	b_th=220 #blue channel threhold for white pixels
	
	tempr = patch[:,:,0]>r_th
	tempg = patch[:,:,1]>g_th
	tempb = patch[:,:,2]>b_th
	
	#mask for white pixels
	temp = tempr*tempg*tempb
	
	r,c = np.shape((temp))
	#percentage of white pixels in patch 
	per= float(np.sum(temp))/float((r*c))
	
	if per>threshold:
		return False
	else:
		return True  

#Sorts the columns of the color basis matrix such that the first column is H, second column is E.
def W_sort(W):

	# 1. Using r values of the vectors. E must have a smaller value of r (as it is redder) than H.
	# 2. Using b values of the vectors. H must have a smaller value of b (as it is bluer) than E.
	# 3. Using r-b values of the vectors. H must have a larger value of r-b.

	method = 3
	# Choose whichever method works best for your images

	if method==1:
		W = W[:,np.flipud(W[0,:].argsort())]

	elif method==2:
		W = W[:,W[2,:].argsort()]

	elif method==3:
		r_b_1 = W[0][0]-W[2][0]
		r_b_2 = W[0][1]-W[2][1]
		if r_b_1<r_b_2: 
			W[:,[0, 1]] = W[:,[1, 0]]
		#else no need to switch

	return W

#Beer-Lambert transform function
#I: array of pixel intensities, i_0: background intensities
def BLtrans(I,i_0):
	#flatten 3D array
	Ivecd = vectorise(I)

	#V=WH, +1 is to avoid divide by zero
	V=np.log(i_0)- np.log(Ivecd+1.0)
	#shape of V = no. of pixels x 3 

	#thresholding white pixel checking
	w_threshold=220
	c = (Ivecd[:,0]<w_threshold) * (Ivecd[:,1]<w_threshold) * (Ivecd[:,2]<w_threshold)
	
	#extract only non-white pixels
	Ivecd=Ivecd[c]
	#BL transform of non-white pixels only
	VforW=np.log(i_0)- np.log(Ivecd+1.0) 
	
	return V,VforW

#function for dictionary learning
#I:patch for W estimation, param: DL parameters, i_0: background intensities
def getstainMat(I,param,i_0):
	#Beer-Lambert transform
	V,VforW=BLtrans(I,i_0)

	out = suppress_stdout()

	#Sparse NMF (Learning W; V=WH)
	#W is learnt only using VforW, i.e. by ignoring the white pixels
	#change VforW to V for W-estimation using all pixels
	Ws = spams.trainDL(np.asfortranarray(np.transpose(VforW)),**param)
	
	suppress_stdout(out)

	return Ws

#makes the columns of the color basis matrix W unit norm
def normalize_W(W):
	W1 = preprocessing.normalize(W, axis=0, norm='l2')
	return W1

#defines parameters for dictionary learning	
def definePar(nstains,lamb,batch=None):
	param={}	
	param['lambda1']=lamb
	param['posAlpha']=True         #positive stains 
	param['posD']=True             #positive staining matrix
	param['modeD']=0               #{W in Real^{m x n}  s.t.  for all j,  ||d_j||_2^2 <= 1 }
	param['whiten']=False          #Do not whiten the data                      
	param['K']=nstains             #No. of stains = 2 for H&E
	param['numThreads']=-1         #number of threads
	param['iter']=40               #20-50 is fine
	param['clean']=True
	if batch is not None:
		param['batchsize']=batch   #Give here input image no of pixels for traditional dictionary learning
	return param

#flattens 3D array of pixel intensities of size (l,b,3) into 2D array of size ((l*b),3) 
def vectorise(I):
	s=I.shape
	return np.reshape(I, (s[0]*s[1],s[2]))

#auxiliary function to improve functionality of parallel pool functions used
def initializer():
    #Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

#auxiliary function to improve functionality of parallel pool functions used
def suppress_stdout(out=None):
	if out is None:
		devnull = open('/dev/null', 'w')
		oldstdout_fno = os.dup(sys.stdout.fileno())
		os.dup2(devnull.fileno(), 1)
		return oldstdout_fno
	else:
		os.dup2(out, 1)