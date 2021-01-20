import numpy as np
import matplotlib.pylab as plt
import scipy.io as spio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
from scipy.signal import butter,filtfilt
from scipy import signal, misc

__all__ = ['plot_mixed', 'plot_I_pp', 'plot_I_ss', 'plot_I_laplace', 'plot_I_pp_total', 'plot_I_laplace_t', 'plot_I_total']

def plot_I_laplace(address, out, vm1, vm2):
	mat_content = spio.loadmat('../outputs/%s.mat'%address)
	Image_pp    = copy.deepcopy(mat_content['Ima'])
	Image_pp_N  = copy.deepcopy(mat_content['Ima_N'])
	nbl         = mat_content['nbl'][0][0]
	shape=(Image_pp.shape[0]-2*nbl+1, Image_pp.shape[1]-2*nbl+1)
	ck = signal.cspline2d(Image_pp, 8.0)
	derfilt = np.array([1.0, -2, 1.0], dtype=np.float32)
	deriv = (signal.sepfir2d(ck, derfilt, [1])+signal.sepfir2d(ck, [1], derfilt))
	
	
	laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float32)
	deriv2 = signal.convolve2d(ck,laplacian,mode='same',boundary='symm')
	
	vmin=vmax=np.max(deriv)/vm1
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(deriv[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_deriv_%s.png'%out)
	
	vmin=vmax=np.max(deriv)/vm2
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(deriv2.T, aspect='auto', vmin=-vmin, vmax=vmax)	
	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('fig/I_deriv2_%s.png'%out)

def plot_I_pp(address, out, vm, vm_N, vm_F):
	mat_content = spio.loadmat('outputs/%s.mat'%address)
	Image_pp    = copy.deepcopy(mat_content['Ima'])
	Image_pp_N  = copy.deepcopy(mat_content['Ima_N'])
	nbl         = mat_content['nbl'][0][0]
	shape=(Image_pp.shape[0]-2*nbl+1, Image_pp.shape[1]-2*nbl+1)
	print('Image shape',shape)
	
	vmin=vmax=np.max(Image_pp)/vm #1.769829199461798e-11 #np.max(Image_pp)/vm
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_pp[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_%s.png'%out)
	
	vmin_n=vmax_n=np.max(Image_pp)/vm_N# 3.5396583989235955e-05 #np.max(Image_pp)/vm_N
	print('colorbar min_n', vmin_n)
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_pp_N[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin_n, vmax=vmax_n)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_N_%s.png'%out)
	s=2
	w=5
	t=(((w - 1)/2)-0.5)/s
	lowpass = ndimage.gaussian_filter(Image_pp,sigma=s, truncate=t)
	Image_Filtered_pp = Image_pp-lowpass
	
	vmin=vmax=np.max(Image_Filtered_pp)/vm_F#  1.3030108655839902e-11 #np.max(Image_Filtered_pp)/vm_F	
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered_pp[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_F_%s.png'%out)

def plot_I_ss(address, out, vm, vm_N, vm_F):
	mat_content = spio.loadmat('../outputs/%s.mat'%address)
	Image_pp    = copy.deepcopy(mat_content['Ima'])
	Image_pp_N  = copy.deepcopy(mat_content['Ima_N'])
	nbl         = mat_content['nbl'][0][0]
	shape=(Image_pp.shape[0]-2*nbl+1, Image_pp.shape[1]-2*nbl+1)
	print('Image shape',shape)
	
	vmin=vmax=np.max(Image_pp)/vm #3.3190877445345015e-12 #np.max(Image_pp)/vm
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_pp[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_ss_%s.png'%out)
	
	vmin_n=vmax_n= np.max(Image_pp)/vm_N #3.319087744534501e-05 #np.max(Image_pp)/vm_N
# 	print('colorbar min_n', vmin_n)
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_pp_N[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin_n, vmax=vmax_n)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_ss_N_%s.png'%out)
	s=4
	w=2
	t=(((w - 1)/2)-0.5)/s
	lowpass = ndimage.gaussian_filter(Image_pp,sigma=s, truncate=s)
	Image_Filtered_pp = Image_pp-lowpass
	
	vmin=vmax= np.max(Image_Filtered_pp)/vm_F #3.1763661609409705e-12 #np.max(Image_Filtered_pp)/vm_F	
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered_pp[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_ss_F_%s.png'%out)

def plot_mixed(nbl, address, vm1, vm2, vm3, vm4):
	mat_contents = spio.loadmat('../outputs/%s.mat'%address)
	ImageT_zz    = mat_contents['ImageT_zz']
	ImageT_N_zz  = mat_contents['ImageT_N_zz']
	ImageT_N_zzf = mat_contents['ImageT_N_zzf']
	
	ImageT_xx    = mat_contents['ImageT_xx']
	ImageT_N_xx  = mat_contents['ImageT_N_xx']
	ImageT_N_xxf = mat_contents['ImageT_N_xxf']
	
	shape=(ImageT_zz.shape[0]-2*nbl+1, ImageT_zz.shape[1]-2*nbl+1)
	
	#################################
	vmin=vmax=np.max(ImageT_xx)/vm1
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_xx[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_xx.png')
	
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_zz[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_zz.png')
	
	##############################################\
	vmin=vmax=np.max(ImageT_xx)/vm2
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_N_xx[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_N_xx.png')
	
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_N_zz[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_N_zz.png')
	
	##############################################
	vmin=vmax=np.max(ImageT_xx)/vm3
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_N_xxf[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_N_xxf.png')
	
	plt.figure(figsize=(8, 8))
	plt.imshow(ImageT_N_zzf[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_N_zzf.png')
	
	##############################################
	
	lowpass = ndimage.gaussian_filter(ImageT_xx,1)
	Image_Filtered_xx = ImageT_xx-lowpass
	lowpass = ndimage.gaussian_filter(ImageT_zz,1)
	Image_Filtered_zz = ImageT_zz-lowpass
	
	vmin=vmax=np.max(Image_Filtered_xx)/vm4
	
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered_xx[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_xxf.png')
	
	vmin=vmax=np.max(Image_Filtered_xx)/vm4
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered_zz[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/ImageT_zzf.png')

def plot_I_pp_total(address, nshots, out, vm, vm_N, vm_F):
	mat_content = spio.loadmat('../outputs/%s_0.mat'%(address))
	Image_pp    = copy.deepcopy(mat_content['Ima'])
	I_pp_t      = np.zeros((Image_pp.shape[0],Image_pp.shape[1]))
	I_pp_N_t    = np.zeros((Image_pp.shape[0],Image_pp.shape[1]))
	for i in range(nshots):
		mat_content = spio.loadmat('../outputs/%s_%i.mat'%(address,i))
		Image_pp    = copy.deepcopy(mat_content['Ima'])
		Image_pp_N  = copy.deepcopy(mat_content['Ima_N'])
		nbl         = mat_content['nbl'][0][0]
		shape=(Image_pp.shape[0]-2*nbl+1, Image_pp.shape[1]-2*nbl+1)
		print('Image shape',shape)
		I_pp_t[:,:]   += Image_pp
		I_pp_N_t[:,:] += Image_pp_N
		
	vmin=vmax=np.max(I_pp_t)/vm #1.769829199461798e-11 #np.max(Image_pp)/vm
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(I_pp_t[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_%s.png'%out)
	
	vmin_n=vmax_n=np.max(I_pp_t)/vm_N# 3.5396583989235955e-05 #np.max(Image_pp)/vm_N
	print('colorbar min_n', vmin_n)
	plt.figure(figsize=(8, 8))
	plt.imshow(I_pp_N_t[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin_n, vmax=vmax_n)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_N_%s.png'%out)
	s=2
	w=5
	t=(((w - 1)/2)-0.5)/s
	lowpass = ndimage.gaussian_filter(I_pp_t,sigma=s, truncate=t)
	Image_Filtered_pp = I_pp_t-lowpass
	
	vmin=vmax=np.max(Image_Filtered_pp)/vm_F#  1.3030108655839902e-11 #np.max(Image_Filtered_pp)/vm_F	
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered_pp[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
# 	plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
	plt.savefig('../fig/I_pp_F_%s.png'%out)

def plot_I_total(address, nshots, out, vm, vm_N, vm_F):
	mat_content = spio.loadmat('../outputs/%s_0.mat'%(address))
	Image       = copy.deepcopy(mat_content['Ima'])
	I_t      = np.zeros((Image.shape[0],Image.shape[1]))
	I_N_t    = np.zeros((Image.shape[0],Image.shape[1]))
	
	for i in range(nshots):
		mat_content = spio.loadmat('../outputs/%s_%i.mat'%(address,i))
		Image    = copy.deepcopy(mat_content['Ima'])
		Image_N  = copy.deepcopy(mat_content['Ima_N'])
		nbl         = mat_content['nbl'][0][0]
		shape=(Image.shape[0]-2*nbl+1, Image.shape[1]-2*nbl+1)
		print('Image shape',shape)
		I_t[:,:]   += Image
		I_N_t[:,:] += Image_N
		
	vmin=vmax=np.max(I_t)/vm 
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(I_t[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
	plt.title('I_%s'%out)
	plt.savefig('../fig/I_%s.png'%out)
	
	outdict                     = dict()
	outdict['Ima']              = I_t[:,:]
	spio.savemat('../outputs/Image_t_%s.mat'%(out), outdict) 
	
	###########
	
	vmin_n=vmax_n=np.max(I_t)/vm_N
	print('colorbar min_n', vmin_n)
	plt.figure(figsize=(8, 8))
	plt.imshow(I_N_t[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin_n, vmax=vmax_n)	
	plt.title('I_N_%s'%out)
	plt.savefig('../fig/I_N_%s.png'%out)
	
	outdict                     = dict()
	outdict['Ima']              = I_N_t
	spio.savemat('../outputs/Image_t_N_%s.mat'%(out), outdict)
	
	###########
	s=2
	w=5
	t=(((w - 1)/2)-0.5)/s
	lowpass = ndimage.gaussian_filter(I_t,sigma=s, truncate=t)
	Image_Filtered = I_t-lowpass
	
	vmin=vmax=np.max(Image_Filtered)/vm_F	
	plt.figure(figsize=(8, 8))
	plt.imshow(Image_Filtered[nbl:shape[0]+nbl, nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
	plt.title('I_F_%s'%out)
	plt.savefig('../fig/I_F_%s.png'%out)
	
	outdict                     = dict()
	outdict['Ima']              = Image_Filtered
	spio.savemat('../outputs/Image_t_F_%s.mat'%(out), outdict) 
	
	###########
	
def plot_I_laplace_t(address, nshots, out, vm1, vm2):
	mat_content = spio.loadmat('../outputs/%s_0.mat'%(address))
	Image_pp    = copy.deepcopy(mat_content['Ima'])
	I_pp_t      = np.zeros((Image_pp.shape[0],Image_pp.shape[1]))
	I_pp_N_t    = np.zeros((Image_pp.shape[0],Image_pp.shape[1]))
	
	for i in range (nshots):
		mat_content = spio.loadmat('../outputs/%s_%i.mat'%(address,i))
		Image_pp    = copy.deepcopy(mat_content['Ima'])
		Image_pp_N  = copy.deepcopy(mat_content['Ima_N'])
		nbl         = mat_content['nbl'][0][0]
		shape=(Image_pp.shape[0]-2*nbl+1, Image_pp.shape[1]-2*nbl+1)
		
		ck = signal.cspline2d(Image_pp, 8.0)
		derfilt = np.array([1.0, -2, 1.0], dtype=np.float32)
		deriv = (signal.sepfir2d(ck, derfilt, [1])+signal.sepfir2d(ck, [1], derfilt))
		print('Image shape',shape)
		I_pp_t[:,:]   += deriv
		
	vmin=vmax=np.max(deriv)/vm1
	print('colorbar min', vmin)
	plt.figure(figsize=(8, 8))
	plt.imshow(I_pp_t[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, aspect='auto', vmin=-vmin, vmax=vmax)	
	plt.savefig('../fig/I_laplace_%s.png'%out)	
	
	outdict                     = dict()
	outdict['Ima']              = I_pp_t
	spio.savemat('../outputs/Image_t_laplace_%s.mat'%(out), outdict) 
	