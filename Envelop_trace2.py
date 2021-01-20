## hilbert transform to take the max amplitude value of the envelop 

import numpy as np
import matplotlib.pylab as plt

import scipy.io as spio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
from scipy import signal, misc


__all__ = ['envelop_trace']

def envelop_trace(address, nbl, vn, out,  plot=False, plot2=False):
	mat_content = spio.loadmat('../output/%s.mat'%address)
	Ima      = copy.deepcopy(mat_content['Ima'])
	print('Image shape', Ima.shape)
	
	Ima[:, 0:400]=0
	
	
	
	if plot:
		plt.figure(figsize=(8, 8))
		plt.imshow(Ima[nbl:Ima.shape[0]-nbl, nbl:Ima.shape[1]-nbl].T, cmap='seismic',  interpolation='nearest', vmin=-vn , vmax=vn)
# 		plt.tight_layout()
# 		plt.colorbar()
		plt.savefig('../fig/Enve_input_%s.png'%out)
		
	imagine_envelop   = np.zeros((Ima.shape[0],Ima.shape[1]))
	print(imagine_envelop.shape)
	for inx in range(Ima.shape[0]):
		ima_hilbert = (np.abs(signal.hilbert(Ima[inx, :].T)))
		imagine_envelop[inx,:] = ima_hilbert 
	vm=np.max(imagine_envelop[nbl:Ima.shape[0]-nbl, nbl:Ima.shape[1]-nbl])/vn	
	
	s=2
	w=5
	t=(((w - 1)/2)-0.5)/s
	lowpass = ndimage.gaussian_filter(imagine_envelop,sigma=s, truncate=t)
	Image_Filtered = imagine_envelop-lowpass
	
	
	plt.figure(figsize=(8, 8))
	plt.imshow(imagine_envelop[nbl:Ima.shape[0]-nbl, nbl:Ima.shape[1]-nbl].T, vmin=-vn , vmax=vn, cmap='Greys',  interpolation='nearest')
# 	plt.tight_layout()
	plt.savefig('../fig/Envelop_Image_%s.png'%out)
	
	##plot trace 
	if plot2: 
		for inx in range(nbl, Ima.shape[0]-nbl+1):
			plt.figure(figsize=(8, 8))
			plt.plot(Ima[inx, nbl:(Ima.shape[0]-nbl+1)].T, label='%s'%address)
			plt.plot((np.abs(signal.hilbert(Ima[inx, 150:350].T))), '--', label='Hilbert_%s'%address)
			plt.ylim([-0.00005, 0.00005])
			plt.plot(100,0,  '*k' , label='pixel location')
			plt.legend(loc='best', shadow=True)
			plt.ylabel('Amplitude')
			plt.xlabel('nz')
			plt.tight_layout()
			plt.savefig('../fig/traces/trace_%s.png'%inx)

	

