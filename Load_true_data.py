## Loading the true data 

import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *



__all__ = ['load_true_data']

def load_true_data(address):
	mat_contents = sio.loadmat(address)
# 	print(mat_contents)
	SIG   				       = mat_contents['SIG']
	print('SIG.shape',SIG.shape)
	Fs                         = mat_contents['Fs'] #temporal_sampling_freq
	print('SIG.shape[0]/Fs',SIG.shape[0]*(1./Fs))
	print('Fs', Fs)
	pitch       = mat_contents['pitch']  	
	dt = 1./Fs #Time sampling in seconds
	t  = np.arange(start=1, stop=SIG.shape[0], step=1)*dt*1e6 #time in microseconds seconds
# 	print(t)
# 	offset = np.arange(start=1, stop=SIG.shape[1], step=1)*spatial_period_array*1e3  #offset in mm
# 	extent=[0,np.max(offset),np.max(t), 0]
# 	
	for ishot in range(2):#(SIG.shape[2]):
		print('shot # %i'%ishot, 'out of ', SIG.shape[2]) 
		scale =  np.max(SIG)/ 100.
		plt.imshow(SIG[:,:,ishot], aspect='auto', cmap=cm.gray, vmin=-scale, vmax=scale)#, extent=extent)
		plt.title('shot # %i'%ishot)
		plt.ylabel('n time') #('Time (us)')
		plt.xlabel('receiver')#('Offset (mm)')
		plt.savefig('../fig/shots/raw_data_%i'%ishot)
		plt.clf()
		
		outdict                         = dict()
		outdict['rec_vz_data']          = SIG[:,:,ishot]
# 		outdict['offset']               = np.max(offset)
# 		outdict['spatial_period_array'] = spatial_period_array[0][0]
		outdict['t_max']                = np.max(t)
		outdict['dt']                   = dt
		sio.savemat('../outputs/shots_T/rec_vz_%i.mat'%ishot, outdict)
		
		
# 	plt.imshow(SIG[:,:,ishot], aspect='auto', cmap=cm.gray, vmin=-scale, vmax=scale)
# 	plt.ylabel('nt')
# 	plt.xlabel('nr')
# 	plt.savefig('../fig/raw_data_%i'%ishot)
# 	plt.clf()
	print('number of receiver = ',SIG.shape[1], 'number of time_steps = ', SIG.shape[0], 'number of shots= ', SIG.shape[2])
# 	print('Distance in x direction', np.max(offset),'in mm', 'with a spatial period', spatial_period_array[0][0], 'in m or', spatial_period_array[0][0]*1e3, 'in mm')
	print('Total time record', np.max(t), 'in microseconds', 'using a sample rate = ', dt[0][0], 'in seconds or ', dt*1e6, 'in microseconds')
	return np.max(t)
	
####


