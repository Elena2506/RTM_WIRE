import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
__all__ = ['Curl', 'Diveregence', 'Derivate_dz', 'Derivate_dx']

##################
def Curl(nshots, dt, address, name_out):
	curl_VF=[]
	for i in range(nshots):
			mat_contents = sio.loadmat('../outputs/%s_%i.mat'%(address, i))
			v_x = copy.deepcopy(mat_contents['v_x'])
			v_z = copy.deepcopy(mat_contents['v_z'])
			print('Curl')
			for it in range(v_z.shape[0]):
# 				print ('Vz time %i out of %d' % (it+1, v_z.shape[0]))
				dVx_dz  = np.gradient(v_x[it,:-1,:-1], axis=1)#1)
				dVz_dx  = np.gradient(v_z[it,:-1,:-1], axis=0)#0)
				curl_V  = dVz_dx-dVx_dz
				curl_VF.append(curl_V)
				curl_VF_n=np.asanyarray(curl_VF)
			
			print('Saving shot', i, 'with shape', curl_VF_n.shape)
			outdict                     = dict()
			outdict['Curl_V']           = curl_VF_n[:,:,:]
			outdict['dt']               = dt
			sio.savemat('../outputs/%s_%i.mat'%(name_out,i), outdict)
			curl_VF.clear()


##################

def Diveregence(nshots, dt, address, name_out):
	div_Vf=[]
	for i in range(nshots):
			mat_contents = sio.loadmat('../outputs/%s_%i.mat'%(address,i))
			v_x = copy.deepcopy(mat_contents['v_x'])
			v_z = copy.deepcopy(mat_contents['v_z'])
			print('Divergence Vf')
			for it in range(v_z.shape[0]):
# 				print ('V time %i out of %d' % (it+1, v_z.shape[0]))
				dVx_dx = np.gradient(v_x[it,:-1,:-1], axis=0)#0)
				dVz_dz = np.gradient(v_z[it,:-1,:-1], axis=1)#1)
				div_V  = dVx_dx + dVz_dz
				div_Vf.append(div_V)
				div_Vf_n=np.asanyarray(div_Vf)
			print('Saving shot', i, 'with shape', div_Vf_n.shape)
			outdict                     = dict()
			outdict['div_V']           = div_Vf_n[:,:,:]
			outdict['dt']               = dt
			sio.savemat('../outputs/%s_%i.mat'%(name_out,i), outdict)
			div_Vf.clear()
		
	
def Derivate_dz(nshots, address, name_out):
	div_Vf=[]
	for i in range(nshots):
			mat_contents = sio.loadmat('outputs/%s_%i.mat'%(address,i))
			v_z = mat_contents['v_z']
			print('Derivative start')
			for it in range(v_z.shape[0]):
				print ('V time %i out of %d' % (it+1, v_z.shape[0]))
				dVz_dz = np.gradient(v_z[it,:-1,:-1], axis=1)
				div_Vf.append(dVz_dz)
				div_Vf_n=np.asanyarray(div_Vf)
			
	print('Saving shot', i, 'with shape', div_Vf_n.shape)
	outdict                     = dict()
	outdict['dVz_dz']           = div_Vf_n[:,:,:]
	sio.savemat('outputs/%s_%i.mat'%(name_out,i), outdict)
	
def Derivate_dx(nshots, address, name_out):
	div_Vf=[]
	for i in range(nshots):
			mat_contents = sio.loadmat('outputs/%s_%i.mat'%(address,i))
			v_x = mat_contents['v_x']
			print('Derivative start')
			for it in range(v_x.shape[0]):
				print ('V time %i out of %d' % (it+1, v_x.shape[0]))
				dVz_dx = np.gradient(v_x[it,:-1,:-1], axis=0)
				div_Vf.append(dVx_dx)
				div_Vf_n=np.asanyarray(div_Vf)
			
	print('Saving shot', i, 'with shape', div_Vf_n.shape)
	outdict                     = dict()
	outdict['dVx_dx']           = div_Vf_n[:,:,:]
	sio.savemat('outputs/%s_%i.mat'%(name_out,i), outdict)
	
