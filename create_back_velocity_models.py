## create and smooth velocity models to back-propagate 

import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from scipy import ndimage
from scipy import signal
from convenient_functions import *
__all__ = ['back_model']

def smooth_model(nbl, so, vp, vs, ro, dx, dz, name):
	new_vp    = vp.copy()
	new_vs    = vp.copy()
	new_ro    = ro.copy()
	#len
	zp_il = int(1.3//dz) #0.0013mm or 1.3mm
	
	new_vp = np.where(vp == 1.0,  1.5,    new_vp)##layer1
	new_vp = np.where(vp == 3.0,  1.5,    new_vp)##bone
	new_vp = np.where(vp == 1.54, 1.5,    new_vp)##tissu around the bone
	new_vp = np.where(vp == 1.75, 1.5,    new_vp)##Marrow
	new_vp[:, :zp_il] = 1.0 #len
	new_vp[int(vp.shape[0]/2)-1:int(vp.shape[0]/2), vp.shape[1]-1:vp.shape[1]]=2.750


	new_vs = np.where(vp == 1.0,   0.0, new_vs)##layer1
	new_vs = np.where(vp == 3.0,   0.0, new_vs)##bone #1.8
	new_vs = np.where(vp == 1.54,  0.0, new_vs)##tissu around the bone
	new_vs = np.where(vp == 1.75,  0.0, new_vs)##Marrow
	new_vs[int(vp.shape[0]/2)-1:int(vp.shape[0]/2), vp.shape[1]-1:vp.shape[1]]= 1.38


	shape = (vp.shape[0], vp.shape[1])
	origin = (0., 0.)
	spacing = (dx, dz)
	nbl=nbl
	so=so
	model  = ModelElastic(vp=new_vp, vs=new_vs, b=1./ro, origin=origin, shape=shape, spacing=spacing, space_order=so, nbl=nbl)

	outdict                         = dict()
	outdict['vP']                   = new_vp
	outdict['vS']                   = new_vs
	outdict['rho']                  = ro
	outdict['dx']                   = dx
	outdict['dz']                   = dz
	sio.savemat('../velocity_models_outputs/model_back_%s.mat'%name, outdict)
	
	return new_vp, new_vs, new_ro, model

#########################
def back_model(name_para):
	mat_para  = sio.loadmat(name_para)
	nshots    = mat_para['nshots'][0][0]
	tn        = mat_para['tn']   
	nbl       = mat_para['nbl'][0][0]
	so        = mat_para['so'][0][0]
	imodel         = mat_para['model'][0][0]
	so=int(so)
	t0=0.0
	print('model_constant ', imodel)
	
	address ='../../velocity_models_BONE/model_%i.mat'%imodel
	model, vp, vs, ro, dx, dz = load_velocity_model(so, nbl, address)

	new_vp, new_vs, new_ro, model_b = smooth_model(nbl, so, vp, vs, ro, dx, dz, name='%i_constant'%imodel)
	print ('model.critical_dt=', model_b.critical_dt)
	plot_model_velocity(model_b, new_vp, new_vs, new_ro, name='01_model_velocity_B_new')
	plot_model(model_b, name='01_model_parameter_B_new')
	
	name='back_model_para'
	plt.figure(figsize=(8, 8))
	shape = model.shape
	plt.imshow(model_b.lam.data[:-1,:-1].T)
	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
	plt.title('Lambda')
	plt.ylabel('nz')
	plt.xlabel('nx')
	# 	plt.colorbar(label='lambda', fraction=0.046, pad=0.04)
	plt.savefig('../fig/%s'%name)











