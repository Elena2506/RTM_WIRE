## Create pixel velocity model 

import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
from skimage.draw import circle_perimeter, disk ##python -m pip install -U scikit-image
from convenient_functions import *

__all__ = ['model_velocity', 'Forward_model']


def model_velocity (nbl, so, vp, vs, ro, dx, dz, name):
	new_vp    = vp.copy()
	new_vs    = vp.copy()
	new_ro    = ro.copy()
	xp_i = int(vp.shape[0]-50)
	xp_f = int(vp.shape[0]-40)
	zp_i = vp.shape[1]//4-5
	zp_f = vp.shape[1]//4
	#len
	zp_il = 26#vp.shape[1]//4-5
	zp_fl = vp.shape[1]//4-4

	new_vp = np.where(vp == 1.0,  1.5,    new_vp)##layer1
	new_vp = np.where(vp == 3.0,  1.5,    new_vp)##bone
	new_vp = np.where(vp == 1.54, 1.5,    new_vp)##tissu around the bone
	new_vp = np.where(vp == 1.75, 1.5,    new_vp)##Marrow
	new_vp[:, :zp_il] = 1.0 #len
	new_vp[xp_i:xp_f, zp_i:zp_f]= 2.750 #point
	

	
# 	rr, cc = disk((new_vp.shape[0]-3, new_vp.shape[1]//2), 2)
# 	new_vp[rr, cc]= 2.750
# 	rr, cc = disk((new_vp.shape[0]//2, new_vp.shape[1]//2), 80)
# 	new_vp[rr, cc]= 1.5
	
# 	new_vp[int(vp.shape[0]/2)-1:int(vp.shape[0]/2), vp.shape[1]-1:vp.shape[1]]=2.750
# 	new_vp = ndimage.gaussian_filter(new_vp, sigma=4, order=0, mode='nearest')
# 	new_vp[int(vp.shape[0]/2)-1:int(vp.shape[0]/2), vp.shape[1]-1:vp.shape[1]] = 1.75


	new_vs = np.where(vp == 1.0,   0.0, new_vs)##layer1
	new_vs = np.where(vp == 3.0,   0.0, new_vs)##bone #1.8
	new_vs = np.where(vp == 1.54,  0.0, new_vs)##tissu around the bone
	new_vs = np.where(vp == 1.75,  0.0, new_vs)##Marrow
	new_vs[xp_i:xp_f, zp_i:zp_f]= 1.38
# 	rr, cc = disk((new_vs.shape[0]-3, new_vs.shape[1]//2), 2)
# 	new_vs[rr, cc]= 1.38
# 	rr, cc = disk((new_vs.shape[0]//2, new_vs.shape[1]//2), 80)
# 	new_vs[rr, cc]= 0.0	
# 	new_vs[int(vp.shape[0]/2)-1:int(vp.shape[0]/2), vp.shape[1]-1:vp.shape[1]]= 1.38

	print ('point location pixel in x',xp_i,xp_f,'in z', zp_i,zp_f )
	print ('point location mm in x',xp_i*dx, xp_f*dx,'in z',  zp_i*dz, zp_f*dz )
	

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
	sio.savemat('../../velocity_models_T/model_forward_%s.mat'%name, outdict)
	return new_vp, new_vs, new_ro, model
	
def Forward_model(nbl, so, imodel):
	print('creating model_resolution ', imodel, nbl, so)
	address ='../../velocity_models/model_%i.mat'%imodel
	model, vp, vs, ro, dx, dz = load_velocity_model(so, nbl, address)
# 	print ('model.critical_dt=', model.critical_dt)
# 	plot_model(model, name='model_parameter_start')
# 	plot_model_velocity(model, vp, vs, ro, name='model_velocity_start')
	new_vp, new_vs, new_ro, model_new = model_velocity(nbl, so, vp, vs, ro, dx, dz, name='%i'%imodel)
	print ('model.critical_dt=', model_new.critical_dt)
	plot_model_velocity(model_new, new_vp, new_vs, new_ro, name='01_model_velocity_F_new')
	plot_model(model_new, name='01_model_parameter_F_new')
	print('np.max(new_vp)',np.max(new_vp), 'np.max(new_vs)',np.max(new_vs))
# 	
# 	name='Forward_model_para'
# 	plt.figure(figsize=(8, 8))
# 	shape = model.shape
# 	plt.imshow(model_new.lam.data[:-1,:-1].T)
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
# 	plt.title('Lambda')
# 	plt.ylabel('nz')
# 	plt.xlabel('nx')
# 	# 	plt.colorbar(label='lambda', fraction=0.046, pad=0.04)
# 	plt.savefig('../fig/%s'%name)

## create a circle 
# def circle_points(r, n):
#     circles = []
#     for r, n in zip(r, n):
#         t = np.linspace(0, np.pi, n, endpoint=False)
#         x = r * np.cos(t)
#         y = r * np.sin(t)
#         circles.append(np.c_[x, y])
#     return circles
#     
# r = [0, 0.1, 0.2]
# n = [1, 10, 20]
# circles = circle_points(r, n)
# 
# fig, ax = plt.subplots()
# for circle in circles:
#     ax.scatter(circle[:, 0], circle[:, 1])
# ax.set_aspect('equal')
# plt.savefig('../fig/circle_test')	