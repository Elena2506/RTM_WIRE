## convenient functions that are use along...

import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
from collections import deque

__all__ = ['load_velocity_model','plot_single_model', 'plot_model', 'plot_model_velocity', 'plot_shotrecord2', 'plot_simple_shotgaher', 'plot_slice_wavefield', 'shift_shotgather']

def load_velocity_model(so, nbl, address):

    mat_contents = sio.loadmat(address)
    vp = copy.deepcopy(mat_contents['vP'])
    vs = copy.deepcopy(mat_contents['vS'])
    ro = copy.deepcopy(mat_contents['rho'])
    dx =  mat_contents['dx'][0][0]
    dz =  mat_contents['dz'][0][0]
    print('np.max(vp),np.max(vs)',np.max(vp),np.max(vs))
    print('dx', dx, 'dz', dz)
    print ('dx_model=dz_model=', mat_contents['dx'][0][0], 'dz_model=',mat_contents['dz'][0][0],)
    shape = (vp.shape[0], vp.shape[1])   
    print('shape', shape)            
    spacing = (dx, dz) 
    print('spacing=', spacing)  
    origin = (0., 0.)
    nbl=nbl
    so=so
    model  = ModelElastic(vp=vp, vs=vs, b=1./ro, origin=origin, shape=shape, spacing=spacing, space_order=so, nbl=nbl)
    print(model.critical_dt)
    return model, vp, vs, ro, dx, dz
     
def plot_model(model, name):
    nbl=model.nbl
    shape=model.shape
    plt.ion()
    plt.figure(figsize=(18,9))
    plt.subplot(1,3,1)
    plt.imshow(model.lam.data[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T)
#     plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
    plt.title('Lambda')
    plt.ylabel('nz')
    plt.xlabel('nx')
    plt.colorbar(label='Lambda', fraction=0.046, pad=0.04)
    
    plt.subplot(1,3,2)
    plt.imshow(model.mu.data[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T)
#     plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
    plt.title('Mu')
    plt.ylabel('nz')
    plt.xlabel('nx')
    plt.colorbar(label='mu',fraction=0.046, pad=0.04)
    
    plt.subplot(1,3,3)
#     plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
    plt.imshow(model.b.data[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T)
    plt.ylabel('nz')
    plt.xlabel('nx')        
    plt.colorbar(label='',fraction=0.046, pad=0.04)
    plt.title('1/rho')
    plt.tight_layout()
    plt.savefig('../fig/%s.png'%name)
    
def plot_model_velocity(model, vp, vs, ro, name):
    nbl=model.nbl
    shape=model.shape
    plt.ion()
    plt.figure(figsize=(18,9))
    plt.subplot(1,3,1)
    plt.imshow(vp.T)
    plt.title('Vp')
    plt.ylabel('nz')
    plt.xlabel('nx')
    plt.colorbar(label='Vp km/(s)', fraction=0.046, pad=0.04)
    
    plt.subplot(1,3,2)
    plt.imshow(vs.T)
    plt.title('Vs')
    plt.ylabel('nz')
    plt.xlabel('nx')
    plt.colorbar(label='Vs km/(s)',fraction=0.046, pad=0.04)
    
    plt.subplot(1,3,3)
    plt.imshow(ro.T)
    plt.ylabel('nz')
    plt.xlabel('nx')        
    plt.colorbar(label='rho (kg/m3)',fraction=0.046, pad=0.04)
    plt.title('Rho')
    plt.tight_layout()
    plt.savefig('../fig/%s.png'%name) 
    
def plot_single_model(model, name):
	plt.figure(figsize=(8, 8))
	shape = model.shape
	nbl= model.nbl
	plt.imshow(model.lam.data[nbl:shape[0]+nbl,nbl:shape[0]+nbl].T)
# 	plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
	plt.title('Lambda')
	plt.ylabel('nz')
	plt.xlabel('nx')
# 	plt.colorbar(label='lambda', fraction=0.046, pad=0.04)
	plt.savefig('../fig/%s'%name)
	
def plot_shotrecord2(rec, model, t0, tn, name, colorbar=True):
    
    scale =  np.max(rec)/ 100.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0], 1e-3*tn, t0]
    
    plt.figure()
    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.savefig('../fig/%s.png'%(name))	
    
def plot_simple_shotgaher(arr, name_output):
	scale =  np.max(arr)/ 100.
	plt.figure()
	im=plt.imshow(arr, cmap=cm.gray, aspect='auto', vmin=-scale, vmax=scale)
	plt.xlabel('receiver')
	plt.ylabel('n time')
# 	plt.colorbar()	
	plt.savefig('../fig/%s.png'%(name_output))
	
def plot_slice_wavefield(nshots, nbl, address):
	for i in range(nshots):
			mat_contents = sio.loadmat('output/%s_%i.mat'%(address,i))
			v_z = mat_contents['v_z']	
			long2=int(v_z.shape[0])
			print('n_time in file', long2, 'shape of fine', v_z.shape[1])
			shape=v_z.shape[1]-nbl
			vmin=vmax=np.max(v_z[long2//2,nbl:shape,nbl:shape])/10
			print(vmin)
# 			print(v_z[long2//2,nbl:shape,nbl:shape])
			for it in range (1, long2, 10):
					plt.figure()
					plt.title('time_%i'%it)
# 					plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
					plt.imshow(v_z[it,nbl:shape,nbl:shape].T, cmap='seismic', vmin=-vmin, vmax=vmax)
					plt.colorbar()
# 					plt.savefig('fig/forward/vz_forward_%i.png'%(it))
					plt.savefig('../fig/back/vz_backward_%i.png'%(it))		
	
def shift_shotgather(rec_vz_data, shiftSrc):
	nr =rec_vz_data.shape[1]
	nt =rec_vz_data.shape[0]
	print ('number of receivers', nr, 'number time', nt)
	print ('shiftSrc', shiftSrc)
	new_rec_vz = np.zeros((nt, nr))
	for i in range (nr):
		rec_vz_shift = deque(rec_vz_data[:,i])
		rec_vz_shift.rotate(-shiftSrc)
		new_rec_vz[:,i]  = rec_vz_shift
	nt_S =rec_vz_data.shape[0]-shiftSrc
	print(nt_S)
	new_rec_vz[nt_S:, :]=0.0
	print ('new_rec_vz.shape shift_test', new_rec_vz.shape)
	print ('-shiftSrc', -shiftSrc)
	return new_rec_vz
	
	