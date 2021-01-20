import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
from scipy import signal
from convenient_functions import *
from scipy import ndimage


__all__ = ['main_backward']

def backward_elastic(model, so, time_axis, rec_vz_data, rec_vz_coordinates):
    clear_cache()
    x, z  = model.grid.dimensions
    t     = model.grid.stepping_dim
    time  = model.grid.time_dim
    s     = time.spacing
    save  = time_axis.num
    nbl   = model.nbl
    shape = model.shape
    factor = 10
    time=model.grid.time_dim
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    nt = (time_axis.num) 
    save_n = (nt+factor-1)//factor
    print('nt=', nt,' save_n=', save_n)
    
    
    v    = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=2)#, save=save)
    tau  = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=2)#, save=save)
    vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=so, time_order=2, save=save_n, time_dim=time_subsampled)
    
    source_vz    = PointSource(name='source_vz', grid=model.grid, time_range=time_axis, coordinates=rec_vz_coordinates, data=rec_vz_data)
    src_in_vz    = source_vz.inject(field=v.backward[1], expr=s*source_vz)    
    
    src_in_expre = (src_in_vz)

    l, mu, ro = model.lam, model.mu, model.b

    u_v = Eq(v.backward, model.damp *  (v + s*ro*div(tau)))
    u_t = Eq(tau.backward,  model.damp *  (tau + s * (l * diag(div(v.backward)) + mu * (grad(v.backward) + grad(v.backward).T))))
 
    op = Operator([u_v] + [u_t] + [Eq(vsave, v)] + src_in_expre, save=True)
    op(dt=model.critical_dt)
    
    return vsave, tau, save_n    
    
# ###############

def main_backward(name_para, shiftSrc, plot=False):
	mat_para  = sio.loadmat(name_para)
	nshots    = mat_para['nshots'][0][0]
	tn        = mat_para['tn'][0][0]   
	nbl       = mat_para['nbl'][0][0]
	so        = mat_para['so'][0][0]
	i         = mat_para['model'][0][0]
	
	so=int(so)
	t0=0.0

	address ='../velocity_models_outputs/model_back_%i_constant.mat'%i
	model_b, vp, vs, ro, dx, dz = load_velocity_model(so, nbl, address)
	plot_model(model_b, name='model_parameters_backward')   
	plot_model_velocity(model_b, vp, vs, ro, name='model_velocity_backward')

	dt = model_b.critical_dt 
	time_axis = TimeAxis(start=t0, stop=tn, step=dt)
	print('dt =', dt)
	print('tn =', tn)
	print('n_time =', time_axis.num)
	shape=model_b.shape
	
	for i in range(nshots):
		mat_contents = sio.loadmat('../outputs/shots_T_inj/rec_vz_%i.mat'%i)
		rec_vz_data        = copy.deepcopy(mat_contents['rec_vz_data'])
		mat_contents2 = sio.loadmat('../outputs/shots_M/rec_coord_%i.mat'%i)
		rec_vz_coordinates = copy.deepcopy(mat_contents2['rec_vz_coordinates'])
		nt_s =rec_vz_data.shape[0]
		
		print ('nt are different ?', rec_vz_data.shape[0], '!=', time_axis.num , 'is so, start to resample the shotgather to inject')

		if nt_s != time_axis.num:
		
			resample_rec_vz = np.zeros((time_axis.num, rec_vz_data.shape[1]))
			print ('bacuase nt are different', rec_vz_data.shape[0], '!=', time_axis.num , 'it starts to resample the shotgather to inject')

			for ii in range (int(rec_vz_data.shape[1])):
				sig = signal.resample(rec_vz_data[:,ii],time_axis.num) 
				resample_rec_vz[:,ii]=sig
			plot_simple_shotgaher(resample_rec_vz, 'shot_resampled_%i'%i)
			rec_vz_data_s = shift_shotgather(resample_rec_vz, shiftSrc)
			plot_simple_shotgaher(rec_vz_data_s, 'shot_test_%i_%i'%(i,shiftSrc))
			vb, _, save_n = backward_elastic(model=model_b, so=so, time_axis=time_axis, rec_vz_data=rec_vz_data_s, rec_vz_coordinates=rec_vz_coordinates)

			outdict                 = dict()
			outdict['v_z']           = vb[1].data[:,:,:]
			outdict['v_x']           = vb[0].data[:,:,:]
			sio.savemat('../outputs/wavefield/Wavefield_Velocity_b_%i.mat'%(i), outdict)

		
# 		if plot:
# 			print('Start plotting vz')
# 			long2=int(vb[1].shape[0])
# 			vmin=vmax=np.max(vb[1].data)/1000
# 			print(vmin)
# 			for it in range (1, long2, 10):
# 				plt.figure()
# 				plt.title('time_%i'%it)
# # 				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
# 				plt.imshow(vb[1].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
# 				plt.colorbar()
# 				plt.savefig('../fig/back_vz/vz_backward_%i.png'%(it))
# 				plt.clf()
# 
# 			print('Start plotting vx')
# 			long2=int(vb[1].shape[0])
# 			vmin=vmax=np.max(vb[0].data)/1000
# 			print(vmin)
# 			for it in range (1, long2, 10):
# 				plt.figure()
# 				plt.title('time_%i'%it)
# # 				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
# 				plt.imshow(vb[0].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
# 				plt.colorbar()
# 				plt.savefig('../fig/back_vx/vx_backward_%i.png'%(it))
# 				plt.clf()
# 					
# 		print('Finish Backward') 	
					
# 		else: 
# 			print('Everything is rigth NO RESAMPLING')
# 			plot_simple_shotgaher(rec_vz_data, 'shot2back_%i'%i)
# 			vb, _, save_n = backward_elastic(model=model_b, so=so, time_axis=time_axis, rec_vz_data=rec_vz_data, rec_vz_coordinates=rec_vz_coordinates)
# 			outdict                 = dict()
# 			outdict['v_z']           = vb[1].data[:,:,:]
# 			outdict['v_x']           = vb[0].data[:,:,:]
# 			sio.savemat('../outputs/wavefield/Wavefield_Velocity_b_%i.mat'%i, outdict)
# 
# 		
# 			if plot:
# 				print('Start plotting vz')
# 				long2=int(vb[1].shape[0])
# 				vmin=vmax=np.max(vb[1].data)/1000
# 				print(vmin)
# 				for it in range (1, long2, 10):
# 					plt.figure()
# 					plt.title('time_%i'%it)
# # 					plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
# 					plt.imshow(vb[1].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
# 					plt.colorbar()
# 					plt.savefig('../fig/back_vz/vz_backward_%i.png'%(it))
# 					plt.clf()
# 					
# 				print('Start plotting vx')
# 				long2=int(vb[1].shape[0])
# 				vmin=vmax=np.max(vb[0].data)/1000
# 				print(vmin)
# 				for it in range (1, long2, 10):
# 					plt.figure()
# 					plt.title('time_%i'%it)
# # 					plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
# 					plt.imshow(vb[0].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
# 					plt.colorbar()
# 					plt.savefig('../fig/back_vx/vx_backward_%i.png'%(it))
# 					plt.clf()
# 					
# 		print('Finish Backward')
#  	
	    	




















