import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
# from examples.seismic.elastic import *
from scipy import ndimage
from convenient_functions import *
# import h5py
# import hdf5storage

__all__ = ['main_forward', 'plot_simple_shotgaher']


def forward_elastic(model, time_axis, so, source_locations,f0 , i):

    x, z  = model.grid.dimensions
    s     = model.grid.time_dim.spacing
    nbl   = model.nbl
    shape = model.shape
    factor = 10
    time=model.grid.time_dim
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    nt = (time_axis.num) 
    save_n = (nt+factor-1)//factor
    print('nt=', nt,' save_n=', save_n)
    v     = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=2)#, save=nt)#, save=Buffer(a))
    tau   = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=2)#, save=nt)#, save=Buffer(nt))
    vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=so, time_order=2, save=save_n, time_dim=time_subsampled)

    dt    = time_axis.step
    dx    = int(model.spacing[0])
    dz    = int(model.spacing[1])
        
    # The source injection term
    src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_axis)
    src.coordinates.data[:]  = source_locations[i, :]
    src_xx = src.inject(field=tau.forward[0, 0], expr=s*src)
    src_zz = src.inject(field=tau.forward[1, 1], expr=s*src)
    shiftSrc = np.argmax(src.data)

    # The receivers
    nreceivers = 96 #int(20*model.domain_size[0]) #
    print('nreceivers', nreceivers)
    rec_coordinates = np.empty((nreceivers, 2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
    rec_coordinates[:, 1] = 2*model.spacing[1]
    
    rec = Receiver(name="rec", grid=model.grid, npoint=nreceivers, time_range=time_axis)
    rec.coordinates.data[:] = rec_coordinates[:]
    
    rec_term = rec.interpolate(expr=v[1])
    
    l, mu, ro = model.lam, model.mu, model.b

    u_v = Eq(v.forward,    model.damp *  (v + s*ro*div(tau)))
    u_t = Eq(tau.forward,  model.damp *  (tau + s * (l * diag(div(v.forward)) + mu * (grad(v.forward) + grad(v.forward).T))))
    
    op = Operator([u_v] + [u_t] + [Eq(vsave, v)] + src_xx + src_zz + rec_term, save=True)# 
    
    op(dt=model.critical_dt)
    
    return rec, vsave, shiftSrc, src.data, save_n    #tau
    
      
def main_forward(tn, so, nbl, imodel, nshots, plot=False):
	so=so
	nbl=nbl

	address ='../velocity_models_outputs/model_forward_C_%i.mat'%imodel
	model, vp, vs, ro, _, _ = load_velocity_model(so, nbl, address)
	plot_model(model, name='02_model_parameter_2_F')   
	plot_model_velocity(model, vp, vs, ro, name='02_model_velocity_2_F')
	print ('Starting')
	

	tn=tn
	t0=0.0
	f0 = 2.5#2.5e+6/1000  ##2.5MHz 2500kHz
	print('frequency in kHz=',f0)
	print('frequency in Hz=',f0*1000)
	name_model=f0
	
	source_locations = np.empty((nshots, 2), dtype=np.float32)
	source_locations[:, 0]  = np.linspace(0., model.domain_size[0], num=nshots) # model.domain_size[0]/2
	source_locations[:, -1] = model.spacing[1]
	
	dt = model.critical_dt
	print('dt =', dt)
	time_axis = TimeAxis(start=t0, stop=tn, step=dt)
	print('n_time =', time_axis.num)
	
# 	outdict                         = dict()
# 	outdict['nbl']                  = nbl
# 	outdict['tn']                   = tn
# 	outdict['nshots']               = nshots
# 	outdict['so']                   = so 
# 	outdict['model']                = imodel
# 	outdict['dt']                   = dt
# 	sio.savemat('../outputs/parameter_%i.mat'%imodel, outdict)
	
	nbl=model.nbl
	shape=model.shape
	div_Vf=[]
	for i in range(nshots):
		print('Imaging source %d out of %d' % (i+1, nshots))
		rec_vz, vf, shiftSrc, src_data, save_n = forward_elastic(model, time_axis, so, source_locations, f0, i)
		print('vf[1].data.shape =', vf[1].data.shape)
		print('shiftSrc', shiftSrc)
		plot_shotrecord2(rec_vz.data,  model, t0, tn, name='rec_true_bone_%i_%i'%(i, imodel))
		plot_simple_shotgaher(arr=rec_vz.data, name_output='rec_true_bone_vz_%i_%2f'%(imodel, name_model))
		
		outdict                         = dict()
		outdict['rec_vz_data']          = rec_vz.data
		outdict['rec_vz_coordinates']   = rec_vz.coordinates_data
		outdict['shiftSrc']             = shiftSrc
		outdict['src_data']             = src_data
		sio.savemat('../outputs/shots_M/rec_vz_%i.mat'%i, outdict)
		
		outdict                         = dict()
		outdict['rec_vz_coordinates']   = rec_vz.coordinates_data
		outdict['shiftSrc']             = shiftSrc
		sio.savemat('../outputs/shots_M/rec_coord_%i.mat'%i, outdict)
		
		outdict                 = dict()
		outdict['v_z']           = vf[1].data[:,:,:]
		outdict['v_x']           = vf[0].data[:,:,:]
		sio.savemat('../outputs/wavefield/Wavefield_Velocity_f_%i.mat'%i, outdict)
		 
		
	if plot:
		long2=int(vf[1].shape[0])
		print ('plotting ', long2, 'snapshots')
		vmin=vmax=np.max(vf[1].data)/1000
		print('Start plotting')
		for it in range (1, long2, 10):
			plt.figure()
			plt.title('time_%i'%it)
			plt.imshow(vf[1].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
			plt.colorbar()
			plt.savefig('../fig/forward_vz/vz_forward_%i.png'%(it))
			plt.clf()
			
		print ('plotting ', long2, 'snapshots')
		vmin=vmax=np.max(vf[0].data)/1000
		print('Start plotting')
		for it in range (1, long2, 10):
			plt.figure()
			plt.title('time_%i'%it)
			plt.imshow(vf[0].data[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-vmin, vmax=vmax)
			plt.colorbar()
			plt.savefig('../fig/forward_vx/vx_forward_%i.png'%(it))
			plt.clf()
			
	
	outdict                         = dict()
	outdict['nbl']                  = nbl
	outdict['tn']                   = tn
	outdict['nshots']               = nshots
	outdict['so']                   = so 
	outdict['model']                = imodel
	outdict['dt']                   = dt
	outdict['shiftSrc']             = shiftSrc
	sio.savemat('../outputs/parameter_%i.mat'%imodel, outdict) 
	print('Finish Forward')





