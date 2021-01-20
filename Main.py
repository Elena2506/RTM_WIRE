import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage
import skimage  ##python -m pip install -U scikit-image 
from Load_true_data import load_true_data
from create_initial_velocity_model_constant import Forward_model
from convenient_functions import *
from Forward_ import main_forward, plot_simple_shotgaher
from create_back_velocity_models import back_model
from Backward_resampling import main_backward
from Curl_Devergence import *
from Imaging import image_pp, image_mixed, image_ss
from plot_image_mixed import plot_I_pp, plot_mixed, plot_I_ss, plot_I_laplace_t, plot_I_pp_total, plot_I_total
from trace_comparison import trace_comparison
from Envelop_trace2 import envelop_trace


#######################
##load true data
address= '../../PMMA_raw_data/wire_in_water_7mm_depth_right_side_P41.mat'
tn = load_true_data(address)
print(tn)
# 	shiftSrc_list     = [0, 5, 10, 20, 30]
# 	shift = 0

#######################
##create initial velocity model 
so=4
nbl=150
nshots=3
list_model = [500] #2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
tn=tn
shiftSrc = 100 # in the figures names + represent the negatives number here i.e. -5, -25,-10 etc 
for imodel in list_model:
	#. creating the initial velocity PPMM model 
# 	Forward_model(nbl, so, imodel)
# 	print('model with number nx and nz of pixels = ', imodel)
	##1. Forward 
# 	main_forward(tn, so, nbl, imodel, nshots, plot=True)
	## 2 create constant velocity model for back-propagation
# 	back_model(name_para='../outputs/parameter_%i.mat'%imodel)
	#3. Backward
	main_backward(name_para = '../outputs/parameter_%i.mat'%imodel, shiftSrc=shiftSrc, plot=False)
	
	###############################################
	mat_para  = sio.loadmat('../outputs/parameter_%i.mat'%imodel)
	dt        = mat_para['dt'][0][0]
	print(nshots, dt, tn)

	################################################
	################# Ipp ###########################
	
	#. Divergence VF and VB
	address= 'wavefield/Wavefield_Velocity_f'
	name_out = 'D_F'
	Diveregence(nshots, dt, address, name_out)
	
	address= 'wavefield/Wavefield_Velocity_b'
	name_out = 'D_B'
	Diveregence(nshots, dt, address, name_out)
	
	##.. Image condition pp
	image_pp(nbl, nshots, dt, imodel, plot_b=False, plot_f=False, plot_ima=False)
	
	##.. Plot Image condition pp
	address='Image_pp'
	out='pp_%i'%imodel
	vm=1000
	vm_N=0.0001
	vm_F=20000
	plot_I_total(address, nshots, out, vm, vm_N, vm_F)
 	
	################################################
	##################Iss#######################
	address= 'wavefield/Wavefield_Velocity_f'
	name_out = 'C_F'
	Curl(nshots, dt, address, name_out)
	
	address= 'wavefield/Wavefield_Velocity_b'
	name_out = 'C_B'
	Curl(nshots, dt, address, name_out)

	## .. Image condition ss 	
	image_ss(nbl, nshots, dt, imodel, plot_b=False, plot_f=False, plot_ima=False)

	address='Image_ss'
	out='ss_%i'%imodel
	vm=10000
	vm_N=0.0001
	vm_F=10000000
	plot_I_total(address, nshots, out, vm, vm_N, vm_F)
	
	###############################################
	###############################################
	# .. Plotting and Laplace filter
	## Change the names inside function to compare!!! 
	
# 	address='Image_pp'
# 	out='pp_%i'%imodel
# 	vm1=20000
# 	vm2=10
# 	plot_I_laplace_t(address,nshots, out, vm1, vm2)
# 	
# 	address='Image_ss'
# 	out='ss_%i'%imodel
# 	vm1=200000
# 	vm2=10
# 	plot_I_laplace_t(address,nshots, out, vm1, vm2)

	################################################
	################################################
	# .. Plot using the max amplitude obtained by the Hilbert transform 
	
# 	address='Image_t_pp_%i'%imodel
# 	vn=1000
# 	out='pp'
# 	envelop_trace(address, nbl, vn, out, plot=True)
	
# 	address='Image_t_ss_%i'%imodel
# 	vn=50000
# 	out='ss'
# 	envelop_trace(address, nbl, vn, out, plot=False)
#  		
# 	address='Image_t_F_pp_%i'%imodel
# 	vn=0.00000000000000001
# 	vm2 = 0.00000000000000001
# 	out='F_pp'
# 	envelop_trace(address, nbl, vn, out, plot=True)
# 	
# 	address='Image_t_F_ss_%i'%imodel
# 	vn=0.000000000000000001
# 	out='F_ss'
# 	envelop_trace(address, nbl, vn, out, plot=True)
#  	
# 	address='Image_t_N_pp_%i'%imodel
# 	vn=1
# 	out='N_pp'
# 	envelop_trace(address, nbl, vn, out, plot=False)
# 	
# 	address='Image_t_N_ss_%i'%imodel
# 	vn=1
# 	out='N_ss'
# 	envelop_trace(address, nbl, vn, out, plot=False)
# 	
# 	address='Image_t_laplace_pp_%i'%imodel
# 	vn=20000
# 	out='laplace_pp'
# 	envelop_trace(address, nbl, vn, out, plot=False)
# 	
# 	address='Image_t_laplace_ss_%i'%imodel
# 	vn=100000
# 	out='laplace_ss'
# 	envelop_trace(address, nbl, vn, out, plot=False)

	###############################################
	###############################################
	#.. Plot parameter model to compare Imagine location
# 	address ='../../velocity_models_T/model_forward_%i.mat'%imodel
# 	model, vp, vs, ro, dx, dz = load_velocity_model(so, nbl, address)
# 	plot_single_model(model, name='model_parameter_lambda')
#  	
# 	##############################################
# 	##############################################
# 	#.. Image condition mixed
# 	address ='../../velocity_models/model_%i.mat'%imodel
# 	model, _, _, _, _, _ = load_velocity_model(so, nbl, address)
# 	address_f= 'wavefields/Wavefield_Velocity_f'
# 	address_b= 'wavefields/Wavefield_Velocity_b'
# 	image_mixed(model, nshots, address_f, address_b, dt)
# 	print('Finishing Mixed image')
# 	
# 	##..Plot with ### vm1=xx, vm2=N_xx, vm3=N_xxf, vm4=xxf
# 
# 	address='Image_mixed'
# 	vm1=1000
# 	vm2=0.00001
# 	vm3=0.000001
# 	vm4=100000
# 	plot_mixed(nbl, address, vm1, vm2, vm3, vm4)
# 	
	###############################################
	###############################################
	

##EXTRA - TEST 
###############################################
###############################################

### Trace comparison from the Image, shift test, the question is
## where the image is formed? right location?

# address='Image_pp'
# trace_comparison(address)

# address='Image_ss'
# trace_comparison(address)


################################################
################################################
##... Plot shotgathers 
# for i in  nshots:
# 	address='output/rec_vz_true_data.mat'
# 	new_rec_vz = shift_wavefield(address)
# 	plot_simple_shotgaher(arr=new_rec_vz, name_output='rec_true_bone_vz_shifted')
##################


################################################
############################################### 
##... Plot wavefield 
# address='wavefields/Wavefield_Velocity_b'
# name_out='test'
# plot_slice_wavefield(nshots, nbl, address)#, name_out)
