import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import copy
from devito import *
from examples.seismic import *
from examples.seismic.elastic import *
from scipy import ndimage

__all__ = ['image_pp', 'image_mixed', 'image_ss']

def imagine_correlation_VzF_VzB_normalized(vz_f, vx_f, vz_b, vx_b, dt):

    imagine   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma_x   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma_z   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma     = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    imagine_n = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    
    nt = vz_f.shape[0]
    print('vz_B.shape',vz_b.shape, 'nt', nt)
    for i in range (nt):
        ima = vz_f[i,:,:] * vz_b[i,:,:]* dt
        nox = vx_f[i,:,:]**2
        noz = vz_f[i,:,:]**2
        imagine[:,:] += ima
        norma_x[:,:] += nox
        norma_z[:,:] += noz
        norma[:,:] = norma_x+norma_z
        imagine_n[:,:]=imagine/norma
    data_n=np.array(imagine_n, dtype=float)
    lowpass = ndimage.gaussian_filter(data_n,1)
    imagie_f = data_n-lowpass
    return imagine, imagie_f, imagine_n
    
def imagine_correlation_VxF_VxB_normalized(vz_f, vx_f, vz_b, vx_b, dt):

    imagine   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma_x   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma_z   = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    norma     = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    imagine_n = np.zeros((vz_f.shape[1],vz_f.shape[2]))
    
    nt = vx_f.shape[0]
    print('vx_B.shape',vx_b.shape, 'nt', nt)
    for i in range (nt):
        ima = vx_f[i,:,:] * vx_b[i,:,:]* dt
        nox = vx_f[i,:,:]**2
        noz = vz_f[i,:,:]**2
        imagine[:,:] += ima
        norma_x[:,:] += nox
        norma_z[:,:] += noz
        norma[:,:] = norma_x+norma_z
        imagine_n[:,:]=imagine/norma
    data_n=np.array(imagine_n, dtype=float)
    lowpass = ndimage.gaussian_filter(data_n,1)
    imagie_f = data_n-lowpass
    return imagine, imagie_f, imagine_n

def image_pp(nbl, nshots, dt, imodel, plot_b=False, plot_f=False, plot_ima=False):
	for ishot in range(nshots):
		#Forward
		mat_contents = sio.loadmat('../outputs/D_F_%i.mat'%ishot)
		div_VF = copy.deepcopy(mat_contents['div_V'])
		
		print('div_VF.shape', div_VF.shape)
		shape=(div_VF.shape[1]-2*nbl+1, div_VF.shape[2]-2*nbl+1)
		print('VF Domain shape', shape)

		
		if plot_f:
			long2=int(div_VF.shape[0]/2)
			for it in range (1, long2, 10):
				plt.figure()
				plt.title('time_%i'%it)
# 				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
				plt.imshow(div_VF[it,nbl:shape[0]+nbl,nbl:shape[0]+nbl].T, cmap='seismic', vmin=-0.0005, vmax=0.0005)
				plt.colorbar()
				plt.savefig('../fig/p/div_vzf/div_Vz_%i.png'%(it))
		
		#Backward
		mat_contents = sio.loadmat('../outputs/D_B_%i.mat'%ishot)
		div_VB = copy.deepcopy(mat_contents['div_V'])
		
		print('div_VB.shape',div_VB.shape)
		shape=(div_VB.shape[1]-2*nbl+1, div_VB.shape[2]-2*nbl+1)
		print('VB Domain shape',shape)
		
		if plot_b:
			long2=int(div_VB.shape[0]/2)
			vmin=vmax=np.max(div_VB)/1000
			
			for it in range (1, long2, 10):
				plt.figure()
				plt.title('time_%i'%it)
				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
				plt.imshow(div_VB[it,:-1,:-1].T, cmap='seismic', vmin=-vmin, vmax=vmax)
				plt.colorbar()
				plt.savefig('../fig/p/div_vzb/div_Vz_%i.png'%(it))
				
		Image_PP     = np.zeros((div_VB.shape[1],div_VB.shape[2]))
		Norma_pp     = np.zeros((div_VB.shape[1],div_VB.shape[2]))
		Image_PP_N   = np.zeros((div_VB.shape[1],div_VB.shape[2]))

		nt = int(div_VB.shape[0])
		for i in range (nt):
			ima = div_VF[i,:,:] * div_VB[i,:,:]* dt
			Image_PP[:,:] += ima
			nox = div_VF[i,:,:]**2
			Norma_pp[:,:] += nox
			Image_PP_N[:,:]=Image_PP/Norma_pp			
			
		if plot_ima:
			vmin=vmax=np.max(Image_PP)/800
			plt.figure()
			plt.imshow(Image_PP.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_pp_%i.png'%imodel)
			
			vmin=vmax=np.max(Norma_pp)/500
			plt.figure()
			plt.imshow(Norma_pp.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_Npp3_%i.png'%imodel)

			vmin=vmax=np.max(Image_PP)/0.005
			plt.figure()
			plt.imshow(Image_PP_N.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_pp_N_%i.png'%imodel)
			
		print('Saving Image PP')
		outdict                     = dict()
		outdict['Ima']              = Image_PP
		outdict['Ima_N']            = Image_PP_N
		outdict['dt']               = dt
		outdict['nbl']              = nbl
		sio.savemat('../outputs/Image_pp_%i.mat'%ishot, outdict)       	
		print('**** finish I_pp ')
		
def image_mixed(model, nshots, address_f, address_b, dt):
	
	ImageT_xx      = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	ImageT_N_xx    = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	ImageT_N_xxf   = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	ImageT_zz      = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	ImageT_N_zz    = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	ImageT_N_zzf   = np.zeros((model.lam.data.shape[0],model.lam.data.shape[1]))
	
	for i in range(nshots):
		mat_contents = sio.loadmat('../outputs/%s_%i.mat'%(address_f,i))
		vx_f = copy.deepcopy(mat_contents['v_x'])
		vz_f = copy.deepcopy(mat_contents['v_z'])
		
		mat_contents = sio.loadmat('../outputs/%s_%i.mat'%(address_b,i))
		vx_b = copy.deepcopy(mat_contents['v_x'])
		vz_b = copy.deepcopy(mat_contents['v_z'])
		
		ii_zz, ii_n_zz, ii_n_zzf = imagine_correlation_VzF_VzB_normalized(vz_f, vx_f, vz_b, vx_b, dt)
		ImageT_zz[:,:]    += ii_zz
		ImageT_N_zz[:,:]  += ii_n_zz
		ImageT_N_zzf[:,:] += ii_n_zzf

		
		ii_xx, ii_n_xx, ii_n_xxf = imagine_correlation_VxF_VxB_normalized(vz_f, vx_f, vz_b, vx_b, dt)
		ImageT_xx[:,:]     += ii_xx
		ImageT_N_xx[:,:]   += ii_n_xx
		ImageT_N_xxf[:,:]  += ii_n_xxf
		
	outdict = dict()
	outdict['ImageT_xx']     = ImageT_xx[:]
	outdict['ImageT_N_xx']   = ImageT_N_xx[:]
	outdict['ImageT_N_xxf']  = ImageT_N_xxf[:]
	outdict['ImageT_zz']     = ImageT_zz[:]
	outdict['ImageT_N_zz']   = ImageT_N_zz[:]
	outdict['ImageT_N_zzf']  = ImageT_N_zzf[:]
	sio.savemat('../outputs/Image_mixed.mat', outdict)

def image_ss(nbl, nshots, dt, imodel, plot_b=False, plot_f=False, plot_ima=False):
	for ishot in range(nshots):
		#Forward
		mat_contents = sio.loadmat('../outputs/C_F_%i.mat'%ishot)
		curl_VF = copy.deepcopy(mat_contents['Curl_V'])
		
		print('Curl_VF.shape', curl_VF.shape)
		shape=(curl_VF.shape[1]-2*nbl+1, curl_VF.shape[2]-2*nbl+1)
		print('Curl VF Domain shape', shape)

		
		if plot_f:
			long2=int(curl_VF.shape[0]/2)
			for it in range (1, long2, 10):
				plt.figure()
				plt.title('time_%i'%it)
				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
				plt.imshow(curl_VF[it,:-1,:-1].T, cmap='seismic', vmin=-0.0005, vmax=0.0005)
				plt.colorbar()
				plt.savefig('../fig/s/curl_vf/curl_VF_%i.png'%(it))
		
		#Backward
		mat_contents = sio.loadmat('../outputs/C_B_%i.mat'%ishot)
		curl_VB = copy.deepcopy(mat_contents['Curl_V'])
		
		print('curl_VB.shape',curl_VB.shape)
		shape=(curl_VB.shape[1]-2*nbl+1, curl_VB.shape[2]-2*nbl+1)
		print('Curl VB Domain shape',shape)
		
		if plot_b:
			long2=int(curl_VB.shape[0]/2)
			vmin=vmax=np.max(curl_VB)/1000
			
			for it in range (1, long2, 10):
				plt.figure()
				plt.title('time_%i'%it)
				plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl),'-r')
				plt.imshow(curl_VB[it,:-1,:-1].T, cmap='seismic', vmin=-vmin, vmax=vmax)
				plt.colorbar()
				plt.savefig('../fig/s/curl_vb/curl_VB_%i.png'%(it))
			
		Image_ss     = np.zeros((curl_VB.shape[1],curl_VB.shape[2]))
		Norma_ss     = np.zeros((curl_VB.shape[1],curl_VB.shape[2]))
		Image_ss_N   = np.zeros((curl_VB.shape[1],curl_VB.shape[2]))

		nt = int(curl_VB.shape[0])
		for i in range (nt):
			ima = curl_VF[i,:,:] * curl_VB[i,:,:]* dt
			Image_ss[:,:] += ima
			nox = curl_VF[i,:,:]**2
			Norma_ss[:,:] += nox
			Image_ss_N[:,:]=Image_ss/Norma_ss
			
		if plot_ima:
			vmin=vmax=np.max(Image_ss)/5000
			plt.figure()
			plt.imshow(Image_ss.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_ss_%i.png'%imodel)
			
			vmin=vmax=np.max(Norma_ss)/500
			plt.figure()
			plt.imshow(Norma_ss.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_Nss_%i.png'%imodel)

			vmin=vmax=np.max(Image_ss)/0.005
			plt.figure()
			plt.imshow(Image_ss_N.T, aspect='auto', vmin=-vmin, vmax=vmax)	
			plt.plot((nbl,nbl,shape[0]+nbl,shape[0]+nbl,nbl),(nbl,shape[1]+nbl,shape[1]+nbl,nbl,nbl), '-k')
			plt.colorbar(label='Amplitud',fraction=0.046, pad=0.04)
			plt.savefig('../fig/I_ss_N_%i.png'%imodel)		
				
		print('Saving Image ss')
		outdict                     = dict()
		outdict['Ima']              = Image_ss
		outdict['Ima_N']            = Image_ss_N
		outdict['dt']               = dt
		outdict['nbl']              = nbl
		sio.savemat('../outputs/Image_ss_%i.mat'%ishot, outdict)       	
		print('**** finish I_ss')
		


			
