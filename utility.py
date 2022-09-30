## @meermehran  || M3RG Lab  || Indian Institute of Technology, Delhi
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt


##### UTILITLY FUNCTIONS #####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


#loss function 
class L2Loss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(L2Loss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def testing(model,trainX, trainY, testX,testY, batch_size = 20,W =48,H =48, strain_channels =3):
    ''' Returns the predictions for TestSet'''
    n = trainX.shape[0]
    assert trainX.shape[0]==trainY.shape[0]
    assert testX.shape[0] == testY.shape[0]
    X_encoder = UnitGaussianNormalizer(trainX)
    testX = X_encoder.encode(testX)

    Y_encoder = UnitGaussianNormalizer(trainY)
    testY = Y_encoder.encode(testY)

    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(testX,testY), batch_size = batch_size, shuffle =False)

    prediction = torch.zeros(testY.shape).cuda()

    Y_encoder.cuda()
    counter = 0
    print(f' TEST SET SHAPE :: {testY.shape}')

    with torch.no_grad():
        for x , y in testloader:
          x,y = x.cuda(), y.cuda()
          # x= x.unsqueeze(-1)
          out = model(x).reshape(batch_size,W,H,strain_channels)  #out.shape[batchsize, 48,48, channels]

          out = Y_encoder.decode(out)
          prediction[counter*batch_size: (counter*batch_size) + batch_size] = out
          counter+=1
    return prediction

def getgrid(x_train):
    ''' GET THE GRID X and Y as per the dimensions of the input'''
    shape=torch.tensor(x_train.shape)
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    zzz=torch.cat((gridx, gridy), dim=-1)

    xcor=zzz.permute(0,3,1,2)[0][0].cpu().detach().numpy()
    ycor=zzz.permute(0,3,1,2)[0][1].cpu().detach().numpy()

    return xcor,ycor

def toNumpy(*args):
    '''Directly convert gpu tensor to cpu numpy array'''
    arr = [x.cpu().detach().numpy() for x in args]
    return arr[:]

def geometry(xtest,index, pos = 'vert'):
    '''MATERIAL GEOMETRY'''
    

    fig,ax =plt.subplots(1, figsize = (7.6,7))
    im = plt.pcolormesh(xtest[index],cmap ='Reds')
    ax.xaxis.set_tick_params(width =0)
    ax.yaxis.set_tick_params(width=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')


    for spine in ax.spines:
        ax.spines[spine].set_linewidth(3)

    pos1 = ax.get_position()

#     if pos == 'vert':
#         ax2 = fig.add_axes([pos1.x0,0.03,pos1.width,.05])
#     else:
#         ax2 = fig.add_axes([0.92,0.11,cbarwidth,0.77])

    cmap =plt.cm.Reds
#     bounds = [0,100,1000]
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     cb = mpl.colorbar.ColorbarBase(ax2,cmap = cmap,norm = norm, spacing = 'uniform', orientation ='horizontal')
#     cb.ax.tick_params(size = 0, labelsize =15)   
    plt.show()
    
def platter(ytest, zpred, index, hline=24, vline=24,name = 'S'):
    '''Plot Values along cross sectional lines-Horizontal and Vertical Line'''
    from matplotlib import ticker
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    ## crack is along horizontal axis. [y - axis is horizontal]
    if name == 'S':
        variable ='Stress'
    else:
        
        variable ='Strain'
    
    tt = ytest[index]  ## FEM
    zz = zpred[index]  ##ML
    t11 = tt[...,0]  ##FEM
    t22 = tt[...,1]  ##FEM
    t33 = tt[...,2]  ##FEM
    z11 = zz[...,0]  ##ML
    z22 = zz[...,1]  ##ML
    z33 = zz[...,2]  ##ML
 

#     hline = 24 ## along crack path-x-axis
    lt11 = t11[hline,:]
    lz11 = z11[hline,:]
    lt22 = t22[hline,:]
    lz22 = z22[hline,:]
    lt33 = t33[hline,:]
    lz33 = z33[hline,:]
    horizontal = np.array([[lt11, lz11],[lt22,lz22],[lt33,lz33]])

#     vline = 24
    lt11 = t11[:,vline]
    lz11 = z11[:,vline]
    lt22 = t22[:,vline]
    lz22 = z22[:,vline]
    lt33 = t33[:,vline]
    lz33 = z33[:,vline]
    vertical = np.array([[lt11, lz11],[lt22,lz22],[lt33,lz33]])



    fig, ax = plt.subplots(3,2, figsize=(20,27))
    c = 0
    for i in range(ax.shape[0]):
        ax[i,0].plot(range(48), horizontal[i][0][:],linewidth=6, color='red', label='FEM-Hor')      
        ax[i,0].plot(range(48), horizontal[i][1],linewidth=4, color='black', label='ML-Hor', linestyle ='dashed')

        # ax[i,0].tick_params(axis = 'both')
        ax[i,0].xaxis.set_tick_params(which='major', size=10, width=3, direction='in', top='on')
        ax[i,0].xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', top='on')
        ax[i,0].yaxis.set_tick_params(which='major', size=10, width=3, direction='in', right='on')
        ax[i,0].yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', right='on')
        M = 6
        yticks = ticker.MaxNLocator(M)
        ax[i,0].yaxis.set_major_locator(yticks)
        minor = AutoMinorLocator()
        ax[i,0].xaxis.set_minor_locator(minor)
        minor = AutoMinorLocator()
        ax[i,0].yaxis.set_minor_locator(minor)
        ax[i,0].yaxis.set_label_coords(-.1, .5)
        
        
        
        ax[i,0].set_xlabel('L', labelpad=20) 
        ax[i,0].set_ylabel(f'{variable}', labelpad=20)
        ax[i,0].set_xlim(0, 48)
        ax[i,0].legend()
 
        ax[i,1].plot(range(48), vertical[i][0][:],linewidth=6, color='red', label='FEM-Vert')  
        ax[i,1].plot(range(48), vertical[i][1][:],linewidth=4, color='black', label='ML-Vert', linestyle ='dashed')

        ax[i,1].xaxis.set_tick_params(which='major', size=10, width=3, direction='in', top='on')
        ax[i,1].xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', top='on')
        ax[i,1].yaxis.set_tick_params(which='major', size=10, width=3, direction='in', right='on')
        ax[i,1].yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', right='on')
        ax[i,1].set_xlabel('L', labelpad=15) 
        ax[i,1].set_ylabel(f'{variable}', labelpad=15)
        ax[i,1].set_xlim(0, 48)
        ax[i,1].legend()
        M = 6
        yticks = ticker.MaxNLocator(M)
        ax[i,1].yaxis.set_major_locator(yticks)
        minor = AutoMinorLocator()
        ax[i,1].xaxis.set_minor_locator(minor)
        minor = AutoMinorLocator()
        ax[i,1].yaxis.set_minor_locator(minor)
        ax[i,1].yaxis.set_label_coords(-.12, .5)

    plt.show()
    
def contour(zpred, ytest,xcor,ycor, index,cmap = 'Reds'):
    '''MAP OF TENSOR COMPONENT WISE'''
    R = zpred[index].shape[-1]
    fig, ax = plt.subplots(R,2, figsize = (16,7*(R+0.5)))
    counter = 0
    index = index
    # cmap = 'plasma_r'
    # cmap =cmap
    comb = [ zpred[index], ytest[index]]
    _min, _max = np.min(comb), np.max(comb)
    cbarwidth = 0.03
    for i in range(1):
        for j in range(R):
            # _min, _max = minmax(zpred[index][...,j], ytest[index][...,j])
            ax1 = ax[j,i]
            pl1 = ax1.pcolormesh(xcor, ycor, ytest[index][:,:,counter], cmap = cmap, vmin=_min, vmax= _max,shading ='auto')
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.xaxis.set_tick_params(width =0)
            ax1.yaxis.set_tick_params(width=0)
            ax1.set_aspect('equal')
            for spine in ax1.spines:
                ax1.spines[spine].set_linewidth(3)
            # ax1.spines['right'].set_linewidth(3)


            axx2 = ax[j,i+1]
            pcm2 = ax[j,i+1].pcolormesh(xcor, ycor,zpred[index][:,:,counter], cmap=cmap,vmin = _min, vmax = _max,shading='auto')
            axx2.axis('on')
            axx2.set_yticklabels([])
            axx2.set_xticklabels([])
            axx2.xaxis.set_tick_params(width =0)
            axx2.yaxis.set_tick_params(width=0)
            axx2.set_aspect('equal')
            for spine in axx2.spines:
                axx2.spines[spine].set_linewidth(3)
            counter+=1
    pos1 = ax1.get_position()
    aaa = plt.axes([0.94,pos1.y0,cbarwidth,0.74])
    colorbar =plt.colorbar(pl1, cax = aaa)
    colorbar.ax.tick_params(labelsize=30) 
    # colorbar.set_label("Strain", labelpad =-90, x  =0.2,y = 1.05, rotation =0,size = 25, weight = 20)  ## - left +rigjt
    # colorbar.ax.set_title('Strain', fontdict={'font':'25'})
    colorbar.outline.set_linewidth(3)
    plt.show()

def minmax(arr1, arr2):
  combined = np.array([arr1,arr2])
  min, max = np.amin(combined) , np.amax(combined)
  return min, max


