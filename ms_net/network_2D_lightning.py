"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

'''
This script includes a forward pass, a training step, a validation step, and a test step. 
'''

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F

try: 
    from .network_tools_2D import scale_tensor
except (ImportError, ModuleNotFoundError):
    from network_tools_2D import scale_tensor

from pytorch_lightning import LightningModule

# converts data into scaler tensors, sharing data and preserving autograd history
EPS    = torch.as_tensor(1e-3)
MAXEPS = torch.as_tensor(5e4)


# Increases the size and diversity of training set
def rot_list(x_list,times,dims):
    """
    Description
    ___________

    Rotate an N-D tensor by 90 degrees in the plane specified by dims axis 

    Parameters
    __________
    x_list : list of tensors
    times : int
        number of times to rotate
    dims : list or tuple
        axis to rotate

    Returns
    _______
    list of tensors
        a  list of rotated tensors, where each tensor in the input list x_list has been rotated times number of times along the denoted in the dims axis

    """
    return [ torch.rot90(x,times,dims) for x in x_list ]

def get_trainable_models(scales, features, filters, f_mult):
    """
    Description
    ___________
    This function returns an array of trainable convolutional neural network models, each with a different scale (level of resolution).

    Parameters
    __________

    scales : int  
        the number of different scales (levels of resolution) to create.
    features : int
        the number of input features (channels) for the first layer of the models.
    filters : int
        the number of filters in the first convolutional layer of the models.
    f_mult : int
        the factor by which the number of filters increases with each scale.

    Returns
    _______

    models : (list of models)
        a list of trainable ConvNet models
    """
    models = []
    nc_in = features
    # normalizes all activations of a single layer from a batch
    norm = True
    # rectified linear unit activation function
    last_act = 'ReLU'
  

    # list of number filters in each model (scale), in reverse order so earlier layers capture low-level features
    num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
    # scale 0: 2, scale 1: 2*4**1 = 8, scale 2: 2*4**2 = 32 scale 3: 2*4**3 = 128
    print(f'Filters per model: {num_filters}')

    for it in range( scales ): 
        # Creates a model for each scale
        if it==1: nc_in+=1     
     
        # To convolve the domain + previous(upscaled) result                      
        models.append( get_model( nc_in    = nc_in,
                                  ncf      = num_filters[it],
                                  norm     = norm,
                                  last_act = last_act) )
       
    return models 


class get_model(nn.Module): 
  
    """
    Description
    ___________
    initializes the model's parameters and builds the layers when the instance of get_model is created
    """
 
  
    def __init__(self, nc_in, ncf, norm, last_act):
        """
        Description
        ___________
        This calls the constructor of the parent class (nn.Module) that the built-in functionality for nn.Module is initialized & it inherits .parameters(), .cuda())

        Parameters
        __________

        nc_in: int
            specifies the number of input channels, used with image data
        ncf: int
            defines the number of convolutional filters in the first
        norm: bool
            this controls whether normalization layers are applied after each convolutional layer
        last_act: str
            specifies the activation function to be used in the last layer
        
        """
   
        super(get_model, self).__init__()

  
        nc_out     = 1  
        # kernel side-length (3x3 filter used)
        ker_size   = 3   

        padd_size  = 1   
   
        ncf_min    = ncf 
        
        
        num_layers = 15   
        
        # HEAD OF MODEL

        self.head = ConvBlock2D( in_channel  = nc_in,
                                 out_channel = ncf,
                                 ker_size    = ker_size,
                                 padd        = padd_size,
                                 stride      = 1, # filter moves one pixel at a time across the input
                                 norm        = norm )
        
        # BODY OF THE MODEL
        self.body = nn.Sequential()
        for i in range( num_layers-1 ):
            
            new_ncf = int( ncf/2**(i+1) )
            # the number of filters that decrease by powers of 2 by each layer
            if i==num_layers-2: norm=False  

            convblock = ConvBlock2D( in_channel  = max(2*new_ncf,ncf_min),
                                     out_channel = max(new_ncf,ncf_min),
                                     ker_size    = ker_size,
                                     padd        = padd_size,
                                     stride      = 1,
                                     norm        = norm)
            
            self.body.add_module( f'block{i+1}', convblock )
          
        # TAIL OF THE MODEL

        if last_act == 'ReLU':
       
            self.tail = nn.Sequential(
                                    nn.Conv2d( max(new_ncf,ncf_min), nc_out,
                                               kernel_size=1,stride=1, padding=0),
                                    nn.ReLU()
                                 )
     
        else: # last_act != 'ReLU' (it is a 1x1 convolution with no activation)

            self.tail = nn.Sequential(
                            nn.Conv2d( max(new_ncf,ncf_min), nc_out, kernel_size=1,
                                       stride=1, padding=0)) # no pad needed since 1x1x1

# FORWARD PASS OF THE MODEL
    def forward(self,x):
        '''
        Description
        ___________
        This defines the forward pass of the model

        Parameters
        __________
        x: tensor
            input tensor to the network
        Returns
        _______
        x : tensor
          x is fed through the head, body, and tail of the network
        '''  
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x



# CUSTOM CONVOLUTIONAL BLOCK: 

class ConvBlock2D( nn.Sequential ):
    """
    Description
    ___________
    This extends nn.Sequential and combines several layers (a convolution, optional normalization, and activation) in a single reusable module

    """
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm):

        """
        Description
        ___________

        Defines the convolutional block class

        Parameters
        __________
        in_channel : int
            number of input channels for the convolutional layer
        out_channel : int
            number of output channels for the convolutional layer
        padd: int or tuple
            amount of padding to be added to the input
        norm : bool
            normalization is added (normalizes the input over each sample) 
        """
        super(ConvBlock2D,self).__init__()
        self.add_module( 'conv',
                         nn.Conv2d( in_channel, 
                                    out_channel,
                                    kernel_size=ker_size,
                                    stride=stride,
                                    padding=padd,
                                    padding_mode='circular',
                                    ) ),
        if norm == True:
            self.add_module( 'i_norm', nn.InstanceNorm2d( out_channel ) ),
        
        self.add_module( 'CeLU', nn.CELU( inplace=False ) )
        # Activation function CeLU (Continously Differentiable Exponential Linear Unit)
        

# ORAGANIZES AND SIMPFLIES THE TRAINING LOOPS

class MS_Net(LightningModule):
    """
    Description
    ___________
    Defines the architecture of the ms-net neural network

    """
    def __init__(
                 self, 
                 net_name     = 'test1', 
                 num_scales   =  4,
                 num_features =  1, 
                 num_filters  =  8, 
                 f_mult       =  4,  
                 lr           = 1e-4, # learning rate
                 steps        = 4, # steps in the training loop or optimization process
                 hparams      = None,
                 ):
        """
        Parameters
        __________
        net_name : str
            name of the network
        num_scales : int
            number of scales or resolutions that the input image will be processed
        num_features : int 
            number of input features. 1 for grayscale images 
        f_mult : int
            a multiplier for the number of filters in the convolutional layers
        lr : float
            the learning rate for the optimization process
        steps : int 
            the number of steps in the training loop or optimization process
        hparams : dict
            a dictionary of hyperparameters for the model
        
        """

        # INITIALIZATION
        super(MS_Net, self).__init__()

        self.save_hyperparameters(
            "net_name",
            "num_scales",
            "num_features",
            "num_filters",
            "f_mult",
            "lr",
            "steps",
        )

        """
        Description
        ___________
        Generates list of models for each scale (function defined above)
        """
        self.net_name = net_name
        self.scales   = num_scales
        self.feats    = num_features
        self.filters  = num_filters
        self.lr       = lr
        self.steps    = steps

        # make deeper convolution neural network, increasing the number of filters
        self.models   = nn.ModuleList( 
                                get_trainable_models( num_scales,
                                                      num_features,
                                                      num_filters,
                                                      f_mult ) ) 
    # FORWARD METHOD IN MS-NET
     
    def forward(self, x_list, masks):
        """
        Description
        ___________
        Allows the model to progressively refine predictions across scales

        Parameters
        __________
        x_list : list
            number of input features in x_list[0] (coarsest input) matches self.feats
        mask : list
            a list of mask tensors

        Returns
        _______
        y
            returns predictions
        """
        
        assert x_list[0].shape[1] == self.feats, \
        f'The number of features provided {x_list[0].shape[1]} \
            does not match with the input size {self.feats}'
        # carries out the first prediction (pass through the coarsest model)
        y = [ self.models[0]( x_list[0] ) ]
        for scale,[ model,x ] in enumerate(zip( self.models[1:],x_list[1:] )):
            # iterates through the remaining models and inputs, starting from second scale

            y_up = scale_tensor( y[scale], scale_factor=2 )*masks[scale]
            # scales the previous prediction up by 2 to match the resolution of current input scale, mask is used focus prediction on specific regions
            y.append( model( torch.cat((x,y_up),dim=1) ) + y_up )
           
        return y
    
    # DEFINES HOW BATCH DATA IS PROCESSED DURING TRAINING
   
    def training_step(self, batch, batch_idx):
        sample, masks, xy = batch['train']
        x, y = xy[0], xy[1]
        y_pred = self(x, masks)
        y_var = 1  # optionally: torch.max(y[-1].var(), EPS)

        # Create a valid autograd-tracked tensor for accumulation
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        last_valid_loss_s = None

        # Loop over each scale's prediction and ground truth
        for scale, (y_hat, yi) in enumerate(zip(y_pred, y)):
            loss_s = F.mse_loss(y_hat, yi) / y_var

            if torch.isnan(loss_s) or loss_s > MAXEPS:
                print(f"The loss at scale {scale} is {loss_s}")
                print("RUN")
                print("-" * 100)
                continue

            loss = loss + loss_s
            last_valid_loss_s = loss_s

            self.log(
                f"loss_scale{scale}", loss_s,
                on_step=False, on_epoch=True, logger=True, rank_zero_only=True
            )

        # Define consistent logging parameters
        log_args = dict(on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)

        # Log total loss and the fine-scale loss if any valid scales existed
        if last_valid_loss_s is not None:
            self.log("loss", loss, **log_args)
            self.log("loss_scale_fine", last_valid_loss_s, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        else:
            # All scales skipped — fallback to dummy tensor to satisfy AMP
            loss = 0.0 * sum([y_hat.sum() for y_hat in y_pred])
            self.log("loss", loss, **log_args)

        return loss





    def validation_step(self, batch, batch_idx):
        """
        Description
        ___________
        Function performs a single validation step on a batch of data & computes the model's predictions
        
        Parameters
        __________
        self :
            the instance of the class
        batch : tuple
            a tuple containing the sample data, masks, and input-output pairs
        batch_idx : int
            index of the current batch
        
        Returns
        _______
        loss : tensor
            the total validation loss for this batch
        """
        sample, masks, xy = batch
        x,y    = xy[0], xy[1]
        y_pred = self(x, masks)
        loss   = 0
        y_var  = 1 # torch.max(y[-1].var(), EPS)
        for scale, [y_hat,yi] in enumerate(zip(y_pred, y)):
            loss_s = F.mse_loss(y_hat,yi)/y_var
            if torch.isnan(loss_s):
                #print(f'The loss at scale {scale} is {loss_s}')
                print('RUN')
                print('-'*100)
            else:
                loss += loss_s
            self.log(f"val_loss_scale{scale}", loss_s, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log(f"val_scale_fine", loss_s, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        return loss
        
    def configure_optimizers(self):
        """
        Description
        ___________
        This collects all the learnable parameters of the model -- it tells the optimizers which parameters to update during backpropagation
        
        Parameters
        __________
        self : the instance of the class
        
        Returns
        _______
        an instance of the Adam optimizer configured with the model's parameters and learning rate

        """
        return Adam(self.parameters(), lr=self.lr)
        # this is where you would add learning rate scheduler (they haven't set up the scheduler) 

    def test(self, masks, x1, x2):
        """
        Description
        ___________

        The function performs a forward pass through the model

        Parameters
        __________

        self : instance
            instance of the class 
        masks : list of tensors
            representing masks to be applied to upscaled predictions
        x1 : list of tensors
            represents the first part of the input
        x2 : list of tensors represent second part of the input

        Returns
        y_up : tensor
            final predictions from the model
        
        """
        with torch.no_grad():
            y_up    = [0]*self.scales # upscaled prediction from coarser model
            x = [ torch.cat((xi1, xi2), dim=1) for xi1, xi2 in zip(x1,x2) ]
            del x1, x2
            for scale, model in enumerate( self.models ):
                if scale == 0:
                    y_up[scale] = model( x[scale] ) 
                else:
                    y_up = [scale_tensor(y,scale_factor=2)*masks[scale-1] if torch.is_tensor(y) else 0 for y in y_up]
                    y_scale = sum(y_up)
                    y_up[scale] = model(torch.cat((x[scale],y_scale), dim=1))+y_scale
                    del y_scale
            return y_up[-1]
    
 
    def num_params(self):
        """
        Description
        ___________

        This function calculates the total number of trainable parameters in the model

        Parameters
        __________

        self : instance
            instance of class
        Returns
        _______
        the total number of trainable parameters in the model

        """
        return sum(p.numel() for p in self.models.parameters() if p.requires_grad)
    
    def maxsize_singlescale(self):
        """
        Description
        ___________

        The function determines the maximum cube size the model can handle without running into memory errors

        Parameters
        __________

        self : instance
            instance of the class 

        Returns
        _______

        The maximum cube size 
        
        """
        with torch.set_grad_enabled( False ):
            for cube_size in np.arange(256,1000,64):
                print(f'Trying out size {cube_size}')
                try:
                    y = self.models[-1](torch.ones([1,3,cube_size,cube_size,cube_size]).to('cuda'))
                    del y
                except RuntimeError:
                    return cube_size-64
