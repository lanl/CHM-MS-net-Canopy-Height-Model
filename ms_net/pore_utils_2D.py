"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
This script provides a set of functions for loading, preprocessing, and organizing the data.
"""
import os
from dotenv import load_dotenv, find_dotenv
import yaml
from tqdm import tqdm
import numpy as np
import torch
from hdf5storage import loadmat 

try: 
    from .network_tools_2D import scale_tensor, get_masks
except (ImportError, ModuleNotFoundError):
    from network_tools_2D import scale_tensor, get_masks

from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd


_MIN = 0
_MAX = 1111111111111115000




# Model Utilities

# Loading environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
directory = os.getenv("project_path")


def load_hparams(yaml_loc):
    """
    Description
    ___________
    This function loads hyperparameters from a YAML file located at the specified yaml_loc

    Parameters
    __________
    yaml_loc :str
        the path to the yaml file
    
    Returns
    _______
    dict
        a dictionary containing the loaded hyperparameters
    """
    with open(yaml_loc, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def create_folder(params):
    """
    Description
    ___________
    This function creates a new directory (if it doesn't exists) and copies necessary files

    Parameters
    _________
    params : dict
        a dictionary containing the model parameters

    """
  
    import datetime    
    from shutil import copyfile, move, SameFileError
    
    model_name = params['net_name']
    
    def rename_file(file_name):
        try:
            move(f'savedModels/{model_name}/{file_name}',
                   f'savedModels/{model_name}/' +
                   datetime.datetime.today().strftime("%d_%H_%M_")+
                   f'{file_name}' )
        except SameFileError:
            pass
        except: 
            print(f'Seems like {file_name} is not here :(')
        
    try:
        os.mkdir(f'savedModels/{model_name}')
        os.mkdir(f'savedModels/{model_name}/plots')
        print(f"Directory {model_name} created") 
        
    except FileExistsError:
        print(f"Directory {model_name} already exists") 
        for file_n in ['train.py', 'net_dict.json',
                       'results_dict.json','losses.png']: 
            rename_file(file_n) 
        
    
    copyfile('train.py', f'savedModels/{model_name}/train.py')




## Data Loading

def samples_avail(file_loc):
    """
    Description
    ___________
    This function checks which data files are available in file loc and creates dictionaries that contain samples, subsample, size, and pressure information

    Parameters
    __________
    file_loc : str
        The location of the directory containing the data files
    
    Returns
    _______
    syn_data : dict
        dictionary containing the available synthetic data samples and its information
    test_data : dict
        dictionary containing the available test data samples and its information
    
    """
    import pandas as pd
    # func to check which files exist
    def avail_check(name):
        """
        Description
        ___________
        To extract sample name, subsample size, and pressure from file

        Parameters
        __________

        name

        Returns
        _______

        sample
        subsample
        size
        pressure

        """
        sample, subsample, size, pressure = [],[],[],[]
        for p in [1,2,5,10,20]: #pressure values
            if os.path.isfile(f'{file_loc}/results_real/{name}_{p}_uz.mat'):
                string = f'{name}_{p}'.split('_')
                if len(string)==4:
                    sample_name = string[0]
                elif len(string)==5:
                    sample_name = f'{string[0]}_{string[1]}'
                    
                sample.append(    sample_name)
                subsample.append( string[-3])
                size.append(      string[-2])
                pressure.append(  string[-1])
                             
        return sample, subsample, size, pressure
    
    
    data    = pd.read_excel('geom_desciption.xlsx')
    data    = np.array(data)

    syn_data, test_data = [ dict( sample    = [],
                                  subsample = [],
                                  size      = [],
                                  pressure  = [],
                                 ) for i in range(2) ]
    
    for s in range(data.shape[0]):
        sample, subsample, size, pressure = avail_check(data[s,0])
        if sample:
            if data[s,1] == 'synthetic':
                syn_data['sample']+=sample
                syn_data['subsample']+=subsample
                syn_data['size']+=size
                syn_data['pressure']+=pressure
            else:
                test_data['sample']+=sample
                test_data['subsample']+=subsample
                test_data['size']+=size
                test_data['pressure']+=pressure
    return syn_data, test_data


def stoch_train_val(params, all_data=False):
    """
    Description
    ___________
    The function sets up the training and validation data based on the synthetic data and populates the params with the data needed for training, validation, and testing

    Parameters
    __________
    params : dict
        dictionary containing the model parameters (data_loc, training size, and validtion size)
    all_data : boolean
        Use  all available data (True) or subset for training and validation (False)
    
    """
    import random
    syn_data, test_data = samples_avail(params.data_loc)
    
    if all_data == False: # For training
        ind = random.sample( range(len(syn_data['sample'])),
                             params.training_size+params.validation_size ) 
    else: # All available data will be loaded
        ind = range(len(syn_data['sample']))
        
    
    train_in = ind[:params.training_size]
    val_in   = ind[params.training_size:]
    
    params.train_sample    = [syn_data['sample'][i] for i in train_in]
    params.train_subsample = [syn_data['subsample'][i] for i in train_in]
    params.train_size      = [syn_data['size'][i] for i in train_in]
    params.train_pressure  = [syn_data['pressure'][i] for i in train_in]
    
    params.val_sample      = [syn_data['sample'][i] for i in val_in]
    params.val_subsample   = [syn_data['subsample'][i] for i in val_in]
    params.val_size        = [syn_data['size'][i] for i in val_in]
    params.val_pressure    = [syn_data['pressure'][i] for i in val_in]
    
    params.test_sample     = test_data['sample']
    params.test_subsample  = test_data['subsample']
    params.test_size       = test_data['size']
    params.test_pressure   = test_data['pressure']
    
    return 
     
            
    

def data4test(data, net_dict, half=False):
    """
    Description
    ___________
    This function prepares the data by moving the data to specified device

    Parameters
    __________
    data : list[Tensor]
        input data to be prepared for testing
    net_dict : dict
        a dictionary containing the network config (including device key)
    half : boolean
        convert data to half-precision floating-point (True) or not (False)
    Returns
    _______
    samplenum : data[0].item() : int
        sample number
    masks : data[1] : list[Tensor]
        masks
    xs : data[2][0]: list[Tensor]
        input features
    ys : data[2][1] : list[Tensor]
        target outputs

    """
    data = movedata( data, net_dict['device'] )
    if half:
        data = tohalf(data)

#   print(data[1])
    return int(data[0].item()), data[1], data[2][0], data[2][1], data[-1]
#          #samplenum,      masks,         xs,                ys, c
          
          
def sortdata(data, net_dict):
    """
    Description
    ___________
    Stacks the features based on the x_array key in the net_dictionary

    Parameters
    __________

    data : list[Tensor]
        the input data to be sorted
    net_dict : dict
        the dictionary containing the network config

    Returns
    _______
    list[Tensor]
        a list containing the sorted features
    """
    # Returns stacked features
    num_xs = len(net_dict['x_array'])
    
    # print("\n========== sortdata DEBUG ==========")
    # print("num_xs:", num_xs)
    # print("x_array:", net_dict['x_array'])
    # print("len(data):", len(data))
    # for i, x in enumerate(data):
    #     print(f"\ndata[{i}]:")
    #     print("  type:", type(x))
    #     try:
    #         print("  len:", len(x))
    #         for j, item in enumerate(x):
    #             if hasattr(item, "shape"):
    #                 print(f"  [{j}] shape: {item.shape}")
    #             else:
    #                 print(f"  [{j}] type: {type(item)}")
    #     except Exception as e:
    #         print("  Could not inspect:", e)
    # print("====================================\n")

    return [[torch.cat(feats) for feats in zip(*data[:num_xs])], *data[num_xs:]]
    
    


def tohalf(data):
    """
    Description
    ___________
    Function converts the data type of input data to float16

    Parameters
    __________
    data : tensor or list
        the input data

    Returns
    _______
    tensor or list
        input data converted to float16
    """
    return [elem.half() if type(elem)==torch.Tensor else 
        tohalf(elem) if type(elem)==list else elem for elem in data]



def movedata(data, device):
    """
    Description
    ___________
    Funtion moves input data to the the specified device

    Parameters
    __________
    data : tensor or list
        the input data
    device : str
        the device where data is moved

    Returns
    _______
    torch or list
        the input data that is moved to the specified device
    """
    return [elem.to(device) if type(elem)==torch.Tensor else 
        movedata(elem, device) if type(elem)==list else elem for elem in data]

def return_fields(net_dict, phase):
    """
    Description
    ___________
    This function reads the data file from the net_dict and returns a list of samples

    Parameters
    __________
    net_dict : dict
        dictionary that contains file paths for the data
    phase : str
        specifies the data that is loaded

    Returns
    _______
    samples : list
        list of samples
    
    
    """
    data_file   = net_dict[f'{phase}_list']
    dataframe   = pd.read_csv(data_file, header=None)
    samples     = dataframe.values.reshape(len(dataframe)).tolist()
                   
    return samples

def get_dataloader(net_dict, phases, data_path, NORM_CONST):
    """
    Description
    ___________
    The function creates a dataloader object for the specified phases ('train', 'val', and 'test').

    Parameters
    __________
    net_dict : dict
        dictionary that contains file paths for the data
    phases : list
        list of strings that specifies the phases to be loaded

    Returns
    _______
    dataloader : dict
        dictionary of dataloader objects

    
    """
    from torch.utils.data import DataLoader
    
    if isinstance(phases,str) == True: # when only testing is needed
        phases = [phases]
    
    """The dataloader will have the following structure per sample:
        - DL[0]: sample number (int)
        - DL[1]: list of masks of size = num of scales, where the last one is binary
        - DL[2]: list of inputs/outputs
            i.e. DL[2][0][-1] should be the largest input image, conversely
            i.e. DL[2][-1][-1] should be the largest output (y)
      """  
    
    dataloader = {}
    for phase in phases:
        #check_inputs(net_dict, phase)
        samples = return_fields(net_dict, phase)
        data = []
        
        for num, name_tuple in enumerate(zip(tqdm(samples))):
            if num != 0:
                if num<_MIN or num>_MAX: # memory limit
                    continue
            sample_name = name_tuple[0]
            data_tmp = get_sample(net_dict, sample_name , phase, data_path, NORM_CONST) 
            if len(net_dict['x_array']) > 1:
                data_tmp = sortdata(data_tmp, net_dict) # concat feats
            masks = get_masks(data_tmp[0][-1][0][None,None,], net_dict['num_scales'])
            data.append( (num,) + (masks,) + (data_tmp,) ) #+ (c,) ) 
        dataloader[phase] = DataLoader(data, batch_size=net_dict['batch_size'],
                                      shuffle = (phase=='train'), 
                                      pin_memory=True, 
                                      num_workers=net_dict['num_workers'],
                                      persistent_workers=True)
    return dataloader
    

def get_sample(net_dict, sample_name, phase, data_path, NORM_CONST):
    """
    Description
    ___________
    The function loads the input and target data for specific sample and phase. 

    Parameters
    __________
    net_dict : dict
        dictionary that contains file paths for the data
    sample_name : str
        name of the sample name

    Returns
    _______
    [ get_downscaled_list(im_array,net_dict) for im_array in tmp_dict ] : list of tensors
        list of tensors in the downscaled format

    """
    # if phase=='train': #TODO implement another phase
    #     phasename = 'training'
    # elif phase=='val':
    #     phasename = 'validation'
    # elif phase=='inf':
    #     phasename = 'inference'
    # else:
    #     print(f'phase: {phase}')
    #     raise NameError('Wrong feature name or not implemented')

    # print(f'*net_dict: {net_dict}')
        
    if phase == 'inf':
        tmp_dict =  [ load_samples( feat, sample_name, data_path, NORM_CONST) 
                        for feat in net_dict['x_array']]
    else:
        tmp_dict =  [ load_samples( feat, sample_name, data_path, NORM_CONST) 
                        for feat in (net_dict["x_array"] + net_dict['y_array']) ]
    return [ get_downscaled_list(im_array,net_dict) for im_array in tmp_dict ]
 
def get_sample_c(net_dict, sample_name):
    """
    Description
    ___________
    This function loads the context data for the sample, applies necessary transformation, and returns the data downscaled.

    Parameters
    __________
    net_dict : dict
        dictionary that contains file paths for the data
    sample_name : str
        name of the sample name

    Returns
    _______
    [get_downscaled_list_c(im_array,net_dict) for im_array in tmp_dict] : list of tensors
        list of tensors in the downscaled format

    """
   
    tmp_dict =  [ load_samples( feat, sample_name, net_dict, xform=xform ) 
                     for feat, xform in zip( 
                             [ *net_dict['c_array'] ],
                             [ *net_dict['c_xform'] ] ) ]
    return [get_downscaled_list_c(im_array,net_dict) for im_array in tmp_dict]


def get_downscaled_list_c(x, net_dict):
    """
    Description
    ___________
    The function takes the x array and returns a list of downscaled tensors. The last element of the tensor has a constant value of 1.

    Parameters
    __________
    x : array
        contains the input data
    net_dict : dict
        dictionary that contains file paths for the data

    Returns
    _______
    ds_x[::-1] : list of tensors
         list of downscaled tensors 
    """
    x = torch.Tensor( add_dims(x, 1) )
    ds_x = []
    ds_x.append(x)
    for i in range( net_dict['num_scales']-1 ): 
        tmp=scale_tensor( ds_x[-1], scale_factor=1/2, mode='nearest')
        tmp[-1][0,:]=1
        ds_x.append(tmp)
    return ds_x[::-1] #returns the reversed list (small images first)

def get_downscaled_list(x, net_dict):
    """
    Description
    ___________
    The function takes the x array and returns a list of downscaled tensors.

    Parameters
    __________
    x : array
        contains the input data
    net_dict : dict
        dictionary that contains file paths for the data

    Returns
    _______
    ds_x[::-1] : list of tensors
         list of downscaled tensors 
    """
    x = torch.Tensor( add_dims(x, 1) )
    ds_x = []
    ds_x.append(x)
    for i in range( net_dict['num_scales']-1 ):  
        ds_x.append( scale_tensor( ds_x[-1], scale_factor=1/2, mode='nearest') )
    return ds_x[::-1] #returns the reversed list (small images first)
            
"""
data stats
"""

def get_sum_stats(feat, net_dict, phase):
    
    """
    Returns the summary stats of a feature over multiple samples
    """
    
    #check_inputs(net_dict, phase)
    samples,subsamples,sizes,pressures = return_fields(net_dict, phase)
    sample_list = []
    for name_tuple in zip(samples, subsamples, sizes, pressures ):
        sample_name = "".join( [str(e)+'_' for e in name_tuple] )
        sample_list.append( load_samples(feat, sample_name, net_dict))
        
    return sum_stats( sample_list, remove_zeros=True )


def load_samples(feat, sample_name, data_path, NORM_CONST): #, phasename): #, net_dict, xform = None):
    """
    Description
    ___________

    Parameters
    __________
    feat: str
        either mpf, edist or uz
    sample_name: str
        name of the sample

    Returns
    _______
    sample : arr
        array containing the preprocessed data for the specified feature and sampl

    """
    if feat == 'wvimg':
        sample = plt.imread(os.path.join(data_path, 'wvimg', sample_name)) / 255
        # sample = sample / 255
        # sample = sample[:,:,0]
        # print('wvimg shape is '+str(sample.shape))
        #sample = np.pad(sample,(8,8),mode='edge')
    elif feat == 'dem':
        sample = plt.imread(os.path.join(data_path, 'dem', sample_name)) / 255
        # sample = sample[:,:,0]
        # print('dem shape is '+str(sample.shape))
        #sample = np.pad(sample,(8,8),mode='edge')   
    elif feat == 'solar':
        sample = plt.imread(os.path.join(data_path, 'solar', sample_name[:16]+'.tif')) / 255
        # print('solar shape is '+str(sample.shape))
        #sample = np.pad(sample,(8,8),mode='edge')        
    elif feat == 'sensor':
        sample = plt.imread(os.path.join(data_path, 'sensor', sample_name[:16]+'.tif')) / 255
        # print('sensor shape is '+str(sample.shape))
        #sample = np.pad(sample,(8,8),mode='edge')    
    elif feat == 'chm':
        # div by 35 as that is the approx height of the tallest tree in the dataset
        sample = plt.imread(os.path.join(data_path, 'chm', sample_name))
        sample = sample[:,:,0] / NORM_CONST
        sample[sample<0] = 0
        # print('chm shape is '+str(sample.shape))
    else:
        print(f'feat: {feat}')
        raise NameError('Wrong feature name or not implemented')
    

    sample[~np.isfinite(sample)]=0  
    return sample
        

def sum_stats(x, remove_zeros=False):
    
    """
    Description
    ___________
    Calculates the summary statistics per input tensor

    Parameters
    __________
    x : tensor
        input tensor 
    remove_zeros : bool
        the function will remove zeros from the input data before calculating summary statistics

    Returns
    _______

    t_dict : dict
        dictionary containing the summary statistics of the input data
    
    """
    
    t_dic = {} #dict for data transforms
    
    if remove_zeros == True:
        x = np.concatenate([xi[xi!=0] for xi in x])
        
    
    t_dic['min']     = np.min(x)
    t_dic['range']   = np.ptp(x)
    t_dic['max']     = np.max(x)
    t_dic['max_abs'] = np.max( np.abs(x) )
    t_dic['std']     = np.std(x)
    t_dic['mean']    = np.mean(x)
    return t_dic



def all_sum_stats( net_dict ):
    """
    Description
    ___________
    This function calculates the summary statistics for all the features in the net_dict dictionary

    Parameters
    __________
    net_dict : dict
        dictionary containing the data for neural network

    Returns
    _______
    net_dict : dict
        dictionary that contains summary statistics 

 
    """
    for feat in [ *net_dict['x_array'], *net_dict['y_array'] ]:
        # get the summary statistics
        net_dict[f'{feat}_stats'] = get_sum_stats( feat, net_dict, 'train' )
    return net_dict
        


## Utils


def get_coarsened_list(x, scales):
    """
    Description
    ___________
    Funtion takes a 3D array and creates a list of coarsened versions of a tensor
    Parameters
    __________
    x : array
        3D np array
    scales : int
        The number of coarsened versions to be created
    
    Returns
    ________
    list
        a list with the desired number of coarse-grained tensors
    """
    
    # converts to tensor and adds channel and batch dim
    x = torch.Tensor(add_dims(x, 1))
    
    ds_x = []
    ds_x.append(x)
    
    for i in range( scales-1 ): 
        ds_x.append( scale_tensor( ds_x[-1], scale_factor=1/2 ) )
    return ds_x[::-1] # returns the reversed list (small images first)




## Tensor Operations

def changepres(x, ttype=None):
    """
    Description
    ___________
    Function changes the data type of the input sensor x to the specified type

    Parameters
    __________
    x : Torch.tensor
        the input tensor to be converted
    ttype : str
        the desired data type
    Returns
    _______
    Torch.tensor
        the converted input tensor
    _______
    """
    if ttype == 'f32':
        return x.float()
    elif ttype == 'f16':
        return x.half()

    
def torchpres(x, ttype=None):
    """
    Description
    ___________
    Function applies the changepres() function to the input (which can be a single tensor or list of tensors)

    Parameters
    __________
    x : Torch.tensor or list
        The input tensor to be converted
    ttype : str
        The desired data type

    Returns
    _______

    Torch.tensor
        the converted input tensor
    """
    if isinstance(x,list) == True:
        x = [torchpres(xi, ttype) for xi in x]
    else:
        x = changepres(x, ttype)
    return x
    

def gpu2np(x):
    """
    Description
    ___________
    Function converts a tensor or list of tensors to a array or list of arrays

    Parameters
    __________
    x : torchTensor or list
        The input tensor to be converted

    Returns
    _______
    input tensor or list of tensors converted
    """
    if type(x) == list:
        x = [gpu2np(xi) for xi in x]
    else:
        x = x.detach().cpu().numpy().squeeze()
    return x

def add_dims(x, num_dims):
    """
    Description
    ___________
    Function adds the specified number of dimensions to input tensor by adding new dimensions

    Parameters
    __________
    x : torchTensor or list
        The input tensor to be converted
    num_dims : int
        the number of dimensions to be added

    Returns
    _______
    input tensor or list of tensors with dimensions added
    """
    for dims in range(num_dims):
        x = x[np.newaxis]
    return x


def rot90(x):
    """
    Description
    ___________
    Function applies a 90-degree cc rotation to the dimensions of the tensor in the input data

    Parameters
    __________
    x : torchTensor or list
        The input tensor or list of tensors
    Returns
    _______
    input tensor or list of tensors rotated
    """
    rot_list = []
    for tensor_list in x: 
        if isinstance(tensor_list,list):
            rot_list.append( [torch.rot90(tensor,k=1,dims=[1,2]) 
                              for tensor in tensor_list] )
    return rot_list


def rnd_array(size, scales, device='cpu'):
    """
    Description
    ___________
    Function creates a 3D tensor of random values and then the tensor is coarsened

    Parameters
    __________
    size : int
        the size of the 3D tensor in each dimension
    scales : list
        a list of scaling factors applied to the coarsened tensor
    device : str (optional)
        the device which tensors are created
    Returns
    _______
    torch.Tensor
        a 3D tensor of random values (that are coarsened)
    """
    return get_coarsened_list( ((torch.rand(1,
                                            size,
                                            size,
                                            size)>0.5)*1.0).to(device),scales)


## Data Transformations

def inv_xform( xform ):
    """
    Description
    ___________
    This function returns the inverse transformation based on the certain transformation type

    Parameters
    __________
    xform : str
        the transformation type (either inv_minMax, inv_div_maxabs, inv_div_scalar)

    Returns
    function
        returns the corresponding inverse transformation function
    _______
    """
    if xform == 'minMax':
        return inv_minMax
    elif xform == 'div_maxabs':
        return inv_div_maxabs
    elif xform == 'div_scalar':
        return inv_div_scalar
    else:
        raise NotImplementedError
    


def data_xform( xform ):
    """
    Description
    ___________
    This function returns the corresponding data transformation based on the transformation type

    Parameters
    __________
    xform : str
         The transformation type (either minMax, div_maxabs, div_mean, div_scalar, or std)

    Returns
    function
        Returns the corresponding transformation function
    _______
    """
    if xform == 'minMax':
        return minMax
    elif xform == 'div_maxabs':
        return div_maxabs
    elif xform == 'div_mean':
        return div_mean
    elif xform == 'div_scalar':
        return div_scalar
    elif xform == 'std':
        return std_t
    else:
        raise NotImplementedError


def minMax(x, t_dict):
    """
    Description
    ___________
    Function applies min-max normalization to input data using the parameters stored in t_dict
    Parameters
    __________
    x : list or array
        input data 
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    Transformed data with values scaled between 0 to 1
    """
    if type(x) == list:
        return  [ minMax(x0, t_dict) for x0 in x]
    else:
        return  (np.copy(x)-t_dict['min'])/t_dict['range']
    
def div_maxabs(x, t_dict):
    """
    Description
    ___________
    Function divides input data by the maximum value stored in t_dict['max_abs']

    Parameters
    __________
    x : list or array
        input data 
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    Transformed data with values scaled between -1 and 1
    """
    if type(x) == list:
        return  [ div_maxabs(x0, t_dict) for x0 in x]
    else:
        return  np.copy(x)/t_dict['max_abs']

def div_scalar(x, t_dict):
    """
    Description
    ___________
    Function divides input data by the maximum value stored in t_dict['scalar']

    Parameters
    __________
    x : list or array
        input data 
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    the transformed scaled data
    """
    if type(x) == list:
        return  [ div_scalar(x0, t_dict) for x0 in x]
    else:
        #return  np.copy(x)/5e-9
        return  np.copy(x)/t_dict['scalar']
    
def div_mean(x, t_dict):
    """
    Description
    ___________
    Function divides input data by the maximum value stored in t_dict['mean']

    Parameters
    __________
    x : list or array
        input data 
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    the transformed scaled (by the mean) data
    """
    if type(x) == list:
        return  [ div_mean(x0, t_dict) for x0 in x]
    else:
        return  np.copy(x)/t_dict['mean']
    
def std_t(x, t_dict):
    """
    Description
    ___________
    Function divides input data by the maximum value stored in t_dict['std'] and t_dict['mean']

    Parameters
    __________
    x : list or array
        Input data 
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    Transformed data with values centered around 0 and scaled by std
    """
    if type(x) == list:
        return  [ std_t(x0, t_dict) for x0 in x]
    else:
        return  (np.copy(x)-t_dict['mean'])/t_dict['std']

def inv_minMax(xt, t_dict):
    """
    Description
    ___________
    Function applies the inverse of min-max normalization to transformed data

    Parameters
    __________
    xt : list or array
        Tranformed data to be inverted
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    original data scaled to its range
    """
    return t_dict['range']*(xt)+t_dict['min']

def inv_div_maxabs(xt, t_dict):
    """
    Description
    ___________
    Function applies the inverse of the division by the maximum absolute value transformation to transformed data

    Parameters
    __________
    xt : list or array
        Tranformed data to be inverted
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    original data scaled to its range
    """
    return t_dict['max_abs']*(xt)

def inv_div_scalar(xt, t_dict):
    """
    Description
    ___________
    Function applies the inverse of the division by a scalar transformation to transformed data

    Parameters
    __________
    xt : list or array
        Tranformed data to be inverted
    t_dict : dict
        The dictionary containing transformation parameters

    Returns
    _______
    original data scaled to its range
    """
    #return xt*5e-9
    return xt*t_dict['scalar']


        
    
