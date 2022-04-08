
# Differentiable FFT Conv Layer with Dense Color Channels
# Copyright 2022
# released under MIT license

# this is meant to be a drop in replacement for torch.conv
# functional_conv1d_fft  replaces  torch.nn.functional.conv1d
# Conv1d_fft             replaces  torch.nn.Conv1d
# supports 1d, 2d and 3d convolution

# api is not exactly matching yet
# unsupported:  stride, dilation, groups, etc


# b[0,:,:] = ifft( fft(x[0,:,:]) * fft(k[0,0,:,:]) + fft(x[1,:,:]) * fft(k[1,0,:,:]) + fft(x[2,:,:]) * fft(k[2,0,:,:]) )
# b[1,:,:] = ifft( fft(x[0,:,:]) * fft(k[0,1,:,:]) + fft(x[1,:,:]) * fft(k[1,1,:,:]) + fft(x[2,:,:]) * fft(k[2,1,:,:]) )
# b[2,:,:] = ifft( fft(x[0,:,:]) * fft(k[0,2,:,:]) + fft(x[1,:,:]) * fft(k[1,2,:,:]) + fft(x[2,:,:]) * fft(k[2,2,:,:]) )
# b[3,:,:] = ifft( fft(x[0,:,:]) * fft(k[0,3,:,:]) + fft(x[1,:,:]) * fft(k[1,3,:,:]) + fft(x[2,:,:]) * fft(k[2,3,:,:]) )

# b_fft[:,0,0] += bias[:] * prod(shape)

import torch

class conv_fft_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, bias=None, padding = 'valid', fft_dim = 1):   
          
        #channel first format only
        
        #if these dims are missing, need to skip the sum_reduce
        if x.dim() < fft_dim + 2:
            raise NotImplementedError('vector input to conv_fft expected to have shape (batch, channels, data_dim0, data_dimN)')
        if k.dim() < fft_dim + 2:
            raise NotImplementedError('kernel input to conv_fft expected to have shape (outchannels, inchannels, data_dim0, data_dimN)')
            
                
        in_channels = k.shape[-(fft_dim + 1)]
        out_channels = k.shape[-(fft_dim + 2)]
        
        #the axes where fft is calculated
        fft_axes = list(range(-fft_dim, 0))
        
        #kernel size along fft_axes
        kernel_size = k.shape[-fft_dim:]

        #input, padded, and output sizes along fft_axes, padded is the size used for fft
        if padding=='roll':
            input_size = x.shape[-fft_dim:]
            padded_size = list(x.shape[-fft_dim:])
            output_size = x.shape[-fft_dim:]             
        if padding=='valid':
            input_size = x.shape[-fft_dim:]
            padded_size = list(x.shape[-fft_dim:])
            output_size = [ input_size[i] - (kernel_size[i] - 1) for i in range(fft_dim) ]
        if padding=='same':
            input_size = x.shape[-fft_dim:]
            padded_size = [ input_size[i] + (kernel_size[i] // 2) for i in range(fft_dim) ]
            output_size = x.shape[-fft_dim:]                       
        if isinstance(padding, int):
            input_size = x.shape[-fft_dim:]
            padded_size = [ input_size[i] + padding * 2 for i in range(fft_dim) ]
            output_size = [ padding * 2 + input_size[i] - (kernel_size[i] - 1) for i in range(fft_dim) ] 

        #the kernel needs rolled, all other data are aligned to zero
        kernel_roll =   [-((size - 1) // 2) for size in kernel_size ]      
        kernel_unroll = [ ((size - 1) // 2) for size in kernel_size ] 

        #corrections to padding
        #    padded_size will be the size of the fft
        #    any larger paddings should work here
        #    other sizes might be faster
        #'valid' and other strange paddings cause a correction to kernel_roll, other data remain aligned to zero
        
        for i in range(fft_dim):
            #for example, if you only want even size fft
            #if padded_size[i] & 1:
            #    padded_size[i] = padded_size[i] + 1            
            if padding!='roll':       
                padded_size[i] = padded_size[i] + 31 & ~31

            if padding=='valid':
                offset = (min(kernel_size[i], input_size[i]) - 1) // 2
                kernel_roll[i] = kernel_roll[i] + offset
                kernel_unroll[i] = kernel_unroll[i] - offset                      
            if isinstance(padding, int):
                offset = (min(kernel_size[i], input_size[i]) - 1) // 2 - padding
                kernel_roll[i] = kernel_roll[i] + offset
                kernel_unroll[i] = kernel_unroll[i] - offset    

       
        #the kernel gets padded up to padded_size before being rolled, slightly inefficient
        if fft_dim == 1:
            kernel_padding = [0, padded_size[-1] - kernel_size[-1]]
        if fft_dim == 2:            
            kernel_padding = [0, padded_size[-1] - kernel_size[-1], 0, padded_size[-2] - kernel_size[-2]]
        if fft_dim == 3:
            kernel_padding = [0, padded_size[-1] - kernel_size[-1], 0, padded_size[-2] - kernel_size[-2], 0, padded_size[-3] - kernel_size[-3]]
        
        #these are used only to insert a 1 into the shape
        x_fft_shape =     x.shape[:-(fft_dim+1)] + (1, in_channels) + tuple(padded_size[:-1]) + (padded_size[-1] // 2 + 1,)
        dz_db_fft_shape = x.shape[:-(fft_dim+1)] + (out_channels,1) + tuple(padded_size[:-1]) + (padded_size[-1] // 2 + 1,)

        #outputs will be trimmed by these slices
        b_slice_size = [...] + [ slice(0, output_size[i]) for i in range(fft_dim) ]
        x_slice_size = [...] + [ slice(0, input_size[i]) for i in range(fft_dim) ]
        k_slice_size = [...] + [ slice(0, kernel_size[i]) for i in range(fft_dim) ]
                           
        x_fft = torch.reshape(torch.fft.rfftn(x, dim=fft_axes, s=padded_size), x_fft_shape) 

        k_fft = torch.fft.rfftn(torch.roll(torch.nn.functional.pad(k, kernel_padding), kernel_roll, fft_axes), dim=fft_axes)
                
        b_fft = torch.sum(x_fft * torch.conj(k_fft), dim=-(fft_dim + 1))   #sum along in_channels dim
        
        #bias is added to zero bin of fft, it needs scaled by prod(padded_size)
        if bias != None:
            prod_padded_size = 1
            for s in padded_size:
                prod_padded_size *= s
            b_fft[ (..., ) + (0, ) * fft_dim ] += bias * prod_padded_size
            
        b = torch.fft.irfftn(b_fft, dim=fft_axes, s=padded_size)[b_slice_size]
                                    
        ctx.save_for_backward(x_fft, k_fft)
        ctx.my_saved_variables = [
            bias, 
            fft_dim, dz_db_fft_shape,
            padded_size,
            kernel_unroll, fft_axes, 
            x_slice_size,
            k_slice_size ]        

        return b 


    @staticmethod
    def backward(ctx, dz_db):
        x_fft, k_fft = ctx.saved_tensors
        bias, fft_dim, dz_db_fft_shape, padded_size, kernel_unroll, fft_axes, x_slice_size, k_slice_size = ctx.my_saved_variables
                    
        dz_db_fft = torch.reshape(torch.fft.rfftn(dz_db, dim=fft_axes, s=padded_size), dz_db_fft_shape)
        
        #the zero freq dc bin of an fft ... is the sum of the signal ...
        #so dz_dbias[out_channel] = dz_db_fft[out_channel, 0, 0].real
        if bias != None:
            #this should instead sum all leading axes
            dz_dbias = torch.sum(dz_db_fft[ (..., 0) + (0,) * fft_dim ], dim=0).real    #sum along batch dim(s)
        else:
            dz_dbias = None
        
        dz_dx_fft = torch.sum(dz_db_fft * k_fft, dim=-(fft_dim + 2))       #sum along out_channels dim

        dz_dx = torch.fft.irfftn(dz_dx_fft, dim=fft_axes, s=padded_size)[x_slice_size]
        
        #this should instead sum all leading axes
        #reshape(-1, out_c, in_c, *fft_size)
        #if i wanted broadcasted conv k=(extradim1, out, in, kernelsize), x=(extradim0, extradim1, in, kernelsize)
        #sum pre-channel axes (size>1) in dz_da_fft that are 1 or missing in k_fft.shape, keepdim if 1 is present
        dz_dk_fft = torch.sum( x_fft * torch.conj(dz_db_fft), dim=0 )      #sum along batch dim(s)

        dz_dk = torch.roll(torch.fft.irfftn(dz_dk_fft, dim=fft_axes, s=padded_size), kernel_unroll, fft_axes)[k_slice_size]
        
        return dz_dx, dz_dk, dz_dbias, None, None


import math

class Conv_fft(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, device=None, dtype=torch.float32):
        super(Conv_fft, self).__init__()
        self.padding = padding
                
            
        weight = torch.zeros((out_channels, in_channels, *kernel_size), dtype=dtype, device=device)
        self.weight = torch.nn.Parameter(weight)
        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        
        if bias:
            bias = torch.zeros((out_channels,), dtype=dtype, device=device)
            self.bias = torch.nn.Parameter(bias)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None
                

class Conv1d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv1d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 1)
    
class Conv2d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv2d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 2)
    
class Conv3d_fft(Conv_fft):
    def __init__(self, *args, **kwargs):
        super(Conv3d_fft, self).__init__(*args, **kwargs)
                
    def forward(self, x):
        return conv_fft_function.apply(x, self.weight, self.bias, self.padding, 3)

def functional_conv1d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 1)

def functional_conv2d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 2)

def functional_conv3d_fft(x, k, bias=None, padding='valid'):
    return conv_fft_function.apply(x, k, bias, padding, 3)
