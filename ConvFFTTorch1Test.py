import torch
import time

from ConvFFTTorch1 import functional_conv1d_fft, functional_conv2d_fft, functional_conv3d_fft

#torch.manual_seed(123456)
    
cuda_device = torch.device("cuda")  # device object representing GPU

fft_dim = 2

if fft_dim == 1:
    batch_size = 1
    data_size= (10240,)
    kernel_size = (2501,)
    in_channels = 1
    out_channels = 1
    padding = 'valid'

if fft_dim == 2:
    batch_size = 3
    data_size= (128,128)
    kernel_size = (25,25)
    in_channels = 2
    out_channels = 3
    padding = 'valid'
    
if fft_dim == 3:
    batch_size = 1
    data_size= (32, 32, 32)
    kernel_size = (15, 15, 15)
    in_channels = 1
    out_channels = 1
    padding = 'same'

if fft_dim == 1:
    conv_fft = functional_conv1d_fft
    conv_torch = torch.nn.functional.conv1d
if fft_dim == 2:
    conv_fft = functional_conv2d_fft
    conv_torch = torch.nn.functional.conv2d
if fft_dim == 3:
    conv_fft = functional_conv3d_fft
    conv_torch = torch.nn.functional.conv3d
    

x_true = torch.rand((batch_size, in_channels, *data_size))
k_true = torch.rand((out_channels, in_channels, *kernel_size))
k_bias_true = torch.rand(out_channels)

b_true = conv_torch(x_true, k_true, bias=k_bias_true, padding=padding)
#b_true = torch.rand(b_true.shape)

x_pred = torch.rand((batch_size, in_channels, *data_size))
x_pred_t = x_pred.clone().detach()

k_pred = torch.rand((out_channels, in_channels, *kernel_size))
k_pred_t = k_pred.clone().detach()

k_bias_pred = torch.rand(out_channels)
k_bias_pred_t = k_bias_pred.clone().detach()

x_true = x_true.to(cuda_device)
k_true = k_true.to(cuda_device)
k_bias_true = k_bias_true.to(cuda_device)
b_true = b_true.to(cuda_device)

x_pred = x_pred.to(cuda_device)
x_pred_t = x_pred_t.to(cuda_device)

k_pred = k_pred.to(cuda_device)
k_pred_t = k_pred_t.to(cuda_device)

k_bias_pred = k_bias_pred.to(cuda_device)
k_bias_pred_t = k_bias_pred_t.to(cuda_device)

x_true.requires_grad = False
k_true.requires_grad = False
b_true.requires_grad = False

x_pred.requires_grad = True
x_pred_t.requires_grad = True

k_pred.requires_grad = True
k_pred_t.requires_grad = True

k_bias_pred.requires_grad = True
k_bias_pred_t.requires_grad = True

lr = 0.0001
steps = 501
mse_loss = torch.nn.MSELoss()

print('solving for a')

optimizer = torch.optim.Adam(params=[k_pred],  lr=lr)
for step in range(steps):
    b_pred = conv_fft(x_true, k_pred, bias=k_bias_pred, padding=padding)
    loss = mse_loss(b_pred, b_true)
    optimizer.zero_grad()
    loss.backward()    
    if step == 0:
        grad_k_fft = k_pred.grad.clone().detach()
        grad_k_bias_fft = k_bias_pred.grad.clone().detach()
        output_b_fft = b_pred.clone().detach()
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='') 
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_fft elapsed time', end_time - start_time)

optimizer = torch.optim.Adam(params=[k_pred_t],  lr=lr)
for step in range(steps):
    b_pred = conv_torch(x_true, k_pred_t, bias=k_bias_pred_t, padding=padding)
    loss = mse_loss(b_pred, b_true)
    optimizer.zero_grad()
    loss.backward()    
    if step == 0:
        grad_k_torch = k_pred_t.grad.clone().detach()  
        grad_k_bias_torch = k_bias_pred_t.grad.clone().detach()
        output_b_torch = b_pred.clone().detach()      
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='') 
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_torch elapsed time', end_time - start_time)

print('solving for x')

optimizer = torch.optim.Adam(params=[x_pred],  lr=lr)
for step in range(steps):
    b_pred = conv_fft(x_pred, k_true, bias=k_bias_true, padding=padding)
    loss = mse_loss(b_pred, b_true)
    optimizer.zero_grad()
    loss.backward()                    
    if step == 0:
        grad_x_fft = x_pred.grad.clone().detach()        
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='') 
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_fft elapsed time', end_time - start_time)

optimizer = torch.optim.Adam(params=[x_pred_t],  lr=lr)
for step in range(steps):    
    b_pred = conv_torch(x_pred_t, k_true, bias=k_bias_true, padding=padding)
    loss = mse_loss(b_pred, b_true)    
    optimizer.zero_grad()
    loss.backward()            
    if step == 0:
        grad_x_torch = x_pred_t.grad.clone().detach()        
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='')
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_torch elapsed time', end_time - start_time)



print('difference of output_b', torch.max(torch.abs(output_b_fft - output_b_torch)).cpu().numpy())
print('difference of grad_k', torch.max(torch.abs(grad_k_fft - grad_k_torch)).cpu().numpy())
print('difference of grad_k_bias', torch.max(torch.abs(grad_k_bias_fft - grad_k_bias_torch)).cpu().numpy())
print('difference of grad_x', torch.max(torch.abs(grad_x_fft - grad_x_torch)).cpu().numpy())
