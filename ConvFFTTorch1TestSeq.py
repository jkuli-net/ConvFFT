import torch
import time

from ConvFFTTorch1 import Conv1d_fft, Conv2d_fft, Conv3d_fft

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
    model = torch.nn.Sequential(
        Conv1d_fft(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        Conv1d_fft(out_channels, in_channels, kernel_size, padding=padding)
    )
    model_t = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        torch.nn.Conv1d(out_channels, in_channels, kernel_size, padding=padding)
    )
if fft_dim == 2:
    model = torch.nn.Sequential(
        Conv2d_fft(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        Conv2d_fft(out_channels, in_channels, kernel_size, padding=padding)
    )
    model_t = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, in_channels, kernel_size, padding=padding)
    )
if fft_dim == 3:
    model = torch.nn.Sequential(
        Conv3d_fft(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        Conv3d_fft(out_channels, in_channels, kernel_size, padding=padding)
    )
    model_t = torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        torch.nn.ReLU(),
        torch.nn.Conv3d(out_channels, in_channels, kernel_size, padding=padding)
    )
    

x_true = torch.rand((batch_size, in_channels, *data_size))

b_true = model(x_true)
b_true = torch.rand(b_true.shape)

model = model.to(cuda_device)
model_t = model_t.to(cuda_device)

x_true = x_true.to(cuda_device)
b_true = b_true.to(cuda_device)
  
param = list(model.parameters())
param_t = list(model_t.parameters())
for i in range(len(param)):
    param_t[i].data = param[i].data.clone().detach()  
    
lr = 0.0001
steps = 501
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(params=model.parameters(),  lr=lr)
for step in range(steps):    
    b_pred = model(x_true)
    loss = mse_loss(b_pred, b_true)    
    optimizer.zero_grad()
    loss.backward()   
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='')
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_fft elapsed time', end_time - start_time)

optimizer = torch.optim.Adam(params=model_t.parameters(),  lr=lr)
for step in range(steps):    
    b_pred = model_t(x_true)
    loss = mse_loss(b_pred, b_true)    
    optimizer.zero_grad()
    loss.backward()   
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='')
    if step == 0:
        start_time = time.perf_counter()
end_time = time.perf_counter()
print('\nconv_torch elapsed time', end_time - start_time)
