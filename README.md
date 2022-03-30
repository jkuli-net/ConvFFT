# ConvFFT
Differentiable FFT Conv Layer with Dense Color Channels


This is meant to be a drop in replacement for torch.conv
  functional_conv1d_fft  replaces  torch.nn.functional.conv1d
  Conv1d_fft             replaces  torch.nn.Conv1d
  supports 1d, 2d and 3d convolution
  
This is just an alpha POC release.  
Written in Python, no optimized cuda.
API is not exactly matching torch yet.
unsupported:  stride, dilation, groups, etc

