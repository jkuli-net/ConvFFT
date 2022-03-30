# ConvFFT
Differentiable FFT Conv Layer with Dense Color Channels


This is meant to be a drop in replacement for torch.conv<br>
<UL>
<LI>functional_conv1d_fft  replaces  torch.nn.functional.conv1d<br>
<LI>Conv1d_fft             replaces  torch.nn.Conv1d<br>
<LI>supports 1d, 2d and 3d convolution<br>
</UL>
 
This is just an alpha POC release.  <br>
Written in Python, no optimized cuda.<br>
API is not exactly matching torch yet.<br>
unsupported:  stride, dilation, groups, etc<br>

