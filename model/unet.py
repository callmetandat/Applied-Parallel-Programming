import torch.nn as nn
from torch import Tensor
import torch

class Double_Conv(nn.Module):
	def __init__(self, in_channel: int , out_channel: int) -> None:
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, kernel_size= 3, padding= 1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channel, out_channel, kernel_size= 3, padding= 1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True)
		)

	def forward(self, data: Tensor):
		return self.double_conv(data)
	
class Up(nn.Module):
	def __init__(self, in_channel: int, out_channel: int) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size= 2, stride= 2)
		self.double_conv = Double_Conv(in_channel, out_channel)

	# def crop_and_concatenate(self, tensor: Tensor, targetTensor: Tensor):
	# 	"""
	# 	Crop and concatenate two tensor for skip connection step
	# 	"""
	# 	delta_width = (tensor.size()[2] - targetTensor.size()[2])//2
	# 	delta_heigh = (tensor.size()[3] - targetTensor.size()[3])//2

	# 	tensor = tensor[:,:, delta_width: tensor.size()[2]- delta_width, delta_heigh: tensor.size()[3]- delta_heigh]
	# 	return torch.cat([tensor, targetTensor], 1)
		
	def forward(self, inputTensor: Tensor, skipTensor: Tensor):
		"""
		inputTensor: input Tensor for Transpose Conv
		skipTensor: take from Down Conv Layer for concatenate step
		"""

		x = self.up(inputTensor)
		return self.double_conv(torch.cat([x, skipTensor],1))

class Down(nn.Module):
	def __init__(self,input_channel: int, output_channel: int) -> None:
		super().__init__()
		self.max_pooling_and_conv = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride= 2),
			Double_Conv(input_channel, output_channel)
		)

	def forward(self, data: Tensor):
		return self.max_pooling_and_conv(data)
		
class U_net(nn.Module):
	"""
	U-Net model for image segmentation
	"""
	def __init__(self, num_classes: int) -> None:
		"""
		Initializes the U-Net model with the specified number of classes for segmentation.

		Args:
			num_classes (int): The number of classes to segment in the output image.
		"""
		super().__init__()
		self.down_1 = Double_Conv(3, 64)
		self.down_2 = Down(64, 128)
		self.down_3 = Down(128, 256)
		self.down_4 = Down(256, 512)
		self.down_5 = Down(512, 1024)

		self.up_1 = Up(1024, 512)
		self.up_2 = Up(512, 256)
		self.up_3 = Up(256, 128)
		self.up_4 = Up(128, 64)

		self.out = nn.Conv2d(64, num_classes, kernel_size=1)

	def forward(self, img):
		"""
		Forward pass of the U-Net model.	
		Args:
			img (Tensor): Input image tensor.	
		Returns:
			Tensor: Segmentation logits for each pixel class.
		"""
		# Encoder
		d1 = self.down_1(img)
		d2 = self.down_2(d1)
		d3 = self.down_3(d2)
		d4 = self.down_4(d3)
		d5 = self.down_5(d4)
		# Decoder
		u = self.up_1(d5, d4)
		u = self.up_2(u, d3)
		u = self.up_3(u, d2)
		u = self.up_4(u, d1)

		out= self.out(u)
		return out
