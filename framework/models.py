import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, AvgPool2d, Dropout
import torch.nn.functional as F
from einops import reduce
from dataset import *
from config import *

# Padding utility
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

# Swish activation
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

# Sinusoidal Positional Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
# Conv1d with weight standardization
class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

# Residual Block
class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = inc == outc
        self.ks = kernel_size
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out
    
# Unet Downsampling Block
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x

# Unet Upsampling Block
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x
    
# Time/context embedding MLP
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.PReLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

#conditionalUnet
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat * 1
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 3
        self.d4_out = n_feat * 4

        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)
        # self.timeembed1 = EmbedFC(n_feat, self.u1_out)
        # self.timeembed2 = EmbedFC(n_feat, self.u2_out)
        # self.timeembed3 = EmbedFC(n_feat, self.u3_out)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=8, factor=2)

        self.up2 = UnetUp(self.d3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.u2_out + self.d2_out, self.u3_out, 1, gn=8, factor=2)
        if task == "SSVEP":
            self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=8, factor=2)
        elif task == "MI":
            self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=1, factor=2)
        else:
            print(f"Warning: Unknown task config '{taks}'. Defaulting to 'SSVEP'")
            self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=8, factor=2) 
        self.out = nn.Conv1d(self.u4_out + in_channels, in_channels, 1)

    def forward(self, x, t):
        down1 = self.down1(x)  # 2000 -> 1000
        down2 = self.down2(down1)  # 1000 -> 500
        down3 = self.down3(down2)  # 500 -> 250

        temb = self.sin_emb(t).view(-1, self.n_feat, 1)  # [b, n_feat, 1]

        up1 = self.up2(down3)  # 250 -> 500
        up2 = self.up3(torch.cat([up1 + temb, down2], 1))  # 500 -> 1000

        # Align the temporal dimension of up2 + temb and down1
        if (up2 + temb).shape[-1] != down1.shape[-1]:
            target_len = min((up2 + temb).shape[-1], down1.shape[-1])
            up2 = F.interpolate(up2, size=target_len)
            down1 = F.interpolate(down1, size=target_len)

        up3 = self.up4(torch.cat([up2 + temb, down1], 1))  # 1000 -> 2000

        # Align the temporal dimension of up3 and x
        if up3.shape[-1] != x.shape[-1]:
            target_len = min(up3.shape[-1], x.shape[-1])
            up3 = F.interpolate(up3, size=target_len)
            x = F.interpolate(x, size=target_len)

        out = self.out(torch.cat([up3, x], 1))  # 2000 -> 2000

        down = (down1, down2, down3)
        up = (up1, up2, up3)
        return out, down, up

# Attention pooling
class AttentionPool1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(in_channels))  # 用 0 初始化稳定收敛

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        scores = torch.einsum('bct,c->bt', x, self.query)       # [B, T]
        weights = torch.softmax(scores, dim=-1)                 # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(1), dim=-1)    # [B, C]
        return pooled

# Encoder(original)
class Encoder(nn.Module):
    def __init__(self, in_channels, dim=512):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.e1_out = dim
        self.e2_out = dim
        self.e3_out = dim

        self.down1 = UnetDown(in_channels, self.e1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.e2_out, self.e3_out, 1, gn=8, factor=2)

        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.act = nn.Tanh()

    def forward(self, x0):
        # Down sampling
        dn1 = self.down1(x0)  # 2048 -> 1024
        dn2 = self.down2(dn1)  # 1024 -> 512
        dn3 = self.down3(dn2)  # 512 -> 256
        z = self.avg_pooling(dn3).view(-1, self.e3_out)  # [b, features]
        down = (dn1, dn2, dn3)
        out = (down, z)
        return out
    
#eegnet-style Encoder
class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=250, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, F3=32, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        # Dropout setting
        if dropoutType == 'SpatialDropout2D':
            self.dropout1 = nn.Dropout2d(p=dropoutRate)
            self.dropout2 = nn.Dropout2d(p=dropoutRate)
            self.dropout3 = nn.Dropout2d(p=dropoutRate)
        else:
            self.dropout1 = nn.Dropout(p=dropoutRate)
            self.dropout2 = nn.Dropout(p=dropoutRate)
            self.dropout3 = nn.Dropout(p=dropoutRate)

        # Block 1
        self.conv1 = Conv2d(1, F1, kernel_size=(1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = BatchNorm2d(F1)
        self.depthwise_conv = Conv2d(F1, F1, kernel_size=(Chans, 1), groups=F1, bias=False)
        self.bn2 = BatchNorm2d(F1)
        self.activation1 = nn.ELU()
        self.pool1 = AvgPool2d(kernel_size=(1, 2))  # 250 → 125

        # Block 2
        self.sep_conv = Conv2d(F1, F2, kernel_size=(1, 16), padding='same', bias=False)
        self.bn3 = BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = AvgPool2d(kernel_size=(1, 2))  # 125 → 62

        # Block 3 (newly added)
        self.conv3 = Conv2d(F2, F3, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn4 = BatchNorm2d(F3)
        self.activation3 = nn.ELU()
        self.pool3 = AvgPool2d(kernel_size=(1, 2))  # 62 → 31

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F3 * 31, nb_classes)

        self.proj1 = nn.Conv1d(F1, 256, kernel_size=1)
        self.proj2 = nn.Conv1d(F2, 256, kernel_size=1)
        self.proj3 = nn.Conv1d(F3, 256, kernel_size=1)

        self.att_pool = AttentionPool1d(256)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 64, 250]

        # Block 1
        dn1 = self.conv1(x)
        dn1 = self.bn1(dn1)
        dn1 = self.depthwise_conv(dn1)
        dn1 = self.bn2(dn1)
        dn1 = self.activation1(dn1)
        dn1 = self.pool1(dn1)
        dn1 = self.dropout1(dn1)

        # Block 2
        dn2 = self.sep_conv(dn1)
        dn2 = self.bn3(dn2)
        dn2 = self.activation2(dn2)
        dn2 = self.pool2(dn2)
        dn2 = self.dropout2(dn2)

        # Block 3
        dn3 = self.conv3(dn2)
        dn3 = self.bn4(dn3)
        dn3 = self.activation3(dn3)
        dn3 = self.pool3(dn3)
        dn3 = self.dropout3(dn3)

        # Squeeze to [B, C, T]
        dn1_ = dn1.squeeze(2)  # e.g. [B, F1, 125]
        dn2_ = dn2.squeeze(2)  # e.g. [B, F2, 62]
        dn3_ = dn3.squeeze(2)  # e.g. [B, F3, 31]

        # Project to the number of channels needed by the decoder (original encoder: 256)

        dn1_out = self.proj1(dn1_)  # [B, 256, 125]
        dn2_out = self.proj2(dn2_)  # [B, 256, 62]
        dn3_out = self.proj3(dn3_)  # [B, 256, 31]

        # z vector is obtained by average pooling on dn3
        #z = torch.mean(dn3_out, dim=-1)  # [B, 256]
        z = self.att_pool(dn3_out)  # [B, 256]

        down = (dn1_out, dn2_out, dn3_out)
        return (down, z)

# Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        self.e3_out = encoder_dim
        self.d1_out = n_feat
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 3
        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.z_proj = nn.Linear(self.e3_out, self.d1_out)

        # Unet up sampling
        self.up1 = UnetUp(self.d3_out + self.e3_out, self.u2_out, 1, gn=8, factor=2)
        self.up2 = UnetUp(self.d2_out + self.u2_out, self.u3_out, 1, gn=8, factor=2)
        
        # Configure the upsampling layer based on decoder_input configuration
        if decoder_input == "x + x_hat + skips":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + self.u3_out + in_channels * 2, in_channels, 1, 1, 0))
        elif decoder_input == "x + x_hat":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(in_channels * 2, in_channels, 1, 1, 0))
        elif decoder_input == "x_hat + skips":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + self.u3_out + in_channels, in_channels, 1, 1, 0))
        elif decoder_input == "x + skips":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + self.u3_out + in_channels, in_channels, 1, 1, 0))
        elif decoder_input == "skips":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + self.u3_out, in_channels, 1, 1, 0))
        elif decoder_input == "z only":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out, in_channels, 1, 1, 0))
        elif decoder_input == "z + x":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + in_channels, in_channels, 1, 1, 0))
        elif decoder_input == "z + x_hat":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + in_channels, in_channels, 1, 1, 0))
        elif decoder_input == "z + skips":
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + self.u3_out + self.d1_out, in_channels, 1, 1, 0))
        else:
            print(f"Warning: Unknown decoder_input config '{decoder_input}'. Defaulting to 'z + x'")
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(self.d1_out + in_channels, in_channels, 1, 1, 0))

        self.pool = nn.AvgPool1d(2)

    def forward(self, x0, encoder_out, diffusion_out):
        # Encoder output
        down, z = encoder_out
        dn1, dn2, dn3 = down

        # DDPM output
        if diffusion_out is None:
            # Create dummy tensors with appropriate shapes
            B, C, T = x0.shape
            x_hat = x0.clone()  # Use input as a placeholder
            
            # Don't use encoder features directly - create properly sized placeholders
            # Get the right channel sizes that match what'd come from the DDPM
            B, C_dn3, T_dn3 = dn3.shape
            B, C_dn2, T_dn2 = dn2.shape
            B, C_dn1, T_dn1 = dn1.shape
            
            # Calculate expected channel dimensions for DDPM features
            expected_ch_dn1 = self.d1_out  # Likely 256
            expected_ch_dn2 = self.d2_out  # Likely 512 
            expected_ch_dn3 = self.d3_out  # Likely 768
            
            dn11 = torch.zeros(B, expected_ch_dn1, T_dn1, device=dn1.device)
            dn22 = torch.zeros(B, expected_ch_dn2, T_dn2, device=dn2.device)
            dn33 = torch.zeros(B, expected_ch_dn3, T_dn3, device=dn3.device)
        else:
            # DDPM output
            x_hat, down_ddpm, up, t = diffusion_out
            dn11, dn22, dn33 = down_ddpm
        
        # Calculate expected input channels for up1 layer
        expected_up1_channels = self.d3_out + self.e3_out
        
        # Check if we need to adjust the channels
        if dn3.shape[1] + dn33.shape[1] != expected_up1_channels:
            print(f"Shape mismatch in up1: dn3 {dn3.shape}, dn33 {dn33.shape}, expected total {expected_up1_channels}")
            
            # Safely adjust the channels for the dummy case
            if diffusion_out is None:
                # Create a concatenated tensor with the right number of channels
                concat_in = torch.cat([dn3, dn33], dim=1)
                
                # Create a new tensor with the correct number of channels needed by up1
                if concat_in.shape[1] < expected_up1_channels:
                    # Need to add channels
                    missing_channels = expected_up1_channels - concat_in.shape[1]
                    padding = torch.zeros(B, missing_channels, T_dn3, device=dn3.device)
                    concat_in = torch.cat([concat_in, padding], dim=1)
                else:
                    # Need to reduce channels
                    concat_in = concat_in[:, :expected_up1_channels, :]
                    
                # Pass the properly sized tensor to up1
                up1 = self.up1(concat_in)
            else:
                # This shouldn't happen with real DDPM output
                raise ValueError(f"Channel mismatch with real DDPM: {dn3.shape[1]} + {dn33.shape[1]} != {expected_up1_channels}")
        else:
            # Normal case - shapes match
            up1 = self.up1(torch.cat([dn3, dn33.detach()], 1))
        
        # Rest of the function remains the same...
        # Calculate expected input channels for up2 layer
        expected_up2_channels = self.u2_out + self.d2_out
        
        # Check if we need to adjust the channels
        if up1.shape[1] + dn22.shape[1] != expected_up2_channels:
            print(f"Shape mismatch in up2: up1 {up1.shape}, dn22 {dn22.shape}, expected total {expected_up2_channels}")
            
            # Safely adjust the channels for the dummy case
            if diffusion_out is None:
                # Create a concatenated tensor with the right number of channels
                concat_in = torch.cat([up1, dn22], dim=1)
                
                # Create a new tensor with the correct number of channels needed by up2
                if concat_in.shape[1] < expected_up2_channels:
                    # Need to add channels
                    missing_channels = expected_up2_channels - concat_in.shape[1]
                    padding = torch.zeros(B, missing_channels, T_dn2, device=dn2.device)
                    concat_in = torch.cat([concat_in, padding], dim=1)
                else:
                    # Need to reduce channels
                    concat_in = concat_in[:, :expected_up2_channels, :]
                    
                # Pass the properly sized tensor to up2
                up2 = self.up2(concat_in)
            else:
                # This shouldn't happen with real DDPM output
                raise ValueError(f"Channel mismatch with real DDPM: {up1.shape[1]} + {dn22.shape[1]} != {expected_up2_channels}")
        else:
            # Normal case - shapes match
            up2 = self.up2(torch.cat([up1, dn22.detach()], 1))

        # Project z vector to feature space if needed
        B, _, T = dn11.shape
        z_proj = self.z_proj(z).unsqueeze(-1).expand(-1, -1, T)  # [B, 256] → [B, 256, T]
    
        if decoder_input == "x + x_hat + skips":
            out = self.up3(torch.cat([self.pool(x0), self.pool(x_hat.detach()), up2, dn11.detach()], 1))
        elif decoder_input == "x + x_hat":
            out = self.up3(torch.cat([self.pool(x0), self.pool(x_hat.detach())], 1))
        elif decoder_input == "x_hat + skips":
            out = self.up3(torch.cat([self.pool(x_hat.detach()), up2, dn11.detach()], 1))
        elif decoder_input == "x + skips":
            out = self.up3(torch.cat([self.pool(x0), up2, dn11.detach()], 1))
        elif decoder_input == "skips":
            out = self.up3(torch.cat([up2, dn11.detach()], 1))
        elif decoder_input == "z only":
            out = self.up3(z_proj)
        elif decoder_input == "z + x":
            out = self.up3(torch.cat([z_proj, self.pool(x0)], 1))
        elif decoder_input == "z + x_hat":
            out = self.up3(torch.cat([z_proj, self.pool(x_hat.detach())], 1))
        elif decoder_input == "z + skips":
            out = self.up3(torch.cat([z_proj, up2, dn11.detach()], 1))
        else:
            out = self.up3(torch.cat([z_proj, self.pool(x0)], 1))

        return out

class DiffE(nn.Module):
    def __init__(self, encoder, decoder, fc):
        super(DiffE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x0, ddpm_out):
        encoder_out = self.encoder(x0)
        z = encoder_out[1]  
        
        # Only call decoder if it exists
        if self.decoder is not None:
            decoder_out = self.decoder(x0, encoder_out, ddpm_out)
        else:
            decoder_out = None # If no decoder, return None for decoder output

        # Pass the appropriate input type directly to the classifier
        if classifier_input == "z":
            fc_in = z  # [B, 256]
        elif classifier_input == "x":
            fc_in = x0  # [B, 64, 250] 
        elif classifier_input == "x_hat" and ddpm_out is not None:
            fc_in = ddpm_out[0].detach()  # [B, 64, 250]
        elif classifier_input == "decoder_out" and decoder_out is not None:
            fc_in = decoder_out.detach()  # [B, 64, 250]
        else:
            fc_in = z  # Default fallback

        fc_out = self.fc(fc_in)
        return decoder_out, fc_out, z
    
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def ddpm_schedules(beta1, beta2, T):
    # assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = cosine_beta_schedule(T, s=0.008).float()
    # beta_t = sigmoid_beta_schedule(T).float()

    alpha_t = 1 - beta_t

    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)

    return {
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device

    def forward(self, x):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise
        times = _ts / self.n_T
        output, down, up = self.nn_model(x_t, times)
        return output, down, up, noise, times

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)
    
# Final classification head
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, latent_dim, emb_dim):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=emb_dim))
        self.eeg_proj = nn.Conv1d(64, 256, kernel_size=1)  # assumes input is [B, 64, T]
        self.att_pool = AttentionPool1d(256)

    def forward(self, x):
        if x.dim() == 2:
            return self.linear_out(x)
        elif x.dim() == 3: # [B, 64, T]
            x = self.eeg_proj(x)  # [B, 256, T]
            x = self.att_pool(x)  # [B, in_dim]
            return self.linear_out(x)
        else:
            raise ValueError(f"Unexpected input shape to LinearClassifier: {x.shape}")
        
class EEGNetClassifier(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNetClassifier, self).__init__()
        if dropoutType == 'Dropout':
            DropoutClass = nn.Dropout
        elif dropoutType == 'SpatialDropout2D':
            DropoutClass = lambda p: nn.Dropout2d(p)
        else:
            raise ValueError("dropoutType must be 'Dropout' or 'SpatialDropout2D'")

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwiseConv = nn.Conv2d(
            F1, F1 * D, kernel_size=(Chans, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = DropoutClass(dropoutRate)

        # Block 2
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding='same', bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            F1 * D, F2, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = DropoutClass(dropoutRate)

        # Final dense layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * ((Samples // 4) // 8), nb_classes)

    def forward(self, x):  # expected input -> (N, 1, Chans, Samples)
        
        if len(x.shape) == 2: # Project z into a compatible 3D shape to pass through conv layers
            x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 128)  # [B, 256, 1, 128] → simulate EEG shape
        
        if len(x.shape) == 3:  # (N, Chans, Samples) -> [B, 64, 250]
            x = x.unsqueeze(1) # Add the channel dimension -> (N, 1, Chans, Samples)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.drop1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        x = self.dense(x) # produced result -> (N, nb_classes) -> [B, 26]

        return x