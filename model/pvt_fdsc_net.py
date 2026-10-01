import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .pvtv2 import pvt_v2_b2

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=True, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x))

class AdapterLayer(nn.Module):
    """ From PVTAdpNet: Enhances dense prediction expressiveness"""
    def __init__(self, in_channels, reduced_channels=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)

class GSA(nn.Module):
    """ From ResPVT: Fuses high-level semantic cues (F2, F3, F4) """
    def __init__(self, channels=32):
        super().__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.reduce_concat = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Backbone channel reduction to 32
        self.reduce2 = nn.Conv2d(128, 32, kernel_size=1)
        self.reduce3 = nn.Conv2d(320, 32, kernel_size=1)
        self.reduce4 = nn.Conv2d(512, 32, kernel_size=1)

    def forward(self, f2, f3, f4):
        # 1. Initial Channel Reduction
        f2, f3, f4 = self.reduce2(f2), self.reduce3(f3), self.reduce4(f4)
        
        # 2. M2 Fusion (High-level: F3 and F4)
        # f4 is upsampled to match f3 (e.g., 11x11 -> 22x22)
        f4_up = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        m2_attn = self.conv_unit(f4_up)
        m2 = self.reduce_concat(torch.cat([f3 * m2_attn, m2_attn], dim=1))
        m2 = self.conv_unit(m2) 
        
        # 3. M1 Fusion (Mid-level: F2, F3, and F4)
        # All must match f2 spatial size (e.g., 44x44)
        f3_up_to_f2 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        f4_up_to_f2 = F.interpolate(f4, scale_factor=4, mode='bilinear', align_corners=True)
        m1 = f2 * self.conv_unit(f3_up_to_f2) * self.conv_unit(f4_up_to_f2)
        
        # 4. Final semantic cue O1
        # FIX: M2 is currently size 22, M1 is size 44. Upsample M2 to 44.
        m2_up_to_f2 = F.interpolate(m2, scale_factor=2, mode='bilinear', align_corners=True)
        
        o1 = self.reduce_concat(torch.cat([m1, m2_up_to_f2], dim=1))
        o1 = self.conv_unit(o1)
        
        # FIX: Upsample final output by 2 to reach 88x88
        return F.interpolate(o1, scale_factor=2, mode='bilinear', align_corners=True)

class LSE(nn.Module):
    """ From ResPVT: Extracts low-level features (edges/color) from F1"""
    def __init__(self, in_c=64, out_c=32):
        super().__init__()
        # c1 reduces channels from backbone (64) to internal processing (32)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        
        # c3 and internal layers now only process 'out_c' (32 channels)
        self.c3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.c1_internal = nn.Conv2d(out_c, out_c, kernel_size=1) # FIX: Use internal 32-ch version
        
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f1):
        # 1. Initial channel reduction from 64 to 32 
        f1_reduced = self.relu(self.bn(self.c1(f1)))
        identity = f1_reduced # Identity is now 32 channels

        # 2. Main root extraction (all operations now 32-channel internal)
        out = self.relu(self.bn(self.c1_internal(f1_reduced))) # Use 32-ch internal conv
        out = self.relu(self.bn(self.c3(out)))
        out = self.bn(self.c1_internal(out)) # Use 32-ch internal conv
        
        # 3. Skip connection addition
        out = self.relu(out + identity)
        
        # 4. Upsample to match decoder spatial size
        return self.c3(F.interpolate(out, scale_factor=1, mode='bilinear', align_corners=True))

class FDSC(nn.Module):
    def __init__(self, channels=32):
        super(FDSC, self).__init__()
        # Learnable weights for Amplitude and Phase fusion
        # Initialized to 0.5 to start with an equal balance
        self.alpha = nn.Parameter(torch.tensor([0.5])) # Amplitude weight
        self.beta = nn.Parameter(torch.tensor([0.5]))  # Phase weight
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_feat, decoder_feat):
        """
        encoder_feat: High-freq structural cues (o2)
        decoder_feat: Low-freq semantic cues (o1)
        """
        # 1. Transform to Frequency Domain
        # rfft2 is used for real-valued spatial data
        freq_e = torch.fft.rfft2(encoder_feat, norm='backward')
        freq_d = torch.fft.rfft2(decoder_feat, norm='backward')

        # 2. Extract Amplitude (A) and Phase (P)
        amp_e, pha_e = torch.abs(freq_e), torch.angle(freq_e)
        amp_d, pha_d = torch.abs(freq_d), torch.angle(freq_d)

        # 3. Mathematical Selective Fusion
        # Sigmoid ensures weights stay between 0 and 1
        a_weight = torch.sigmoid(self.alpha)
        p_weight = torch.sigmoid(self.beta)

        # Weighted Amplitude (favoring decoder semantics)
        amp_fused = a_weight * amp_d + (1 - a_weight) * amp_e
        
        # Weighted Phase (favoring encoder boundaries/edges)
        # Note: We use the encoder phase as the anchor for boundary alignment
        pha_fused = p_weight * pha_e + (1 - p_weight) * pha_d

        # 4. Reconstruct Complex Frequency Map
        # Re-combining: A * exp(j * P)
        fused_complex = amp_fused * torch.exp(1j * pha_fused)

        # 5. Inverse DFT back to Spatial Domain
        fused_spatial = torch.fft.irfft2(fused_complex, s=encoder_feat.shape[-2:], norm='backward')
        
        return self.refine(fused_spatial)

class PVTFDSCNet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.backbone = pvt_v2_b2()  ## [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # Adapter-based Skip Connections
        self.adp1 = AdapterLayer(64)
        self.adp2 = AdapterLayer(128)
        self.adp3 = AdapterLayer(320)
        self.adp4 = AdapterLayer(512)

        # Decoder Components from ResPVT
        self.fm = GSA() # Semantic cues
        self.structural_res = LSE(64, 32) # Color/Edges

        # NEW: FDSC Module
        self.fdsc = FDSC(channels=32)

        # Channels: o1(32) + o2(32) + fdsc_out(32) + s1(64) = 160 channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, inputs):
        # Encoder
        e1, e2, e3, e4 = self.backbone(inputs)
        
        # Apply Adapters to Skip Connections
        s1, s2, s3, s4 = self.adp1(e1), self.adp2(e2), self.adp3(e3), self.adp4(e4)

        # High-level Semantic Fusion (O1)
        o1 = self.fm(s2, s3, s4)
        
        # Low-level Structural Extraction (O2)
        o2 = self.structural_res(s1)

        # 3. FDSC Spectral Fusion
        o_fused = self.fdsc(o2, o1)   # Mathematical Spectral Integration

        # 4. Dense Concatenation (All cues + Raw Adapter)
        # Note: All components are [B, C, H/4, W/4]
        combined = torch.cat([o1, o2, o_fused, s1], dim=1)
        
        # 5. Prediction Head
        x = self.final_conv(combined)
        
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = PVTFDSCNet()
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}") 