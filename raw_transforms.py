import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn.functional as F
from typing import Literal, Tuple, List, Union
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
)


def _masks_CFA_Bayer_torch(
    shape: Tuple[int, int],
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates RGB masks for a given Bayer pattern in PyTorch.
    """
    pattern = pattern.upper()
    if len(pattern) != 4:
        raise ValueError("Pattern must be a 4-character string (e.g., 'RGGB')")

    channels = {c: torch.zeros((2, 2), device=device) for c in "RGB"}
    for i, char in enumerate(pattern):
        channels[char][i // 2, i % 2] = 1

    reps = ((shape[0] + 1) // 2, ((shape[1] + 1) // 2))
    R_m = channels['R'].tile(reps)[:shape[0], :shape[1]]
    G_m = channels['G'].tile(reps)[:shape[0], :shape[1]]
    B_m = channels['B'].tile(reps)[:shape[0], :shape[1]]

    return R_m, G_m, B_m


def _conv1d(
    x: torch.Tensor,
    y: torch.Tensor,
    axis: int
) -> torch.Tensor:
    """
    Performs 1D convolution on a 2D tensor using PyTorch's 2D convolution.
    'axis=0' for vertical, 'axis=1' for horizontal.
    Uses 'reflect' padding, with a fallback to 'replicate' for small inputs.
    """
    k_size = y.shape[0]
    padding_val = (k_size - 1) // 2

    x_unsqueezed = x.unsqueeze(0).unsqueeze(0)
    y = y.to(x.device)

    if axis == 1:  # Horizontal
        dim_size = x.shape[1]
        kernel = y.view(1, 1, 1, -1)
        padding_tuple = (padding_val, padding_val, 0, 0)
    elif axis == 0:  # Vertical
        dim_size = x.shape[0]
        kernel = y.view(1, 1, -1, 1)
        padding_tuple = (0, 0, padding_val, padding_val)
    else:
        raise ValueError("Axis must be 0 (vertical) or 1 (horizontal)")

    pad_mode = 'reflect' if padding_val < dim_size else 'replicate'
    x_padded = F.pad(x_unsqueezed, padding_tuple, mode=pad_mode)
    return F.conv2d(x_padded, kernel, padding=0).squeeze()


def bayer2rgb_menon2007_pytorch(
    CFA: torch.Tensor,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
) -> torch.Tensor:
    """
    Returns the demosaiced RGB image from a Bayer CFA using the Menon (2007)
    algorithm, implemented in PyTorch.
    """
    device = CFA.device
    CFA = CFA.float()
    R_m, G_m, B_m = _masks_CFA_Bayer_torch(CFA.shape, pattern, device)

    h_0 = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0], device=device, dtype=torch.float)
    h_1 = torch.tensor([-0.25, 0.0, 0.5, 0.0, -0.25], device=device, dtype=torch.float)

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = torch.where(G_m == 0, _conv1d(CFA, h_0, axis=1) + _conv1d(CFA, h_1, axis=1), G)
    G_V = torch.where(G_m == 0, _conv1d(CFA, h_0, axis=0) + _conv1d(CFA, h_1, axis=0), G)

    C_H = torch.where(R_m == 1, R - G_H, torch.tensor(0., device=device))
    C_H = torch.where(B_m == 1, B - G_H, C_H)
    C_V = torch.where(R_m == 1, R - G_V, torch.tensor(0., device=device))
    C_V = torch.where(B_m == 1, B - G_V, C_V)

    # Directional derivatives (gradients) of the color difference
    C_H_4d = C_H.unsqueeze(0).unsqueeze(0)
    C_V_4d = C_V.unsqueeze(0).unsqueeze(0)

    # --- Calculate D_H (Horizontal Gradient) ---
    pad_mode_h = 'reflect' if C_H.shape[1] > 2 else 'replicate'
    C_H_padded = F.pad(C_H_4d, (0, 2, 0, 0), mode=pad_mode_h)
    C_H_shifted = C_H_padded[:, :, :, 2:].squeeze()
    D_H = torch.abs(C_H - C_H_shifted)

    # --- Calculate D_V (Vertical Gradient) ---
    pad_mode_v = 'reflect' if C_V.shape[0] > 2 else 'replicate'
    C_V_padded = F.pad(C_V_4d, (0, 0, 0, 2), mode=pad_mode_v)
    C_V_shifted = C_V_padded[:, :, 2:, :].squeeze()
    D_V = torch.abs(C_V - C_V_shifted)

    k = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ], device=device, dtype=torch.float)

    d_H = _conv1d(D_H, k[2, :], axis=1) + _conv1d(D_H, k[:, 2], axis=0) - D_H * k[2,2] # Using separable convolution for efficiency
    d_V = _conv1d(D_V, k[2, :], axis=1) + _conv1d(D_V, k[:, 2], axis=0) - D_V * k[2,2]
    
    mask = (d_V >= d_H).float()
    G = torch.where(G_m == 1, G, mask * G_H + (1 - mask) * G_V)
    M = mask

    R_r = torch.any(R_m == 1, dim=1, keepdim=True).expand_as(R)
    B_r = torch.any(B_m == 1, dim=1, keepdim=True).expand_as(B)

    k_b = torch.tensor([0.5, 0, 0.5], device=device, dtype=torch.float)

    R = torch.where(torch.logical_and(G_m == 1, R_r), G + _conv1d(R, k_b, 1) - _conv1d(G, k_b, 1), R)
    R = torch.where(torch.logical_and(G_m == 1, B_r), G + _conv1d(R, k_b, 0) - _conv1d(G, k_b, 0), R)
    B = torch.where(torch.logical_and(G_m == 1, B_r), G + _conv1d(B, k_b, 1) - _conv1d(G, k_b, 1), B)
    B = torch.where(torch.logical_and(G_m == 1, R_r), G + _conv1d(B, k_b, 0) - _conv1d(G, k_b, 0), B)

    R_at_B = M * (B + _conv1d(R, k_b, 1) - _conv1d(B, k_b, 1)) + \
             (1 - M) * (B + _conv1d(R, k_b, 0) - _conv1d(B, k_b, 0))
    R = torch.where(torch.logical_and(B_r, B_m == 1), R_at_B, R)

    B_at_R = M * (R + _conv1d(B, k_b, 1) - _conv1d(R, k_b, 1)) + \
             (1 - M) * (R + _conv1d(B, k_b, 0) - _conv1d(R, k_b, 0))
    B = torch.where(torch.logical_and(R_r, R_m == 1), B_at_R, B)

    RGB = torch.stack([R, G, B], dim=0)

    return RGB


class RawToRgbTransform(nn.Module):
    """
    A PyTorch transform to convert a 16-bit raw Bayer image to a processed
    8-bit sRGB image tensor.

    The pipeline includes:
    1. Scaling 16-bit input to [0, 1].
    2. Demosaicing using the Menon (2007) algorithm.
    3. Applying white balance.
    4. Clipping and contrast stretching.
    5. Applying gamma correction.
    6. Resizing to a fixed output dimension.

    Args:
        output_size (Union[int, Tuple[int, int]]): The final output size.
        wb_factors (List[float]): White balance multipliers for [R, G, B].
        gamma (float): The gamma correction factor.
        pattern (str): The Bayer pattern for demosaicing (e.g., 'RGGB').
    """
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 256,
        wb_factors: List[float] = [1.6, 1.2, 1.4],
        gamma: float = 2.2,
        pattern: str = 'RGGB'
    ):
        super().__init__()
        if not (0 < gamma):
            raise ValueError("Gamma must be a positive number.")
        
        self.pattern = pattern.upper()

        # --- Instantiate sub-modules for the pipeline ---
        self.resize = v2.Resize(output_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
        
        # --- Store parameters ---
        self.gamma_exponent = 1.0 / gamma
        
        # Register white balance factors as a buffer for proper device placement
        # and reshape for broadcasting: (C, 1, 1)
        wb_tensor = torch.tensor(wb_factors, dtype=torch.float32).view(3, 1, 1)
        self.register_buffer('wb_factors', wb_tensor)

    def forward(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the full raw-to-RGB processing pipeline.

        Args:
            raw_tensor (torch.Tensor): A 16-bit raw image tensor of shape (H, W).

        Returns:
            torch.Tensor: A processed RGB tensor of shape (3, H, W).
        """
        # --- 1. Input Handling and Scaling ---
        # Ensure tensor is 2D (H, W)
        if raw_tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, but got shape {raw_tensor.shape}")

        # Scale from 16-bit [0, 65535] range to float [0, 1]
        scaled_raw = raw_tensor.float() / 65535.0

        # --- 2. Demosaic ---
        # Input is a 2D tensor, output is a (3, H, W) tensor
        rgb = bayer2rgb_menon2007_pytorch(scaled_raw, pattern=self.pattern)

        
        # --- 3. White Balance ---
        rgb = rgb * self.wb_factors
        rgb = torch.clamp(rgb, 0.0, 1.0) # Clip to valid range after WB

        # --- 4. Contrast Stretching (Min-Max Normalization) ---
        min_val = torch.min(rgb)
        max_val = torch.max(rgb)
        range_val = max_val - min_val
        
        # Avoid division by zero for solid color images
        if range_val > 1e-6:
            rgb = (rgb - min_val) / range_val

        # --- 5. Gamma Correction ---
        rgb = torch.pow(rgb, self.gamma_exponent)

        return rgb

class RawTransform(nn.Module):
    """
    A custom PyTorch transform to process a 2D raw Bayer image.

    This module performs a 4-channel "packing" of the raw Bayer data,
    then normalizes it using specified black and white levels. It's a
    PyTorch-native reimplementation of the provided NumPy-based logic.
    """
    def __init__(self, black_levels=[62.0, 60.0, 58.0, 61.0], white_level=1023.0):
        super().__init__()
        
        self.register_buffer('black_levels', torch.tensor(black_levels, dtype=torch.float32).view(4, 1, 1))
        
        self.white_level = white_level

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        packed_raw = torch.stack([
            raw_image[1::2, 1::2],  # Blue channel
            raw_image[0::2, 1::2],  # Green channel at blue row
            raw_image[0::2, 0::2],  # Red channel
            raw_image[1::2, 0::2],  # Green channel at red row
        ], dim=0)

        # --- 3. Normalize channels ---
        # Convert to float for calculation and subtract black levels
        normalized = packed_raw.float() - self.black_levels
        # Divide by the dynamic range (white_level - black_levels)
        normalized /= (self.white_level - self.black_levels)
        
        # --- 4. Clamp values ---
        # Ensure final values are clipped to the standard [0, 1] range
        return torch.clamp(normalized, 0.0, 1.0)
    


raw_transform_ = Compose([
            torch.from_numpy,
            RawTransform(),
        ])
raw_to_rgb_transform_ = Compose([
            torch.from_numpy,
            RawToRgbTransform(),
        ])

rgb_transform_ = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])