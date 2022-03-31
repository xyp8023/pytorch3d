# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Sequence, Union

import torch
from pytorch3d import _C


# Example functions for blending the top K colors per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return an RGBA image per batch element


class BlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): Controls the width of the sigmoid function used to
            calculate the 2D distance based probability. Determines the
            sharpness of the edges of the shape.
            Higher => faces have less defined edges.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)


def hard_rgb_blend(
    colors: torch.Tensor, fragments, blend_params: BlendParams
) -> torch.Tensor:
    """
    Naive blending of top K faces to return an RGBA image
      - **RGB** - choose color of the closest point i.e. K=0
      - **A** - 1.0

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = colors.new_tensor(background_color_)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


# Wrapper for the C++/CUDA Implementation of sigmoid alpha blend.
class _SigmoidAlphaBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dists, pix_to_face, sigma):
        alphas = _C.sigmoid_alpha_blend(dists, pix_to_face, sigma)
        ctx.save_for_backward(dists, pix_to_face, alphas)
        ctx.sigma = sigma
        return alphas

    @staticmethod
    def backward(ctx, grad_alphas):
        dists, pix_to_face, alphas = ctx.saved_tensors
        sigma = ctx.sigma
        grad_dists = _C.sigmoid_alpha_blend_backward(
            grad_alphas, alphas, dists, pix_to_face, sigma
        )
        return grad_dists, None, None


# pyre-fixme[16]: `_SigmoidAlphaBlend` has no attribute `apply`.
_sigmoid_alpha = _SigmoidAlphaBlend.apply


def sigmoid_alpha_blend(colors, fragments, blend_params: BlendParams) -> torch.Tensor:
    """
    Silhouette blending to return an RGBA image
      - **RGB** - choose color of the closest point.
      - **A** - blend based on the 2D distance based probability map [1].

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """
    N, H, W, K = fragments.pix_to_face.shape
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    pixel_colors[..., :3] = colors[..., 0, :]
    alpha = _sigmoid_alpha(fragments.dists, fragments.pix_to_face, blend_params.sigma)
    pixel_colors[..., 3] = alpha
    return pixel_colors


def softmax_rgb_blend(
    colors: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        znear = znear[:, None, None, None]

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors

def nan_to_num(y, num=0.0):
    """Helper to handle indices and logical indices of NaNs.

    """
    y[torch.isnan(y)]=num
    return y

def zeros_to_num(y, num=1.0):
    """Helper to handle indices and logical indices of zeros.

    """
    y[y==0]=num
    return y

def softmax_sss_blend_p(
    colors: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    **kwargs    
) -> torch.Tensor:
    """
    In parallel!

    SSS intensity blending to return an SSS image based on the method
    proposed in [1]
      - **SSS** - blend the colors based on the 2D distance based probability map and
        relative z distances.
    Args:
        colors: (N, H, W, K) intensity color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.


    Returns:
        
        SSS (N, nbr_time_bins) 
    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    N, H, W, K = fragments.pix_to_face.shape # H corresponds to the vertical beam angle; W corresponds to the horizontal beam angle
    device = fragments.pix_to_face.device
    nbr_time_bins = kwargs.get("nbr_time_bins", 256) 
    max_slant_range = kwargs.get("max_slant_range", 50.0)
    beam_pattern = kwargs.get("beam_pattern", torch.ones((1,1,W,1))).to(device) # pixel_colors: (N, H, W, nbr_time_bins)
    topk_angles = kwargs.get("topk_angles", 10)  
    eps_ = kwargs.get("eps_", 1e-6)  
    valid_mean = kwargs.get("valid_mean", False)  
    

    sss_rendered = torch.zeros((N, nbr_time_bins), dtype=colors.dtype, device=colors.device)
    
    # Weight for background color
    mask = torch.logical_and((fragments.pix_to_face >= 0), fragments.zbuf<=max_slant_range+max_slant_range/nbr_time_bins)
    mask = torch.logical_and(mask, (fragments.dists>-1)) # dists==-1 the same as pix_to_face<0; dists<-1 bug? when the fov is too small 
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask  # (N, H, W, K)
    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    # alpha = torch.prod((1.0 - prob_map), dim=-1)  # (N, H, W)
    prob_map = prob_map.unsqueeze(-1) # (N,H,W,K,1)
    slant_range = (torch.arange(0.5, nbr_time_bins+0.5, dtype=torch.float32, device=colors.device)/nbr_time_bins*max_slant_range).expand(1,1,1,1,-1) # # (1,1,1,1,nbr_time_bins)
    z_diff = (fragments.zbuf.unsqueeze(-1) - slant_range ) /max_slant_range * mask.unsqueeze(-1) # # (N, H, W, K, nbr_time_bins)
    kernel = torch.exp(-z_diff**2 / blend_params.gamma)   # (N, H, W, K, nbr_time_bins)
    weights_num = prob_map *kernel * (kernel>eps_**2) # (N, H, W, K, nbr_time_bins)
    denom = weights_num.sum(dim=-2) # no background color # (N, H, W, nbr_time_bins)
    denom_mask = denom>eps_
    weighted_colors = (weights_num * colors.unsqueeze(-1)).sum(dim=-2) # (N,H,W,nbr_time_bins)
    pixel_colors = torch.zeros((N, H, W, nbr_time_bins), dtype=colors.dtype, device=colors.device)
    pixel_colors[denom_mask] = (weighted_colors[denom_mask]) / denom[denom_mask]
    # pixel_colors = nan_to_num(pixel_colors) # (N, H, W, nbr_time_bins)
    pixel_colors = pixel_colors * beam_pattern # (N, H, W, nbr_time_bins)
    # apply kernel w.r.t angles to get rid of the dip in front of the dome
    values, _ = torch.topk(pixel_colors,k=topk_angles,dim=1) # (N,topk_angles,W,nbr_time_bins)
    if not valid_mean:
        sss_rendered=torch.mean(values, axis=(1,2))# mean TODO let's see what happens
    else:
        sss_rendered=torch.sum(values, axis=(1,2))# valid mean TODO let's see what happens
        valid_mask_sum = (values>0).sum(dim=(1,2))# (N,nbr_time_bins)
        sss_rendered[valid_mask_sum!=0] = sss_rendered[valid_mask_sum!=0]/valid_mask_sum[valid_mask_sum!=0]
    # sss_rendered=torch.mean(pixel_colors**2, axis=(1,2))# mean TODO let's see what happens
    
    return sss_rendered

