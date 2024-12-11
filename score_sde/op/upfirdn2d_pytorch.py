import torch

def upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """
    PyTorch implementation of upfirdn2d.

    Args:
        input: Input tensor of shape (B, C, H, W).
        kernel: Kernel tensor of shape (kernel_h, kernel_w).
        up_x: Upsampling factor in x-direction.
        up_y: Upsampling factor in y-direction.
        down_x: Downsampling factor in x-direction.
        down_y: Downsampling factor in y-direction.
        pad_x0: Padding on the left in x-direction.
        pad_x1: Padding on the right in x-direction.
        pad_y0: Padding on the top in y-direction.
        pad_y1: Padding on the bottom in y-direction.

    Returns:
        Output tensor.
    """

    in_h, in_w = input.shape[-2:]
    kernel_h, kernel_w = kernel.shape

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    out = torch.zeros(input.shape[:-2] + (out_h, out_w), dtype=input.dtype, device=input.device)

    # Upsample and pad the input
    upsampled_input = torch.nn.functional.interpolate(input, scale_factor=(up_y, up_x), mode='nearest')
    padded_input = torch.nn.functional.pad(upsampled_input, (pad_x0, pad_x1, pad_y0, pad_y1))

    # Perform convolution
    for y in range(out_h):
        for x in range(out_w):
            # Extract the region of input corresponding to the current output pixel
            in_y_start = y * down_y
            in_x_start = x * down_x
            in_patch = padded_input[..., in_y_start:in_y_start + kernel_h, in_x_start:in_x_start + kernel_w]

            # Perform convolution and accumulate the result
            out[..., y, x] = (in_patch * kernel[None, None, ...]).sum(dim=(-1, -2))

    return out