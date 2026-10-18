# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check GPU imports, optionally running small operators on the current GPU."""

import sys
from argparse import ArgumentParser
from importlib import import_module


def check_operators() -> None:
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from transformer_engine.pytorch import Linear

    is_hopper = torch.cuda.get_device_capability() == (9, 0)
    # Check both ordinary heads and the large-head FA2 kernels used by MLA/VLMs.
    for head_dim in (64, 256):
        qkv = torch.randn(3, 2, 16, 4, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_qkv = qkv.detach().float().requires_grad_()
        reference = F.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in reference_qkv.unbind()), is_causal=True
        ).transpose(1, 2)
        output = flash_attn_func(*qkv.unbind(), causal=True)
        torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.02)
        output.float().square().sum().backward()
        reference.square().sum().backward()
        torch.testing.assert_close(qkv.grad.float(), reference_qkv.grad, rtol=0.08, atol=0.08)

        if is_hopper:
            fa3_output = import_module("flash_attn_interface").flash_attn_func(*qkv.unbind(), causal=True)
            torch.testing.assert_close(fa3_output.float(), reference.detach(), rtol=0.03, atol=0.02)
            fa3_output.float().square().sum().backward()
            assert torch.isfinite(qkv.grad).all()

    linear = Linear(64, 64, params_dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = linear(inputs)
    reference = F.linear(inputs.float(), linear.weight.float(), linear.bias.float())
    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.02)
    output.float().square().sum().backward()
    assert torch.isfinite(inputs.grad).all() and torch.isfinite(linear.weight.grad).all()

    quantize = import_module("fake_int4_quant_cuda").fake_int4_quant_cuda
    values = (torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 32) / 16 - 4).bfloat16()
    blocks = values.float()
    for symmetric in (True, False):
        if symmetric:
            scales = (blocks.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-5)
            zero = 0
        else:
            minimum = blocks.amin(dim=1, keepdim=True)
            scales = ((blocks.amax(dim=1, keepdim=True) - minimum) / 15).clamp_min(1e-5)
            zero = (-minimum / scales).round().clamp(0, 15)
        expected = ((blocks / scales).round() + zero).bfloat16()
        output, actual_scales, actual_zero = quantize(values, [1, 32], symmetric)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, scales.bfloat16(), rtol=0, atol=0)
        if not symmetric:
            torch.testing.assert_close(actual_zero, zero.bfloat16(), rtol=0, atol=0)

    weights = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    normalized = import_module("sgl_kernel").rmsnorm(inputs.detach(), weights, 1e-6)
    reference = inputs.detach().float() * torch.rsqrt(inputs.detach().float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized.float(), reference, rtol=0.03, atol=0.02)
    torch.cuda.synchronize()
    sys.stdout.write("GPU operators OK: FA2 forward/backward, TE Linear, INT4, SGLang RMSNorm\n")
    if is_hopper:
        sys.stdout.write("Hopper FA3 forward/backward OK\n")


def main() -> None:
    import torch

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--operators", action="store_true", help="also check small forward/backward CUDA operators")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU/driver unavailable"
    import_module("transformer_engine.pytorch")
    import_module("sgl_kernel")
    sys.stdout.write(f"GPU imports OK: {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}\n")
    if args.operators:
        check_operators()


if __name__ == "__main__":
    main()
