# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import torch.cuda.nvtx as nvtx

try:
  from grouped_gemm import permute, unpermute, groupedgemm
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  from moe.ops import permute, unpermute, groupedgemm, permute_topK

def permute(tokens, indices, expand_factor: int = 1):
    """Permute the tokens based on the indices.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token2expert indices tensor.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    expand_factor = indices.size(1)

    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, merge_factor: int = 1):
    """Unpermute the sorted tokens based on the indices.
    
    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        probs (torch.Tensor, optional): The probabilities tensor. Defaults to None.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    merge_factor = probs.size(1)

    if merge_factor > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // merge_factor
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // merge_factor
        assert (
            probs.size(1) == merge_factor
        ), f"probs size {probs.size()} merge_factor {merge_factor}"

    # unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, merge_factor, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens

def permute_topK_test(
    dtype,
    num_token,
    num_expert,
    hidden_size,
    num_topK):
    
    print(f"{dtype} token:{num_token} hidden_size:{hidden_size} expert:{num_expert} topK:{num_topK}")

    input = torch.empty((num_token, hidden_size), dtype=dtype)
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     input[i][j] = i * 100 + j
    
    indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])
    # print(input)
    # print(indices)

    # probs = torch.tensor([[0.1, 0.9],
    #                       [0.2, 0.8],
    #                       [0.3, 0.7]])
    # 0.5
    # probs = torch.ones_like(indices) / 2
    # rand
    probs = torch.rand(num_token, num_topK)
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    # print(probs)

    input = input.cuda()
    input_fp32 = input.detach().to(torch.float32)
    input_ = input.detach()
    indices = indices.to(torch.int32).cuda()
    probs = probs.cuda()
    input.requires_grad_(True)
    input_fp32.requires_grad_(True)
    input_.requires_grad_(True)
    probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch
    #
    ###################################################################################################################################
    permuted_tokens_fp32, sorted_indices = permute(input_fp32, indices, 2)
    permuted_tokens, sorted_indices = permute(input, indices, 2)
    # print("-----------------------------------------------------------------")
    # print(permuted_tokens)
    # print(sorted_indices)

    backward_input = torch.rand_like(permuted_tokens)
    backward_input_fp32 = backward_input.detach().to(torch.float32)
    # for i in range(num_token * num_topK):
    #   for j in range(hidden_size):
    #     backward_input[i][j] = i * 100 + j
    # print(backward_input)

    permuted_tokens_fp32.backward(backward_input_fp32, retain_graph=True)
    permuted_tokens.backward(backward_input, retain_graph=True)

    # unpermuted_tokens = unpermute(permuted_tokens, sorted_indices, probs=probs, merge_factor=2)
    # print(unpermuted_tokens)
    # print("-----------------------------------------------------------------")

    ###################################################################################################################################
    #
    # Mine
    #
    ###################################################################################################################################
    permuted_act, row_id_map = permute_topK(input_, indices)
    assert torch.allclose(permuted_tokens_fp32, permuted_act.to(torch.float32))

    permuted_act.backward(backward_input)

    if torch.allclose(input.grad, input_.grad)==False:
      original_inputs = input_.grad.float().cpu().numpy().flatten()
      original_output = input_fp32.grad.float().cpu().numpy().flatten()
      max_abs_error = abs(original_inputs - original_output).max()
      print(f"bwd max error: \t{max_abs_error:.3e} ({dtype})")

      original_inputs = input.grad.float().cpu().numpy().flatten()
      original_output = input_fp32.grad.float().cpu().numpy().flatten()
      max_abs_error = abs(original_inputs - original_output).max()
      print(f"bwd max error: \t{max_abs_error:.3e} ({dtype})")
    #   print(input.grad)
    #   print(input_.grad)
    #   print(input_fp32.grad)

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    t1 = triton.testing.do_bench(lambda: permute(input, indices, 2))
    print(f"pytorch fwd: {t1:.3f}")
    t2 = triton.testing.do_bench(lambda: permuted_tokens.backward(backward_input, retain_graph=True))
    print(f"pytorch bwd: {t2:.3f}")

    t3 = triton.testing.do_bench(lambda: permute_topK(input_, indices))
    print(f"mine    fwd: {t3:.3f}")
    t4 = triton.testing.do_bench(lambda: permuted_act.backward(backward_input))
    print(f"mine    bwd: {t4:.3f}")

if __name__ == "__main__":

    # num_token = 4
    # num_expert = 3
    # hidden_size = 8
    # num_topK = 2

    num_token = 4096
    num_expert = 4
    hidden_size = 4096
    num_topK = 2

    dtype = torch.float32
    permute_topK_test(dtype, num_token, num_expert, hidden_size, num_topK)
    dtype = torch.float16
    permute_topK_test(dtype, num_token, num_expert, hidden_size, num_topK)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, num_expert, hidden_size, num_topK)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5)