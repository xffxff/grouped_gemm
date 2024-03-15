# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import os
from sys import stderr

so_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/build'
torch.classes.load_library(so_dir + '/libmoe_unit_ops.so')

# TODO by Jiang Shao, add parameter `out` which can be optionally given to be used as output buffers.

################################################################################################
##
## PermuteMoE
##
################################################################################################

class PermuteMoE(torch.autograd.Function):
  
  workspace_fw=None
  dtype=None
  max_token_num=0

  @staticmethod
  def forward(ctx, 
              unpermuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              max_token_num: int):

    # Device check
    if unpermuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"unpermuted_inputs\" of permute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
      print("[Warning] The input \"expert_for_rows\" of permute op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if unpermuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"[Error] permute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {unpermuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"expert_for_rows\" of permute op is {expert_for_rows.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      expert_for_rows = expert_for_rows.to(torch.int32)

    # Contiguous check
    if not unpermuted_inputs.is_contiguous():
      print("[Warning] The input \"unpermuted_inputs\" of permute op is discontiguous!", file=stderr)
      unpermuted_inputs = unpermuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of permute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()

    input_max_token_num = max(max_token_num, unpermuted_inputs.size(0))
    if PermuteMoE.max_token_num < input_max_token_num:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = input_max_token_num
      PermuteMoE.workspace_fw = []

    if PermuteMoE.dtype != unpermuted_inputs.dtype:
      # print("Permute op workspace reset!")
      PermuteMoE.dtype = unpermuted_inputs.dtype
      PermuteMoE.workspace_fw = []

    permuted_inputs, row_id_map, PermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw,
      PermuteMoE.max_token_num)

    ctx.row_id_map = row_id_map

    return permuted_inputs, row_id_map

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()
    row_id_map = ctx.row_id_map

    original_output = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs_grad,
      row_id_map)

    return original_output, None, None

################################################################################################
##
## UnpermuteMoE
##
################################################################################################

class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_token_num=0
  
  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              row_id_map: torch.Tensor,
              max_token_num: int):

    # Device check
    if permuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"permuted_inputs\" of unpermute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
      print("[Warning] The input \"expert_for_rows\" of unpermute op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()
    if row_id_map.is_cpu:
      print("[Warning] The input \"row_id_map\" of unpermute op is on the device: CPU!", file=stderr)
      row_id_map = row_id_map.cuda()

    # Shape check
    if permuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"[Error] unpermute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {permuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"expert_for_rows\" of unpermute op is {expert_for_rows.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      expert_for_rows = expert_for_rows.to(torch.int32)
    if row_id_map.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"row_id_map\" of unpermute op is {row_id_map.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      row_id_map = row_id_map.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
      print("[Warning] The input \"permuted_inputs\" of unpermute op is discontiguous!", file=stderr)
      permuted_inputs = permuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of unpermute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()
    if not row_id_map.is_contiguous():
      print("[Warning] The input \"row_id_map\" of unpermute op is discontiguous!", file=stderr)
      row_id_map = row_id_map.contiguous()

    input_max_token_num = max(max_token_num, permuted_inputs.size(0))
    if UnpermuteMoE.max_token_num < input_max_token_num:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = input_max_token_num
      UnpermuteMoE.workspace_fw = []

    if UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []

    ctx.expert_for_rows = expert_for_rows

    original_output = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs,
      row_id_map)
    
    return original_output

  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):
    if not unpermuted_inputs_grad.is_contiguous():
      unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw,
      UnpermuteMoE.max_token_num)

    return permuted_inputs, None, None, None

################################################################################################
##
## PermuteMoE topK
##
################################################################################################

class PermuteMoE_topK(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_expanded_token_num=0

  @staticmethod
  def forward(ctx, 
              input_act: torch.Tensor,
              indices: torch.Tensor,
              max_token_num: int):

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input \"input_act\" of permute_topK op is on the device: CPU!")
    if indices.is_cpu:
      print("[Warning] The input \"indices\" of permute_topK op is on the device: CPU!", file=stderr)
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if input_act.size(0) != indices.size(0):
      raise RuntimeError(f"[Error] permute_topK op input \"indices\" shape mismatch! "
                         f"Expect {input_act.size(0)}, but got {indices.size(0)}.")

    # Data type check
    if indices.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"indices\" of permute_topK op is {indices.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      indices = indices.to(torch.int32)

    # Contiguous check
    if not input_act.is_contiguous():
      print("[Warning] The input \"input_act\" of permute_topK op is discontiguous!", file=stderr)
      input_act = input_act.contiguous()
    if not indices.is_contiguous():
      print("[Warning] The input \"indices\" of permute_topK op is discontiguous!", file=stderr)
      indices = indices.contiguous()

    num_topK = indices.size(1)

    input_max_expanded_token_num = max(max_token_num, input_act.size(0)) * num_topK
    if PermuteMoE_topK.max_expanded_token_num < input_max_expanded_token_num:
      PermuteMoE_topK.max_expanded_token_num = input_max_expanded_token_num
      PermuteMoE_topK.workspace_fw = []

    if PermuteMoE_topK.dtype != input_act.dtype:
      PermuteMoE_topK.dtype = input_act.dtype
      PermuteMoE_topK.workspace_fw = []

    permuted_act, row_id_map, PermuteMoE_topK.workspace_fw = torch.ops.moe_unit_ops.moe_permute_topK_op(
      input_act,
      indices,
      PermuteMoE_topK.workspace_fw,
      PermuteMoE_topK.max_expanded_token_num)

    ctx.row_id_map = row_id_map
    ctx.num_tokens = indices.size(0)
    ctx.num_topK = indices.size(1)

    return permuted_act, row_id_map


  @staticmethod
  def backward(ctx, permuted_act_grad, _):
    if not permuted_act_grad.is_contiguous():
      permuted_act_grad = permuted_act_grad.contiguous()

    row_id_map = ctx.row_id_map
    num_tokens = ctx.num_tokens
    num_topK = ctx.num_topK

    unpermuted_act_grad = torch.ops.moe_unit_ops.moe_recover_topK_op(
      permuted_act_grad,
      row_id_map,
      torch.tensor([]),
      num_tokens,
      num_topK)

    return unpermuted_act_grad, None, None

################################################################################################
##
## UnpermuteMoE topK
##
################################################################################################

class UnpermuteMoE_topK(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              input_act: torch.Tensor,
              row_id_map: torch.Tensor,
              probs: torch.Tensor):

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input \"input_act\" of unpermute_topK op is on the device: CPU!")
    if row_id_map.is_cpu:
      print("[Warning] The input \"row_id_map\" of unpermute_topK op is on the device: CPU!", file=stderr)
      row_id_map = row_id_map.cuda()
    if probs.is_cpu:
      print("[Warning] The input \"probs\" of unpermute_topK op is on the device: CPU!", file=stderr)
      probs = probs.cuda()

    # Shape check
    if row_id_map.size(0) != input_act.size(0):
      raise RuntimeError(f"[Error] unpermute_topK op input \"row_id_map\" shape mismatch! "
                         f"Expect {input_act.size(0)}, but got {row_id_map.size(0)}.")
    if input_act.size(0) != probs.size(0) * probs.size(1):
      raise RuntimeError(f"[Error] unpermute_topK op input \"probs\" shape mismatch! "
                         f"Expect {input_act.size(0)}, but got {probs.size(0) * probs.size(1)}.")

    # Data type check
    if row_id_map.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"row_id_map\" of unpermute_topK op is {row_id_map.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      row_id_map = row_id_map.to(torch.int32)
    if probs.dtype != torch.float32:
      print(f"[Warning] The data type of the input \"probs\" of unpermute_topK op is {probs.dtype}! "
            "The recommended type is torch.float32.", file=stderr)
      probs = probs.to(torch.float32)

    # Contiguous check
    if not input_act.is_contiguous():
      print("[Warning] The input \"input_act\" of unpermute_topK op is discontiguous!", file=stderr)
      input_act = input_act.contiguous()
    if not row_id_map.is_contiguous():
      print("[Warning] The input \"row_id_map\" of unpermute_topK op is discontiguous!", file=stderr)
      row_id_map = row_id_map.contiguous()
    if not probs.is_contiguous():
      print("[Warning] The input \"probs\" of unpermute_topK op is discontiguous!", file=stderr)
      probs = probs.contiguous()

    num_tokens = probs.size(0)
    num_topK = probs.size(1)

    unpermuted_output = torch.ops.moe_unit_ops.moe_recover_topK_op(
      input_act,
      row_id_map,
      probs,
      num_tokens,
      num_topK)

    ctx.save_for_backward(input_act, row_id_map, probs)

    return unpermuted_output

  @staticmethod
  def backward(ctx, unpermuted_act_grad):

    if not unpermuted_act_grad.is_contiguous():
      unpermuted_act_grad = unpermuted_act_grad.contiguous()

    input_act, row_id_map, probs = ctx.saved_tensors

    act_grad = None
    if ctx.needs_input_grad[0]:
      act_grad, prob_grad = torch.ops.moe_unit_ops.moe_recover_topK_bwd_op(
        unpermuted_act_grad,
        input_act,
        row_id_map,
        probs)
    
    if not ctx.needs_input_grad[2]:
      prob_grad = None

    return act_grad, None, prob_grad


################################################################################################
##
## GroupedGemmMoE
##
################################################################################################

class GroupedGemmMoE(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              tokens_per_expert: torch.Tensor,
              transB: bool,
              *weights_list):

    # Weight matrices num check
    if len(weights_list) != tokens_per_expert.size(0):
      raise RuntimeError(f"[Error] groupedgemm op input \"weights_list\" matrices num mismatch! "
                         f"Expect ({tokens_per_expert.size(0)}), but got ({len(weights_list)}).")

    # Device check
    if permuted_inputs.is_cpu:
      raise RuntimeError("[Error] The input \"permuted_inputs\" of groupedgemm op is on the device: CPU!")
    if weights_list[0].is_cpu:
      raise RuntimeError("[Error] The input \"weights_list\" of groupedgemm op is on the device: CPU!")
    if tokens_per_expert.is_cpu:
      print("[Warning] The input \"tokens_per_expert\" of groupedgemm op is on the device: CPU!", file=stderr)
      tokens_per_expert = tokens_per_expert.cuda()

    # Shape check
    if not transB:
      if permuted_inputs.size(1) != weights_list[0].size(0):
        raise RuntimeError(f"[Error] groupedgemm op input \"weights_list\" shape mismatch! "
                           f"Expect ({permuted_inputs.size(1)}), but got ({weights_list[0].size(0)}).")
    else:
      if permuted_inputs.size(1) != weights_list[0].size(1):
        raise RuntimeError(f"[Error] groupedgemm op input \"weights_list\" shape mismatch! "
                           f"Expect ({permuted_inputs.size(1)}), but got ({weights_list[0].size(1)}).")

    # Data type check
    if permuted_inputs.dtype != weights_list[0].dtype:
      raise RuntimeError(f"[Error] groupedgemm op input data type mismatch! "
                         f"\"permuted_inputs\": {permuted_inputs.dtype}, \"weights_list\": {weights_list[0].dtype}.")
    if tokens_per_expert.dtype != torch.int32:
      print(f"[Warning] The data type of the input \"tokens_per_expert\" of groupedgemm op is {tokens_per_expert.dtype}! "
            "The recommended type is torch.int32.", file=stderr)
      tokens_per_expert = tokens_per_expert.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
      print("[Warning] The input \"permuted_inputs\" of groupedgemm op is discontiguous!", file=stderr)
      permuted_inputs = permuted_inputs.contiguous()
    if not weights_list[0].is_contiguous():
      print("[Warning] The input \"weights_list\" of groupedgemm op is discontiguous!", file=stderr)
      for w in weights_list:
        w = w.contiguous()

    output = torch.ops.moe_unit_ops.moe_group_gemm_op(
      permuted_inputs,
      weights_list,
      tokens_per_expert,
      transB)
    
    ctx.save_for_backward(permuted_inputs, tokens_per_expert)
    ctx.transB = transB
    ctx.weights_list = weights_list

    return output


  @staticmethod
  def backward(ctx, permuted_inputs_grad):
    permuted_inputs, tokens_per_expert = ctx.saved_tensors
    transB = ctx.transB
    weights_list = ctx.weights_list

    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()

    activation_grad = None
    if ctx.needs_input_grad[0]:
      activation_grad = torch.ops.moe_unit_ops.moe_group_gemm_op(
        permuted_inputs_grad,
        weights_list,
        tokens_per_expert,
        not transB)
      
    weight_grad = None
    if ctx.needs_input_grad[3]:
      weight_grad = torch.ops.moe_unit_ops.moe_group_gemm_backward_op(
        permuted_inputs,
        permuted_inputs_grad,
        tokens_per_expert,
        transB)

    weight_grad_list = []
    for i in range(weight_grad.shape[0]):
      weight_grad_list.append(weight_grad[i])

    return activation_grad, None, None, *weight_grad_list

################################################################################################
##
## Ops Wrapper
##
################################################################################################

def permute(unpermuted_inputs, expert_for_rows, max_token_num=0):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows, max_token_num)

def permute_topK(input_act, indices, max_token_num=0):
  return PermuteMoE_topK.apply(input_act, indices, max_token_num)

def unpermute(permuted_inputs, expert_for_rows, row_id_map, max_token_num=0):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, row_id_map, max_token_num)

def unpermute_topK(input_act, row_id_map, probs):
  return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)

def groupedgemm(permuted_inputs, tokens_per_expert, transB=False, *weights_list):
  return GroupedGemmMoE.apply(permuted_inputs, tokens_per_expert, transB, *weights_list)

def sinkhorn_kernel(cost, tol=0.0001):
    return torch.ops.moe_unit_ops.sinkhorn(cost, tol)
