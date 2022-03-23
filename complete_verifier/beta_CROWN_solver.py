#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import copy
import time
import random
from collections import defaultdict, OrderedDict

import torch
import arguments

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import reduction_sum, stop_criterion_sum, stop_criterion_min
from auto_LiRPA.operators.activation import FIXED_SPLIT, SIGN
from lp_mip_solver import *

total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_transfer_time = total_finalize_time = 0.0


class LiRPAConvNet:
    def __init__(self, model_ori, pred, test, device='cuda', simplify=False, in_size=(1, 3, 32, 32),
                 conv_mode='patches', deterministic=False, c=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.simplify = False
        self.c = c
        self.pred = pred
        self.layers = layers
        self.input_shape = in_size
        self.net = BoundedModule(net, torch.zeros(in_size, device=device), bound_opts={'relu': 'adaptive', 'deterministic': deterministic, 'conv_mode': conv_mode},
                                 device=device)
        self.net.eval()
        self.needed_A_dict = None
        self.pool = None   # For multi-process.
        self.pool_result = None
        self.pool_termination_flag = None

    
    def get_lower_bound(self, pre_lbs, pre_ubs, split, slopes=None, betas=None, history=None, layer_set_bound=True, 
                        split_history=None, single_node_split=True, intermediate_betas=None):

        """
        # (in) pre_lbs: layers list -> tensor(batch, layer shape)
        # (in) relu_mask: relu layers list -> tensor(batch, relu layer shape (view-1))
        # (in) slope: relu layers list -> tensor(batch, relu layer shape)
        # (out) lower_bounds: batch list -> layers list -> tensor(layer shape)
        # (out) masks_ret: batch list -> relu layers list -> tensor(relu layer shape)
        # (out) slope: batch list -> relu layers list -> tensor(relu layer shape)
        """
        print("--------------------------")
        print("Computing lower bound for")
        print("Split:{}".format(split))


        if history is None:
            history = []
        start = time.time()

        lp_test = arguments.Config["debug"]["lp_test"]

        if single_node_split:
            print("Using single node split")
            ret = self.update_bounds_parallel(pre_lbs, pre_ubs, split, slopes, betas=betas, early_stop=False, history=history,
                                              layer_set_bound=layer_set_bound)
        else:
            ret = self.update_bounds_parallel_general(pre_lbs, pre_ubs, split, slopes, early_stop=False,
                                            history=history, split_history=split_history, 
                                            intermediate_betas=intermediate_betas, layer_set_bound=layer_set_bound)

        # if get_upper_bound and single_node_split, primals have p and z values; otherwise None
        lower_bounds, upper_bounds, lAs, slopes, betas, split_history, best_intermediate_betas, primals = ret

        beta_crown_lbs = [i[-1].item() for i in lower_bounds]
        beta_time = time.time()-start

        if lp_test == "LP_intermediate_refine":
            refine_time = time.time()
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                init_lp_glb0, refined_lp_glb0 = self.update_the_model_lp(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                init_lp_glb1, refined_lp_glb1 = self.update_the_model_lp(lower_bounds[bdi + total_batch],
                                         upper_bounds[bdi + total_batch], bd[0], choice=0)
                print("############ bound tightness summary ##############")
                print(f"init opt crown: {pre_lbs[-1][-1].item()}")
                print("beta crown for split:", beta_crown_lbs)
                print(f"init lp for split: [{init_lp_glb0}, {init_lp_glb1}]")
                print(f"lp intermediate refined for split: [{refined_lp_glb0}, {refined_lp_glb1}]")
                print("lp_refine time:", time.time() - refine_time, "beta crown time:", beta_time)
                exit()

        elif lp_test == "MIP_intermediate_refine":
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                self.update_the_model_mip(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                self.update_the_model_mip(lower_bounds[bdi + total_batch],
                                          upper_bounds[bdi + total_batch], bd[0], choice=0)

        end = time.time()
        print('batch bounding time: ', end - start)
        print("Lower bound of this batch:")
        print(lower_bounds)
        return [i[-1].item() for i in upper_bounds], [i[-1].item() for i in lower_bounds], None, lAs, lower_bounds, \
               upper_bounds, slopes, split_history, betas, best_intermediate_betas, primals


    def get_relu(self, model, idx):
        # find the i-th ReLU layer
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer


    """Trasfer all necessary tensors to CPU in a batch."""
    def transfer_to_cpu(self, net, non_blocking=True, opt_intermediate_beta=False):
        # Create a data structure holding all the tensors we need to transfer.
        cpu_net = lambda : None
        cpu_net.relus = [None] * len (net.relus)
        for i in range(len(cpu_net.relus)):
            cpu_net.relus[i] = lambda : None
            cpu_net.relus[i].inputs = [lambda : None]
            cpu_net.relus[i].name = net.relus[i].name

        # Transfer data structures for each relu.
        # For get_candidate_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # For get_candidate_parallel.
            cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
            cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)
        # For get_lA_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
        # For get_slope().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # Per-neuron alpha.
            cpu_layer.alpha = OrderedDict()
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha[spec_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)
        # For get_beta().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            if layer.sparse_beta is not None:
                cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)
        # For intermediate beta.
        if opt_intermediate_beta and net.best_intermediate_betas is not None:
            cpu_net.best_intermediate_betas = OrderedDict()
            for split_layer, all_int_betas_this_layer in net.best_intermediate_betas.items():
                # Single neuron split so far.
                assert 'single' in all_int_betas_this_layer
                assert 'history' not in all_int_betas_this_layer
                assert 'split' not in all_int_betas_this_layer
                cpu_net.best_intermediate_betas[split_layer] = {'single': defaultdict(dict)}
                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['lb'] = this_layer_intermediate_betas['lb'].to(device='cpu', non_blocking=non_blocking)
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['ub'] = this_layer_intermediate_betas['ub'].to(device='cpu', non_blocking=non_blocking)

        return cpu_net


    def get_primal_upper_bound(self, A):
        with torch.no_grad():
            assert self.x.ptb.norm == np.inf, print('we only support to get primals for Linf norm perturbation so far')
            input_A_lower = A[self.net.output_name[0]][self.net.input_name[0]]["lA"]
            batch = input_A_lower.shape[0]

            x_lb, x_ub, eps = self.x.ptb.x_L, self.x.ptb.x_U, self.x.ptb.eps
            x_lb = x_lb.repeat(batch, 1, 1, 1)
            x_ub = x_ub.repeat(batch, 1, 1, 1)
            input_primal = x_lb.clone().detach()
            input_primal[input_A_lower.squeeze(1) < 0] = x_ub[input_A_lower.squeeze(1) < 0]

        return input_primal, self.net(input_primal, clear_forward_only=True).matmul(self.c[0].transpose(-1, -2))


    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        lower_bounds = []
        upper_bounds = []
        self.pre_relu_indices = []
        i = 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {}

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())
            self.name_dict[i] = layer.inputs[0].name
            self.pre_relu_indices.append(i)
            i += 1

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(1, -1).detach())
        upper_bounds.append(ub.view(1, -1).detach())

        print(self.name_dict)
        return lower_bounds, upper_bounds, self.pre_relu_indices


    def get_candidate_parallel(self, model, lb, ub, batch, diving_batch=0):
        # get the intermediate bounds in the current model
        lower_bounds = []
        upper_bounds = []

        for layer in model.relus:
            print("In get_candidate_parallel")
            print("layer.inputs", layer.inputs)
            lower_bounds.append(layer.inputs[0].lower)
            upper_bounds.append(layer.inputs[0].upper)

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch + diving_batch, -1).detach())
        upper_bounds.append(ub.view(batch + diving_batch, -1).detach())

        print("lower_bounds:\n", lower_bounds)


        return lower_bounds, upper_bounds


    def get_mask_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.relus:
            # 1 is unstable neuron, 0 is stable neuron.
            mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float()
            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            if this_relu.lA is not None:
                lA.append(this_relu.lA.squeeze(0))
            else:
                # It might be skipped due to inactive neurons.
                lA.append(None)

        ret_mask, ret_lA = [], []
        for i in range(mask[0].size(0)):
            ret_mask.append([j[i:i+1] for j in mask])
            ret_lA.append([j[i:i+1] if j is not None else None for j in lA])
        return ret_mask, ret_lA


    def get_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None]
        # get lower A matrix of ReLU
        lA = []
        for this_relu in model.relus:
            lA.append(this_relu.lA.squeeze(0))

        ret_lA = []
        for i in range(lA[0].size(0)):
            ret_lA.append([j[i:i+1] for j in lA])
        return ret_lA


    def get_beta(self, model, splits_per_example, diving_batch=0):
        # split_per_example only has half of the examples.



        batch = splits_per_example.size(0) - diving_batch
        print(">>>>>>batch", batch)
        if FIXED_SPLIT:
            retb = [[] for i in range(batch + diving_batch)]
            for mi, m in enumerate(model.relus):
                for i in range(batch):
                    # Save only used beta, discard padding beta.
                    retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                for i in range(diving_batch):
                    retb[batch + i].append(m.sparse_beta[batch + i, :splits_per_example[batch + i, mi]])

        else:
            retb = [[] for i in range(batch * 2 + diving_batch)]
            for mi, m in enumerate(model.relus):
                for i in range(batch):
                    # Save only used beta, discard padding beta.
                    retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                    retb[i + batch].append(m.sparse_beta[i + batch, :splits_per_example[i, mi]])
                for i in range(diving_batch):
                    retb[2 * batch + i].append(m.sparse_beta[2 * batch + i, :splits_per_example[batch + i, mi]])
        return retb


    def get_slope(self, model):
        if len(model.relus) == 0:
            return [None]

        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.
        batch_size = next(iter(model.relus[0].alpha.values())).size(2)
        ret = [defaultdict(dict) for i in range(batch_size)]
        for m in model.relus:
            for spec_name, alpha in m.alpha.items():
                # print(f'save layer {m.name} start_node {spec_name} shape {alpha.size()} norm {alpha.abs().sum()}')
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:]
        return ret


    def set_slope(self, model, slope, intermediate_refinement_layers=None, diving_batch=0, should_duplicate = True):
        cleanup_intermediate_slope = isinstance(intermediate_refinement_layers, list) and len(intermediate_refinement_layers) == 0
        if cleanup_intermediate_slope:
            # Clean all intermediate betas if we are not going to refine intermeidate layer neurons anymore.
            del model.best_intermediate_betas
            for m in model.relus:
                if hasattr(m, 'single_intermediate_betas'):
                    print(f'deleting single_intermediate_betas for {m.name}')
                    del m.single_intermediate_betas
                if hasattr(m, 'history_intermediate_betas'):
                    print(f'deleting history_intermediate_betas for {m.name}')
                    del m.history_intermediate_betas
                if hasattr(m, 'split_intermediate_betas'):
                    print(f'deleting split_intermediate_betas for {m.name}')
                    del m.split_intermediate_betas

        if type(slope) == list:
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[0][m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
                            if len(slope) - diving_batch > 0:
                                # Merge all slope vectors together in this batch. Size is (2, spec, batch, *shape).
                                m.alpha[spec_name] = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch)], dim=2)
                                # Duplicate for the second half of the batch.
                                """Nham: looks like A is duplicate here"""
                                if should_duplicate:
                                    m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3))).detach().requires_grad_()
                            if diving_batch > 0:
                                # create diving alpha
                                diving_alpha = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch, len(slope))], dim=2)
                                if diving_batch == len(slope):
                                    m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                                else:
                                    m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                                del diving_alpha
                            # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        elif type(slope) == defaultdict:
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
                            # create diving alpha
                            diving_alpha = slope[m.name][spec_name]
                            assert diving_batch == diving_alpha.shape[2]
                            m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                            # else:
                            #     m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                            del diving_alpha
                        # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        else:
            raise NotImplementedError


    def reset_beta(self, model, batch, max_splits_per_layer=None, betas=None, diving_batch=0):
        # Recreate new beta with appropriate shape.
        for mi, m in enumerate(self.net.relus):
            # Create only the non-zero beta. For each layer, it is padded to maximal length.
            # We create tensors on CPU first, and they will be transferred to GPU after initialized.
            m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
            m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            # Load beta from history.
            # for bi in range(len(betas)):
            for bi in range(batch):
                if betas[bi] is not None:
                    # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                    valid_betas = len(betas[bi][mi])
                    m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
            # This is the beta variable to be optimized for this layer.
            if FIXED_SPLIT:
                m.sparse_beta = m.sparse_beta.detach().to(device=self.net.device, non_blocking=True).requires_grad_()
            else:
                m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()
            
            assert batch + diving_batch == len(betas)
            if diving_batch != 0:
                m.diving_sparse_beta = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                m.diving_sparse_beta_loc = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                m.diving_sparse_beta_sign = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                # Load diving beta from history.
                for dbi in range(diving_batch):
                    if betas[batch + dbi] is not None:
                        # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                        valid_betas = len(betas[batch + dbi][mi])
                        m.diving_sparse_beta[dbi, :valid_betas] = betas[batch + dbi][mi]
                m.diving_sparse_beta = m.diving_sparse_beta.to(device=self.net.device, non_blocking=True)
                m.sparse_beta = torch.cat([m.sparse_beta, m.diving_sparse_beta], dim=0).detach().\
                            to(device=self.net.device, non_blocking=True).requires_grad_()
                del m.diving_sparse_beta


    """Main function for computing bounds after branch and bound in Beta-CROWN."""
    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None, beta=None, betas=None,
                        early_stop=True, history=None, layer_set_bound=True, shortcut=False):
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time

        if beta is None:
            beta = arguments.Config["solver"]["beta-crown"]["beta"] # might need to set beta False in FSB node selection
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]["lr_alpha"]
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]

        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0
        # update optimize-CROWN bounds in a parallel way

        # if history is None:
        #     history = []
        
        diving_batch = 0
        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)
        print("batch", batch)
        print("decision", decision)
        print("beta", beta)
        # initial results with empty list

        if FIXED_SPLIT and beta:
            ret_l = [[] for _ in range(batch + diving_batch)]
            ret_u = [[] for _ in range(batch + diving_batch)]
            ret_s = [[] for _ in range(batch + diving_batch)]
            ret_b = [[] for _ in range(batch + diving_batch)]
            new_split_history = [{} for _ in range(batch + diving_batch)]
            best_intermediate_betas = [defaultdict(dict) for _ in range(batch + diving_batch)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.


        else:

            ret_l = [[] for _ in range(batch * 2 + diving_batch)]
            ret_u = [[] for _ in range(batch * 2 + diving_batch)]
            ret_s = [[] for _ in range(batch * 2 + diving_batch)]
            ret_b = [[] for _ in range(batch * 2 + diving_batch)]
            new_split_history = [{} for _ in range(batch * 2 + diving_batch)]
            best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2 + diving_batch)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        start_prepare_time = time.time()
        # iteratively change upper and lower bound from former to later layer

        if beta:
            # count how many split nodes in each batch example (batch, num of layers)
            splits_per_example = torch.zeros(size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu', requires_grad=False)
            for bi in range(batch):
                d = decision[bi][0]
                for mi, layer_splits in enumerate(history[bi]):
                    splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.
            # This is the maximum number of split in each relu neuron for each batch.
            if batch > 0: max_splits_per_layer = splits_per_example.max(dim=0)[0]

            if diving_batch != 0:
                diving_splits_per_example = torch.zeros(size=(diving_batch, len(self.net.relus)),
                            dtype=torch.int64, device='cpu', requires_grad=False)
                for dbi in range(diving_batch):
                    # diving batch does not have decision splits but only have history splits
                    for mi, diving_layer_splits in enumerate(history[dbi + batch]):
                        diving_splits_per_example[dbi, mi] = len(diving_layer_splits[0])  # First element of layer_splits is a list of split neuron IDs.

                # import pdb; pdb.set_trace()
                splits_per_example = torch.cat([splits_per_example, diving_splits_per_example], dim=0)
                max_splits_per_layer = splits_per_example.max(dim=0)[0]
                del diving_splits_per_example

            # Create and load warmup beta.
            self.reset_beta(self.net, batch, betas=betas, max_splits_per_layer=max_splits_per_layer, diving_batch=diving_batch)  # warm start beta

            for bi in range(batch):
                # Add history splits.
                d, idx = decision[bi][0], decision[bi][1]
                # Each history element has format [[[layer 1's split location], [layer 1's split coefficients +1/-1]], [[layer 2's split location], [layer 2's split coefficients +1/-1]], ...].
                for mi, (split_locs, split_coeffs) in enumerate(history[bi]):
                    split_len = len(split_locs)
                    self.net.relus[mi].sparse_beta_sign[bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                    self.net.relus[mi].sparse_beta_loc[bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                    # Add current decision for positive splits.
                    if mi == d:
                        self.net.relus[mi].sparse_beta_sign[bi, split_len] = 1.0
                        self.net.relus[mi].sparse_beta_loc[bi, split_len] = idx
            """
            Nham: here is where we enforce our mask
            Strategy1: if a ReLU should be fixed, keep track of its index. Only use half of the output
                       corresponding to the fixed half
            Strategy2: if a ReLU should be fixed, then it is no longer duplicate. 
                      Then we set the sign properly. We should find all instances when `torch.repeat` is used,
                      then change it.

            
            """
            if FIXED_SPLIT:
                """
                Nham: is a stable node, no duplication
                """
                for m in self.net.relus:
                    m.sparse_beta_loc = m.sparse_beta_loc.to(
                    device=self.net.device, non_blocking=True)
                    m.sparse_beta_sign = m.sparse_beta_sign.detach()
                # Fixup it values
                for bi in range(batch):
                    d = decision[bi][0]  # layer of this split.
                    split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
                    self.net.relus[d].sparse_beta_sign[bi, split_len] = SIGN
            else:
                # Duplicate split location.
                for m in self.net.relus:
                    m.sparse_beta_loc = m.sparse_beta_loc.repeat(2, 1).detach()
                    m.sparse_beta_loc = m.sparse_beta_loc.to(device=self.net.device, non_blocking=True)
                    m.sparse_beta_sign = m.sparse_beta_sign.repeat(2, 1).detach()
                # Fixup the second half of the split (negative splits).
                for bi in range(batch):
                    d = decision[bi][0]  # layer of this split.
                    split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
                    self.net.relus[d].sparse_beta_sign[bi + batch, split_len] = -1.0
            # Transfer tensors to GPU.
            for m in self.net.relus:
                print(m)
                print("S matrix location\n", m.sparse_beta_loc, m.sparse_beta_loc.shape)
                print("S matrix\n", m.sparse_beta_sign, m.sparse_beta_sign.shape)

                m.sparse_beta_sign = m.sparse_beta_sign.to(device=self.net.device, non_blocking=True)

            #Nham: DEADCODE
            if diving_batch > 0:
                # add diving domains history splits, no decision in diving domains
                for dbi in range(diving_batch):
                    for mi, (split_locs, split_coeffs) in enumerate(history[dbi + batch]):
                        split_len = len(split_locs)
                        self.net.relus[mi].diving_sparse_beta_sign[dbi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                        self.net.relus[mi].diving_sparse_beta_loc[dbi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                for m in self.net.relus:
                    # cat beta loc and sign to have the correct shape
                    m.diving_sparse_beta_loc = m.diving_sparse_beta_loc.to(device=self.net.device, non_blocking=True)
                    m.diving_sparse_beta_sign = m.diving_sparse_beta_sign.to(device=self.net.device, non_blocking=True)
                    m.sparse_beta_loc = torch.cat([m.sparse_beta_loc, m.diving_sparse_beta_loc], dim=0).detach()
                    m.sparse_beta_sign = torch.cat([m.sparse_beta_sign, m.diving_sparse_beta_sign], dim=0).detach()
                    # do no need to store the diving beta params any more
                    del m.diving_sparse_beta_loc, m.diving_sparse_beta_sign
        else:
            for m in self.net.relus:
                m.beta = None

        # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
        if FIXED_SPLIT and beta:
            with torch.no_grad():
                print("pre_lb_all", pre_lb_all)

                # Setting the neuron upper/lower bounds with a split to 0.
                zero_indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
                zero_indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
                for i in range(batch):
                    d, idx = decision[i][0], decision[i][1]
                    # We save the batch, and neuron number for each split, and will set all corresponding elements in batch.
                    zero_indices_batch[d].append(i)
                    zero_indices_neuron[d].append(idx)
                zero_indices_batch = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_batch]
                zero_indices_neuron = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_neuron]

                print(zero_indices_batch)
                print(zero_indices_neuron)

                # 2 * batch + diving_batch
                upper_bounds = [i[:batch] for i in pre_ub_all[:-1]]
                lower_bounds = [i[:batch] for i in pre_lb_all[:-1]]
                logger.debug("in update_bounds_parallel{}".format(lower_bounds))
                # Only the last element is used later.
                pre_lb_last = torch.cat([pre_lb_all[-1][:batch], pre_lb_all[-1][batch:]])
                pre_ub_last = torch.cat([pre_ub_all[-1][:batch], pre_ub_all[-1][batch:]])


                # breakpoint()

                new_candidate = {}
                for d in range(len(lower_bounds)):
                    # for each layer except the last output layer
                    if len(zero_indices_batch[d]):
                        # we set lower = 0 in first half batch, and upper = 0 in second half batch
                        #Nham: If we do not split RELU here, only lower_bound or upper_bound should be updated
                        if SIGN:
                            #Nham: Fix the ReLU to be positive, so only need to change lower bound to be 0
                            lower_bounds[d][:batch].view(batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                        else:
                            #Nham: Fix the ReLU to be negative, so only need to update upper bound to be 0
                            upper_bounds[d][:batch].view(batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                    new_candidate[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]

        else:            
            with torch.no_grad():
                # Setting the neuron upper/lower bounds with a split to 0.
                zero_indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
                zero_indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
                for i in range(batch):
                    d, idx = decision[i][0], decision[i][1]
                    # We save the batch, and neuron number for each split, and will set all corresponding elements in batch.
                    zero_indices_batch[d].append(i)
                    zero_indices_neuron[d].append(idx)
                zero_indices_batch = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_batch]
                zero_indices_neuron = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_neuron]

                # 2 * batch + diving_batch
                upper_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_ub_all[:-1]]
                lower_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_lb_all[:-1]]

                logger.debug("in update_bounds_parallel{}".format(lower_bounds))
                # Only the last element is used later.
                pre_lb_last = torch.cat([pre_lb_all[-1][:batch], pre_lb_all[-1][:batch], pre_lb_all[-1][batch:]])
                pre_ub_last = torch.cat([pre_ub_all[-1][:batch], pre_ub_all[-1][:batch], pre_ub_all[-1][batch:]])

                new_candidate = {}

                # breakpoint()

                for d in range(len(lower_bounds)):
                    # for each layer except the last output layer
                    if len(zero_indices_batch[d]):
                        # we set lower = 0 in first half batch, and upper = 0 in second half batch
                        lower_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                        upper_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                    new_candidate[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]
                # breakpoint()
        # create new_x here since batch may change
        if FIXED_SPLIT and beta:
            ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                    x_L=self.x.ptb.x_L.repeat(batch + diving_batch, 1, 1, 1),
                                    x_U=self.x.ptb.x_U.repeat(batch + diving_batch, 1, 1, 1))
            new_x = BoundedTensor(self.x.data.repeat(batch + diving_batch, 1, 1, 1), ptb)
            c = None if self.c is None else self.c.repeat(new_x.shape[0], 1, 1)
        else:
            ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                    x_L=self.x.ptb.x_L.repeat(batch * 2 + diving_batch, 1, 1, 1),
                                    x_U=self.x.ptb.x_U.repeat(batch * 2 + diving_batch, 1, 1, 1))
            new_x = BoundedTensor(self.x.data.repeat(batch * 2 + diving_batch, 1, 1, 1), ptb)
            c = None if self.c is None else self.c.repeat(new_x.shape[0], 1, 1)

        # self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

        if len(slopes) > 0:
            if FIXED_SPLIT and beta:
                should_duplicate = False
            else:
                should_duplicate = True
            # set slope here again
            self.set_slope(self.net, slopes, diving_batch=diving_batch, should_duplicate=should_duplicate)

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': beta, 'ob_single_node_split': True,
                'ob_update_by_layer': layer_set_bound, 'ob_optimizer':optimizer}})
            with torch.no_grad():
                print(new_x.shape)
                lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='backward',
                                                 new_interval=new_candidate, bound_upper=False, return_A=False)
            return lb

        return_A = True if get_upper_bound else False  # we need A matrix to consturct adv example
        if layer_set_bound:
            start_beta_bound_time = time.time()
            self.net.set_bound_opts({'optimize_bound_args':
                                         {'ob_beta': beta, 'ob_single_node_split': True,
                                          'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                          'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                                          'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            # if diving_batch != 0: import pdb; pdb.set_trace()
            tmp_ret = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=return_A, bound_upper=False, needed_A_dict=self.needed_A_dict)
            beta_bound_time += time.time() - start_beta_bound_time

            # we don't care about the upper bound of the last layer

        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                    'ob_iteration': iteration, 'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                    'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            tmp_ret = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=return_A, bound_upper=False, needed_A_dict=self.needed_A_dict)

        if get_upper_bound:
            lb, _, A = tmp_ret
            primal_x, ub = self.get_primal_upper_bound(A)
        else:
            lb, _ = tmp_ret
            ub = lb + 99  # dummy upper bound
            primal_x = None

        bound_time += time.time() - start_bound_time

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()

            lAs = self.get_lA_parallel(transfer_net)
            if len(slopes) > 0:
                ret_s = self.get_slope(transfer_net)

            if beta:
                ret_b = self.get_beta(transfer_net, splits_per_example, diving_batch=diving_batch)

            # Reorganize tensors.
            if FIXED_SPLIT and beta:
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, batch, diving_batch=diving_batch)
            else:
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, batch * 2, diving_batch=diving_batch)

            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_last.cpu())
            print("lower_bounds_new shape:\n", [tmp.shape for tmp in lower_bounds_new])
            if not get_upper_bound:
                # Do not set to min so the primal is always corresponding to the upper bound.
                upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_last.cpu())
            # reshape the results based on batch.

            if FIXED_SPLIT:
                for i in range(batch):
                    ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                    ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                for i in range(batch, batch + diving_batch):
                    ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                    ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
            else:
                for i in range(batch):
                    ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                    ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]

                    ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                    ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]
                for i in range(2 * batch, 2 * batch + diving_batch):
                    ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                    ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]


            finalize_time = time.time() - start_finalize_time

        ret_p, primals = None, None
        if get_upper_bound:
            # print("opt crown:", lb)
            # primal_values, integer_primals = self.get_neuron_primal(primal_x, lb=lower_bounds_new, ub=upper_bounds_new)
            # correct intermediate primal should produce the correct primal output lb
            # print("primal lb with beta:", primal_values[-1])
            # print("### Extracting primal values and mixed integers with beta for intermeidate nodes done ###")
            # exit()
            # primals = {"p": primal_values, "z": integer_primals}
            pass

        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')

        # if primals is not None: ret_p = self.layer_wise_primals(primals)

        # assert (ret_p[1]['p'][0][0] == primal_x[1]).all()
        return ret_l, ret_u, lAs, ret_s, ret_b, new_split_history, best_intermediate_betas, primal_x

    def get_neuron_primal(self, input_primal, lb, ub, slope_opt=None):
        # calculate the primal values for intermediate neurons
        # slope_opt is a list, each element has the dict for slopes of each batch

        if slope_opt is None:
            slope_opt = self.get_slope(self.net)
        
        batch_size = input_primal.shape[0]
        primal_values = [input_primal]
        # save the integer primal values in MIP constructions
        integer_primals = []
        primal = input_primal
        relu_idx = 0
        keys = list(slope_opt[0].keys())
        output_key = list(slope_opt[0][keys[0]].keys())[-1]
        # load saved primals from gurobi lp for debug
        # gurobi_primals = None
        # gurobi_primals = [np.load(f"gurobi_primals/{i}.npy") for i in range(10)]
        # gurobi_integer_primals = [np.load(f"gurobi_primals/z_relu{relu_idx}.npy") for relu_idx in range(5)]

        dual_values = torch.zeros((batch_size, 1), device=primal.device)

        for layer_idx, layer in enumerate(self.layers):
            # print(type(layer), primal.shape)
            # if gurobi_primals is not None and layer_idx < len(gurobi_primals):
            #     gp = torch.tensor(gurobi_primals[layer_idx]).float().to(primal.device)
            #     gip = torch.tensor(gurobi_integer_primals[relu_idx]).float().to(primal.device)
            if not isinstance(layer, nn.ReLU):
                # just propagate the primal value if linear function or flatten layer
                primal = layer(primal)
            else:
                # only store input, pre_relu primal values, and output primals
                primal_values.append(primal.clone().detach())

                # handling nonlinear relus for primal propagations
                # we can use the lA from get_mask_lA_parallel but relu.lA is more straightforward
                # lA = lAs[0][relu_idx]
                lA = self.net.relus[relu_idx].lA.squeeze(0)

                # primal on lower boundary: lA<=0 & unstable 
                u, l = ub[relu_idx].to(primal.device), lb[relu_idx].to(primal.device)
                unstable = (u > 0).logical_and(l < 0)
                
                # slope = slope_opt[which batch][keys[relu_idx]][output_key][0, 0]
                slope = self.net.relus[relu_idx].alpha[output_key][0, 0].to(primal.device)
                primal_l = primal * slope
                z_l =  primal / u
                z_l[z_l < 0] = 0

                # primal on upper boundary: lA>0 & unstable 
                slope = (u / (u-l))
                bias = (-u * l / (u - l))
                primal_u = (primal * slope + bias).detach()
                z_u = (primal - l) / (u - l)
                # calculate z integer first, using last linear layer node primal values
                z = z_u
                z[(lA>0).logical_and(unstable)] = z_l[(lA>0).logical_and(unstable)]
                
                primal[(lA<=0).logical_and(unstable)] = primal_u[(lA<=0).logical_and(unstable)].detach()
                primal[(lA>0).logical_and(unstable)] = primal_l[(lA>0).logical_and(unstable)].detach()
                primal[(u<0)] = 0

                if self.net.relus[relu_idx].sparse_beta is not None and self.net.relus[relu_idx].sparse_beta.nelement() != 0:
                    beta_loc = self.net.relus[relu_idx].sparse_beta_loc
                    sparse_beta = self.net.relus[relu_idx].sparse_beta * self.net.relus[relu_idx].sparse_beta_sign
                    
                    # we select split neuron from primal with tuple index
                    beta_loc_tuple = (torch.ones(beta_loc.shape).nonzero(as_tuple=True)[0], beta_loc.view(-1))
                    # we get the pre relu primal values for each split node
                    selected_primals = primal.view(batch_size, -1).gather(dim=1, index=beta_loc)
                    # we will add beta * split node pre relu primal to the eventual primal output obj 
                    dual_values = dual_values + (sparse_beta * selected_primals).sum(1, keepdim=True).detach()
                    # for split node, we need to force choice 1 to be pre relu primal and 0 for choice 0
                    beta_c = (self.net.relus[relu_idx].sparse_beta_sign.view(-1) + 1) / 2
                    primal.view(batch_size, -1)[beta_loc_tuple] = primal_values[-1].view(batch_size, -1)[beta_loc_tuple] * beta_c
                    # force mixed integer z to be 1 and 0 for split nodes
                    z.view(batch_size, -1)[beta_loc_tuple] = beta_c

                # store the primal values of mixed integers
                # if z[unstable].view(-1).shape[0] % batch_size !=0:
                #     import pdb; pdb.set_trace()
                ip = torch.ones(z.shape, device=z.device) * (-1.)
                ip[unstable] = z[unstable]
                integer_primals.append(ip.view(batch_size, -1))

                # We should not force primal to be larger than 0, otherwise not correct !!!
                # primal = layer(primal)
                # if relu_idx == 4: import pdb; pdb.set_trace()
                relu_idx += 1

            # primal_values.append(primal.clone().detach())

        primal_values.append(primal.clone().detach())
        primal_values[-1] = primal_values[-1] - dual_values

        integer_primals = [iv.to(device='cpu', non_blocking=True) for iv in integer_primals]
        primal_values = [pv.to(device='cpu', non_blocking=True) for pv in primal_values]

        return primal_values, integer_primals


    def layer_wise_primals(self, primals):
        # originally layer -> batch, 
        # now need to be a list with batch elements
        neuron_primals, integer_primals = primals["p"], primals["z"]
        ret_p = []
        for bi in range(neuron_primals[0].size(0)):
            pv, iv = [], []
            for layer_idx in range(len(neuron_primals)):
                pv.append(neuron_primals[layer_idx][bi:bi + 1])
            for relu_idx in range(len(integer_primals)):
                iv.append(integer_primals[relu_idx][bi:bi + 1])
            ret_p.append({"p": pv, "z": iv})
        return ret_p


    def build_the_model(self, input_domain, x, stop_criterion_func=stop_criterion_sum(0)):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]


        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        # first get CROWN bounds
        # Reference bounds are intermediate layer bounds from initial CROWN bounds.
        lb, ub, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
        print('initial CROWNNNN bounds:', lb, ub)
        if stop_criterion_func(lb).all().item():
            # Fast path. Initial CROWN bound can verify the network.
            if not self.simplify:
                return None, lb[-1], None, None, None, None, None, None, None, None, None, None
            else:
                return None, lb[-1].item(), None, None, None, None, None, None, None, None, None, None
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                 'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                 'ob_early_stop': False, 'ob_verbose': 0,
                                 'ob_keep_best': True, 'ob_update_by_layer': True,
                                 'ob_lr': lr_init_alpha, 'ob_init': False,
                                 'ob_loss_reduction_func': loss_reduction_func,
                                 'ob_stop_criterion_func': stop_criterion_func,
                                 'ob_lr_decay': lr_decay}})
        lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
                                                 bound_upper=False, aux_reference_bounds=aux_reference_bounds)
        slope_opt = self.get_slope(self.net)[0]  # initial with one node only

        # build a complete A_dict
        # self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
        # self.layer_names.sort()

        # update bounds
        print('initial alpha-CROWN bounds:', lb, ub)
        primals, duals, mini_inp = None, None, None
        # mini_inp, primals = self.get_primals(self.A_dict)
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)

        if not self.simplify or stop_criterion_func(lb[-1]):
            history = [[[], []] for _ in range(len(self.net.relus))]
            return ub[-1], lb[-1], mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

        # for each pre-relu layer, we initial 2 lists for active and inactive split
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

    def build_the_model_with_refined_bounds(self, input_domain, x, refined_lower_bounds, refined_upper_bounds,
                                            stop_criterion_func=stop_criterion_sum(0), reference_slopes=None):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        no_joint_opt = arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        
        self.x = x
        self.input_domain = input_domain

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        slope_opt = None
        if not no_joint_opt:
            ######## using bab_verification_mip_refine.py ########
            lb, ub = refined_lower_bounds, refined_upper_bounds
        primals, duals, mini_inp = None, None, None

        return_A = True if get_upper_bound else False
        if get_upper_bound:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        # first get CROWN bounds
        self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c)
        # If we already have slopes available, we initialize them.
        if reference_slopes is not None:
            for m in self.net.relus:
                for spec_name, alpha in m.alpha.items():
                    # each slope size is (2, spec, 1, *shape); batch size is 1.
                    if spec_name in reference_slopes[m.name]:
                        reference_alpha = reference_slopes[m.name][spec_name]
                        if alpha.size() == reference_alpha.size():
                            print(f"setting alpha for layer {m.name} start_node {spec_name}")
                            alpha.data.copy_(reference_alpha)
                        else:
                            print(f"not setting layer {m.name} start_node {spec_name} because shape mismatch ({alpha.size()} != {reference_alpha.size()})")
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                 'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                 'ob_early_stop': False, 'ob_verbose': 0,
                                 'ob_keep_best': True, 'ob_update_by_layer': True,
                                 'ob_lr': lr_init_alpha, 'ob_init': False,
                                 'ob_loss_reduction_func': loss_reduction_func,
                                 'ob_stop_criterion_func': stop_criterion_func,
                                 'ob_lr_decay': lr_decay}})

        if no_joint_opt:
            # using init crown bounds as new_interval
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                new_interval[nd] = [layer.inputs[0].lower, layer.inputs[0].upper] ##!!
                # reference_bounds[nd] = [lb[i], ub[i]]
        else:
            # using refined bounds with init opt crown
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                print(i, nd, lb[i].shape)
                new_interval[nd] = [lb[i], ub[i]]
                # reference_bounds[nd] = [lb[i], ub[i]]
        ret = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='crown-optimized', return_A=return_A,
                                    new_interval=new_interval, bound_upper=False, needed_A_dict=self.needed_A_dict)
                                    #reference_bounds=reference_bounds, bound_upper=False)
        if return_A:
            lb, ub, A = ret
        else:
            lb, ub = ret

        print("alpha-CROWN with fixed intermediate bounds:", lb, ub)
        slope_opt = self.get_slope(self.net)[0]
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds

        if False and stop_criterion_func(lb[-1]):
            #################
            # using refined bounds with LP
            glb = self.build_the_model_lp(lb, ub)
            lb[-1] = torch.tensor([[glb]])
            print("LP with intermediate bounds from MIP:", lb[-1])
            # #################
            

        mask, lA = self.get_mask_lA_parallel(self.net)
        #import pdb; pdb.set_trace()
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            print("opt crown:", lb[-1])
            primal_x, ub_x = self.get_primal_upper_bound(A)
            print("### Extracting primal values for inputs done ###")

            # get the primal values for intermediate layers
            primal_values, integer_primals = self.get_neuron_primal(primal_x, lb, ub)
            # correct intermediate primal should produce the correct primal output lb
            print("primal lb:", primal_values[-1])
            print("### Extracting primal values and mixed integers for intermeidate nodes done ###")
        
            primals = {"p": primal_values, "z": integer_primals}
        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history


    def build_solver_model(self, lower_bounds, upper_bounds, timeout, mip_multi_proc=None, 
            mip_threads=1, input_domain=None, target=None, model_type="mip", simplified=False):
        # build the gurobi model according to the alpha-CROWN bounds
        return build_solver_model(self, lower_bounds, upper_bounds, timeout, mip_multi_proc, mip_threads, input_domain, target, model_type, simplified)


    def build_the_model_mip(self, lower_bounds, upper_bounds, simplified=False, labels_to_verify=None):
        # using the built gurobi model to solve mip formulation
        return build_the_model_mip(self, lower_bounds, upper_bounds, simplified, labels_to_verify=labels_to_verify)
    

    def build_the_model_lp(self, lower_bounds, upper_bounds, using_integer=True, simplified=True):
        # using the built gurobi model to solve lp formulation
        return build_the_model_lp(self, lower_bounds, upper_bounds, using_integer, simplified)


    def update_the_model_lp(self, lower_bounds, upper_bounds, decision, choice, using_integer=True, refine_intermediate_neurons=False):
        # solve split bounds with gurobi, set refine_intermeidate_neurons to be True if we need refinement
        return update_the_model_lp(self, lower_bounds, upper_bounds, decision, choice, using_integer, refine_intermediate_neurons)


    def update_the_model_mip(self, relu_mask, lower_bounds, upper_bounds, decision, choice):
        # solve split bounds with gurobi mip formulation
        return update_the_model_mip(self, relu_mask, lower_bounds, upper_bounds, decision, choice)


    def update_mip_model_fix_relu(self, relu_idx, status, target=None, async_mip=False, best_adv=None, adv_activation_pattern=None):
        # update mip model by manually fixing intermediate relus
        return update_mip_model_fix_relu(self, relu_idx, status, target, async_mip, best_adv, adv_activation_pattern)

    
    def build_the_model_mip_refine(m, lower_bounds, upper_bounds, stop_criterion_func=stop_criterion_min(1e-4), score=None, FSB_sort=True, topk_filter=1.):
            # using mip solver to refine the bounds of intermediate nodes
            return  build_the_model_mip_refine(m, lower_bounds, upper_bounds, stop_criterion_func, score, FSB_sort, topk_filter)

    def visualize(self):
        return
        print("visualizing internal state")
        relu_layers = []
        for layer in self.net.children():
            if isinstance(layer, BoundRelu):
                print(layer)
                relu_layers.append(layer)
