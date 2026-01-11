#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:51:51 2022

@author: Bing Bai
"""

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
import os
import sys
import math
import pickle
import copy
from csv import writer
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import argparse
from scipy.optimize import linprog
from scipy.stats import norm
from os.path import join
import torch

# GPU device setup (will be initialized by set_gpu_device after parsing arguments)
torch_device = None
torch_dtype = torch.float32

def set_gpu_device(gpu_id=None):
    """Set the GPU device to use, otherwise GPU 0."""
    global torch_device
    
    gpu_id = int(os.environ.get('CUDA_DEVICE', '0')) if gpu_id is None else gpu_id
    num_gpus = torch.cuda.device_count()
    selected_gpu = gpu_id if num_gpus > gpu_id else 0
    
    if selected_gpu != gpu_id:
        print(f'Warning: GPU {gpu_id} not available (only {num_gpus} GPUs), using GPU 0')
    
    torch_device = torch.device(f"cuda:{selected_gpu}")
    return torch_device


class OptimizeLatentModel:
    def __init__(self, args, prob_slot_h, prob_slot_s, V_slot, par_init, par_est, n_day):
        self.args = args
        self.prob_slot_h = prob_slot_h
        self.prob_slot_s = prob_slot_s
        self.V_slot = V_slot
        self.par_init = par_init
        self.par_est = par_est
        self.n_day = n_day
        self.experiment = ''
        
        # PredictLabel initialization constants
        self.const1 = 0.5772 * self.par_est[5]
        self.const2 = 0.5772 * self.par_est[6]
        self.const3 = 0.5772 * self.par_est[7]
        self.norm1 = self.norm2 = self.norm3 = 0
        
        self._read_data()
    
    def _read_data(self):
        print('Process data')
        self.customer_data = customer_data
        self.customer_data_A = self.customer_data.loc[self.customer_data['n_station'] > 0]
        self.customer_data_B = self.customer_data.loc[self.customer_data['n_station'] <= 0]
        self.customer_data_A.reset_index(drop=True, inplace=True)
        self.customer_data_B.reset_index(drop=True, inplace=True)
        self.time_slot = np.concatenate((np.arange(8, 22), np.arange(32, 46), np.arange(56, 70)))
        self.time_slot0 = np.arange(8, 22)
    
    def _append_list_as_row(self, filename, list_of_elem):
        """Append a list as a row to a CSV file."""
        with open(filename, 'a+', newline='') as f:
            writer(f).writerow(list_of_elem)  
            
        
    def _matrix_setup(self):
        # =============================================================================
        #     Matrix setup
        # =============================================================================
        
        pickup_max_len = 0
        deliver_max_len = 0
        for i in self.customer_data_A.index:
            if self.customer_data_A.at[i,'pickup'] is not np.nan and self.customer_data_A.at[i,'pickup'].shape[0] > pickup_max_len:
                pickup_max_len = self.customer_data_A.at[i,'pickup'].shape[0]
            if self.customer_data_A.at[i,'deliver'] is not np.nan and self.customer_data_A.at[i,'deliver'].shape[0] > deliver_max_len:
                deliver_max_len = self.customer_data_A.at[i,'deliver'].shape[0]
            
        self.pickup3d = np.empty((len(self.customer_data_A), pickup_max_len, 42))
        self.deliver3d = np.empty((len(self.customer_data_A), deliver_max_len, 42))
        self.pickup3d[:] = np.nan
        self.deliver3d[:] = np.nan
        for i in self.customer_data_A.index:
            if self.customer_data_A.at[i,'pickup'] is not np.nan:
                current_len = self.customer_data_A.at[i,'pickup'].shape[0]
                self.pickup3d[i,0:current_len,:] = self.customer_data_A.at[i,'pickup']
            if self.customer_data_A.at[i,'deliver'] is not np.nan:
                current_len = self.customer_data_A.at[i,'deliver'].shape[0]
                self.deliver3d[i,0:current_len,:] = self.customer_data_A.at[i,'deliver']
        
        
        
    def _optimization_setup(self,
                              seed=0, # only used with bootstrap
                              ):
        if self.experiment == 'vanilla':
            self.bootstrap_seed = seed
            self.par = self.par_init
            
        if self.experiment =='bootstrap':
            self.par = self.par_est.copy()  # Make a copy to avoid modifying original
            self.bootstrap_seed = seed
            # resample
            self.customer_data = customer_data.sample(frac=1,replace=True,random_state = seed)
            self.customer_data_A = self.customer_data.loc[self.customer_data['n_station']>0,]
            self.customer_data_B = self.customer_data.loc[self.customer_data['n_station']<=0,]
            self.customer_data_A.reset_index(drop=True, inplace=True)
            self.customer_data_B.reset_index(drop=True, inplace=True)
        
        self._matrix_setup()
        
        

    
        
    def _optimization_run(self):
        """Run optimization using GPU or CPU."""
        self.iter = 0
        self.nllloss = []
        
        # Use GPU loglikelihood function (GPU is required)
        loglikelihood_func = self._loglikelihood_gpu
        print(f'Running optimization on GPU ({torch_device})...', flush=True)
        
        # Set bounds: scale parameters (5-7) must be positive, probability parameters (8-9) in [0,1]
        # Distance sensitivity parameters (3, 13, 18) are set to be negative only on simulated data to penalize the distance sensitivity in counterfactual analysis
        bounds = [(-np.inf, np.inf)] * 23
        for idx in [5, 6, 7]:
            bounds[idx] = (0.001, np.inf)
        for idx in [8, 9]:
            bounds[idx] = (0.001, 0.999)
        for idx in [3, 13, 18]:
            bounds[idx] = (-np.inf, -0.001)  
        
        start_time = time.time()
        self.result = minimize(loglikelihood_func, self.par, method=self.args.method,
                               bounds=bounds,
                               options={'maxiter': self.args.maxiter_num,
                                       'maxfev': self.args.maxfev_num,
                                       'xatol': self.args.xatol_num,
                                       'fatol': self.args.fatol_num})
        self.time_record = (time.time() - start_time) / 3600
        return self.result
    
       
    def _loglikelihood(self, par):
        """CPU version of log-likelihood function."""
        self.iter += 1
        
        beta_x01 = np.array([0, 0] + [par[0]]*6 + [par[1]]*4 + [0, 0])
        beta_x02 = np.array([0, 0] + [par[10]]*6 + [par[11]]*4 + [0, 0])
        beta_x03 = np.array([0, 0] + [par[15]]*6 + [par[16]]*4 + [0, 0])
        beta_x1 = np.array(([0, 0] + [par[0]]*6 + [par[1]]*4 + [0, 0]) * 3)
        beta_x2 = np.array(([0, 0] + [par[10]]*6 + [par[11]]*4 + [0, 0]) * 3)
        beta_x3 = np.array(([0, 0] + [par[15]]*6 + [par[16]]*4 + [0, 0]) * 3)
        
        U_exp_s1 = beta_x1 + par[2] * self.time_slot
        U_exp_s2 = beta_x2 + par[12] * self.time_slot
        U_exp_s3 = beta_x3 + par[17] * self.time_slot
        
        # Work in log domain to avoid overflow
        log_U_exp_exp_s1 = U_exp_s1 / par[5]
        log_U_exp_exp_s2 = U_exp_s2 / par[6]
        log_U_exp_exp_s3 = U_exp_s3 / par[7]
        
        utility_hx1 = par[20] + sum(self.prob_slot_h * (beta_x01 + par[2] * self.time_slot0)) + 0.5772 * par[5]
        utility_hx2 = par[21] + sum(self.prob_slot_h * (beta_x02 + par[12] * self.time_slot0)) + 0.5772 * par[6]
        utility_hx3 = par[22] + sum(self.prob_slot_h * (beta_x03 + par[17] * self.time_slot0)) + 0.5772 * par[7]
        
        # Compute log(V_slot * U_exp_exp_s) in log domain without logsumexp
        # For each row i: log(sum_j V_slot[i,j] * exp(log_U_exp_exp_s[j]))
        # Use stable computation: subtract max before exp to avoid overflow
        log_V_U1 = np.zeros(self.V_slot.shape[0])
        log_V_U2 = np.zeros(self.V_slot.shape[0])
        log_V_U3 = np.zeros(self.V_slot.shape[0])
        
        for i in range(self.V_slot.shape[0]):
            # Get indices where V_slot[i,j] > 0 (i.e., V_slot[i,j] == 1)
            mask = self.V_slot[i, :] > 0
            if np.any(mask):
                # Get values for non-zero positions
                log_vals_1 = log_U_exp_exp_s1[mask]
                log_vals_2 = log_U_exp_exp_s2[mask]
                log_vals_3 = log_U_exp_exp_s3[mask]
                
                # Subtract max to avoid overflow, then add it back
                max_1 = np.max(log_vals_1)
                max_2 = np.max(log_vals_2)
                max_3 = np.max(log_vals_3)
                
                log_V_U1[i] = max_1 + np.log(np.sum(np.exp(log_vals_1 - max_1)))
                log_V_U2[i] = max_2 + np.log(np.sum(np.exp(log_vals_2 - max_2)))
                log_V_U3[i] = max_3 + np.log(np.sum(np.exp(log_vals_3 - max_3)))
            else:
                # If all V_slot[i,j] = 0, result is log(0) = -inf
                log_V_U1[i] = -np.inf
                log_V_U2[i] = -np.inf
                log_V_U3[i] = -np.inf
        
        utility_sx1 = (par[4] + par[3] * self.customer_data['closest_distance'] + 
                      sum(self.prob_slot_s * log_V_U1) * par[5] + 0.5772 * par[5])
        utility_sx2 = (par[14] + par[13] * self.customer_data['closest_distance'] + 
                      sum(self.prob_slot_s * log_V_U2) * par[6] + 0.5772 * par[6])
        utility_sx3 = (par[19] + par[18] * self.customer_data['closest_distance'] + 
                      sum(self.prob_slot_s * log_V_U3) * par[7] + 0.5772 * par[7])
        
        utility_sx1 = utility_sx1.values
        utility_sx2 = utility_sx2.values
        utility_sx3 = utility_sx3.values
        
        # 3D operations - work in log domain
        pickup_1 = np.einsum('ijk,k->ij', self.pickup3d, U_exp_s1)
        pickup_2 = np.einsum('ijk,k->ij', self.pickup3d, U_exp_s2) 
        pickup_3 = np.einsum('ijk,k->ij', self.pickup3d, U_exp_s3)
        pickup_sum_1 = np.nansum(pickup_1, axis=1) / par[5]
        pickup_sum_2 = np.nansum(pickup_2, axis=1) / par[6]
        pickup_sum_3 = np.nansum(pickup_3, axis=1) / par[7]
        
        # Deliver operations: clip to avoid overflow, then compute einsum and take log
        log_U_exp_exp_s1_clipped = np.clip(log_U_exp_exp_s1, -700, 700)
        log_U_exp_exp_s2_clipped = np.clip(log_U_exp_exp_s2, -700, 700)
        log_U_exp_exp_s3_clipped = np.clip(log_U_exp_exp_s3, -700, 700)
        
        U_exp_exp_s1_safe = np.exp(log_U_exp_exp_s1_clipped)
        U_exp_exp_s2_safe = np.exp(log_U_exp_exp_s2_clipped)
        U_exp_exp_s3_safe = np.exp(log_U_exp_exp_s3_clipped)
        
        deliver_1 = np.einsum('ijk,k->ij', self.deliver3d, U_exp_exp_s1_safe)
        deliver_2 = np.einsum('ijk,k->ij', self.deliver3d, U_exp_exp_s2_safe)
        deliver_3 = np.einsum('ijk,k->ij', self.deliver3d, U_exp_exp_s3_safe)
        
        # Take log, handling zeros
        log_deliver_1 = np.log(np.maximum(deliver_1, 1e-323))
        log_deliver_2 = np.log(np.maximum(deliver_2, 1e-323))
        log_deliver_3 = np.log(np.maximum(deliver_3, 1e-323))
        
        # log(prod) = sum(log), handle NaN by replacing with very negative value
        log_deliver_prod_1 = np.nansum(log_deliver_1, axis=1)
        log_deliver_prod_2 = np.nansum(log_deliver_2, axis=1)
        log_deliver_prod_3 = np.nansum(log_deliver_3, axis=1)
        
        # Replace -inf with very negative value (equivalent to 1e-323 in original)
        log_deliver_prod_1[~np.isfinite(log_deliver_prod_1)] = np.log(1e-323)
        log_deliver_prod_2[~np.isfinite(log_deliver_prod_2)] = np.log(1e-323)
        log_deliver_prod_3[~np.isfinite(log_deliver_prod_3)] = np.log(1e-323)
        
        # Clip log_deliver_prod to prevent extreme values that cause numerical instability
        # If log_deliver_prod is too negative (e.g., < -1000), it makes log_term extremely large
        # This happens when deliver values are extremely small, which indicates the parameters
        # are in a region where the likelihood is essentially zero
        # Clip to a reasonable minimum to prevent numerical overflow
        min_log_deliver = -500  # log(1e-217) - reasonable minimum for numerical stability
        log_deliver_prod_1 = np.clip(log_deliver_prod_1, min_log_deliver, None)
        log_deliver_prod_2 = np.clip(log_deliver_prod_2, min_log_deliver, None)
        log_deliver_prod_3 = np.clip(log_deliver_prod_3, min_log_deliver, None)
        
        # Split customers by station usage
        index_has_station = self.customer_data['n_station'] > 0
        index_no_station = self.customer_data['n_station'] <= 0
        
        utility_sx1_A = utility_sx1[index_has_station]
        utility_sx2_A = utility_sx2[index_has_station]
        utility_sx3_A = utility_sx3[index_has_station]
        utility_sx1_B = utility_sx1[index_no_station]
        utility_sx2_B = utility_sx2[index_no_station]
        utility_sx3_B = utility_sx3[index_no_station]
        
        # Compute result_cus_A in log domain
        # Original: par[8]*(1+exp(uh)+exp(us))**(-n_day) * exp(...) / deliver_prod
        # Log domain: log(par[8]) - n_day*log(1+exp(uh)+exp(us)) + (...) - log_deliver_prod
        n_home_A = self.customer_data_A['n_home'].values
        n_station_A = self.customer_data_A['n_station'].values
        
        # Type 1 term in log domain
        # log(1+exp(uh)+exp(us)) = log(exp(0)+exp(uh)+exp(us)) = max(0,uh,us) + log(exp(0-max) + exp(uh-max) + exp(us-max))
        max_denom_1 = np.maximum(np.maximum(0, utility_hx1), utility_sx1_A)
        log_denom_1 = max_denom_1 + np.log(np.exp(0 - max_denom_1) + np.exp(utility_hx1 - max_denom_1) + np.exp(utility_sx1_A - max_denom_1))
        log_term1_A = (np.log(par[8]) - self.n_day * log_denom_1 + 
                      n_home_A * utility_hx1 + n_station_A * utility_sx1_A + 
                      pickup_sum_1 - log_deliver_prod_1)
        
        # Type 2 term in log domain
        max_denom_2 = np.maximum(np.maximum(0, utility_hx2), utility_sx2_A)
        log_denom_2 = max_denom_2 + np.log(np.exp(0 - max_denom_2) + np.exp(utility_hx2 - max_denom_2) + np.exp(utility_sx2_A - max_denom_2))
        log_term2_A = (np.log(par[9]) - self.n_day * log_denom_2 + 
                      n_home_A * utility_hx2 + n_station_A * utility_sx2_A + 
                      pickup_sum_2 - log_deliver_prod_2)
        
        # Type 3 term in log domain
        max_denom_3 = np.maximum(np.maximum(0, utility_hx3), utility_sx3_A)
        log_denom_3 = max_denom_3 + np.log(np.exp(0 - max_denom_3) + np.exp(utility_hx3 - max_denom_3) + np.exp(utility_sx3_A - max_denom_3))
        log_term3_A = (np.log(1 - par[8] - par[9]) - self.n_day * log_denom_3 + 
                      n_home_A * utility_hx3 + n_station_A * utility_sx3_A + 
                      pickup_sum_3 - log_deliver_prod_3)
        
        # Sum in log domain: log(sum(exp(a), exp(b), exp(c)))
        # Use stable computation: max_term + log(exp(a-max_term) + exp(b-max_term) + exp(c-max_term))
        max_term_A = np.maximum(np.maximum(log_term1_A, log_term2_A), log_term3_A)
        result_cus_A_log = max_term_A + np.log(np.exp(log_term1_A - max_term_A) + 
                                               np.exp(log_term2_A - max_term_A) + 
                                               np.exp(log_term3_A - max_term_A))
        
        # Compute result_cus_B in log domain
        n_home_B = self.customer_data_B['n_home'].values
        
        # Type 1 term in log domain
        max_denom_1_B = np.maximum(np.maximum(0, utility_hx1), utility_sx1_B)
        log_denom_1_B = max_denom_1_B + np.log(np.exp(0 - max_denom_1_B) + np.exp(utility_hx1 - max_denom_1_B) + np.exp(utility_sx1_B - max_denom_1_B))
        log_term1_B = (np.log(par[8]) - self.n_day * log_denom_1_B + 
                      n_home_B * utility_hx1)
        
        # Type 2 term in log domain
        max_denom_2_B = np.maximum(np.maximum(0, utility_hx2), utility_sx2_B)
        log_denom_2_B = max_denom_2_B + np.log(np.exp(0 - max_denom_2_B) + np.exp(utility_hx2 - max_denom_2_B) + np.exp(utility_sx2_B - max_denom_2_B))
        log_term2_B = (np.log(par[9]) - self.n_day * log_denom_2_B + 
                      n_home_B * utility_hx2)
        
        # Type 3 term in log domain
        max_denom_3_B = np.maximum(np.maximum(0, utility_hx3), utility_sx3_B)
        log_denom_3_B = max_denom_3_B + np.log(np.exp(0 - max_denom_3_B) + np.exp(utility_hx3 - max_denom_3_B) + np.exp(utility_sx3_B - max_denom_3_B))
        log_term3_B = (np.log(1 - par[8] - par[9]) - self.n_day * log_denom_3_B + 
                      n_home_B * utility_hx3)
        
        # Sum in log domain
        max_term_B = np.maximum(np.maximum(log_term1_B, log_term2_B), log_term3_B)
        result_cus_B_log = max_term_B + np.log(np.exp(log_term1_B - max_term_B) + 
                                              np.exp(log_term2_B - max_term_B) + 
                                              np.exp(log_term3_B - max_term_B))
        
        result = -np.sum(result_cus_A_log) - np.sum(result_cus_B_log)
        
        # Check for NaN/Inf
        if np.isnan(result) or ~np.isfinite(result):
            if self.args.verbose:
                n_of_nan_A = np.sum(~np.isfinite(result_cus_A_log))
                n_of_nan_B = np.sum(~np.isfinite(result_cus_B_log))
                print(f'[Warning] {n_of_nan_A} values are not finite (Condition A), {n_of_nan_B} (Condition B).')
            result = 1e10
        
        # Check for abnormally large values
        max_log_term = 100
        if len(log_term1_A) > 0:
            log_term1_max = np.max(log_term1_A)
            log_term2_max = np.max(log_term2_A)
            log_term3_max = np.max(log_term3_A)
            
            if np.any(log_term1_A >= max_log_term) or np.any(log_term2_A >= max_log_term) or np.any(log_term3_A >= max_log_term):
                # If log_term values are too large, the parameters are in an invalid region
                # Return a large penalty to guide optimization away from this region
                result = 1e10
            else:
                # Additional check: if result_cus_A_log or result_cus_B_log have extreme values
                # This can happen even when log_term < max_log_term if there are many customers
                max_result_A = np.max(result_cus_A_log) if len(result_cus_A_log) > 0 else 0
                max_result_B = np.max(result_cus_B_log) if len(result_cus_B_log) > 0 else 0
                min_result_A = np.min(result_cus_A_log) if len(result_cus_A_log) > 0 else 0
                min_result_B = np.min(result_cus_B_log) if len(result_cus_B_log) > 0 else 0
                
                # If result_cus_A_log or result_cus_B_log are extremely large positive values,
                # the final result will be a large negative value, which is suspicious
                if max_result_A > 1000 or max_result_B > 1000:
                    result = 1e10
        
        if np.abs(result) > 1e15:
            result = 1e10
        
        self.nllloss.append(result)
        return result
    
    def _loglikelihood_gpu(self, par):
        """
        GPU-accelerated version of _loglikelihood using PyTorch
        Maintains numerical equivalence with CPU version
        """
        self.iter += 1
        
        # Convert parameters to torch tensors
        par_t = torch.tensor(par, dtype=torch_dtype, device=torch_device)
        
        # Build beta arrays
        beta_x01 = torch.cat([torch.zeros(2, device=torch_device), 
                              par_t[0].repeat(6), 
                              par_t[1].repeat(4), 
                              torch.zeros(2, device=torch_device)])
        beta_x02 = torch.cat([torch.zeros(2, device=torch_device), 
                              par_t[10].repeat(6), 
                              par_t[11].repeat(4), 
                              torch.zeros(2, device=torch_device)])
        beta_x03 = torch.cat([torch.zeros(2, device=torch_device), 
                              par_t[15].repeat(6), 
                              par_t[16].repeat(4), 
                              torch.zeros(2, device=torch_device)])
        beta_x1 = torch.cat([beta_x01, beta_x01, beta_x01])
        beta_x2 = torch.cat([beta_x02, beta_x02, beta_x02])
        beta_x3 = torch.cat([beta_x03, beta_x03, beta_x03])
        
        # Convert time slots to tensors if not already (cache them)
        if not hasattr(self, 'time_slot_t'):
            self.time_slot_t = torch.tensor(self.time_slot, dtype=torch_dtype, device=torch_device)
            self.time_slot0_t = torch.tensor(self.time_slot0, dtype=torch_dtype, device=torch_device)
            self.prob_slot_h_t = torch.tensor(self.prob_slot_h, dtype=torch_dtype, device=torch_device)
            self.prob_slot_s_t = torch.tensor(self.prob_slot_s, dtype=torch_dtype, device=torch_device)
            self.V_slot_t = torch.tensor(self.V_slot, dtype=torch_dtype, device=torch_device)
            self.n_day_t = torch.tensor(self.n_day, dtype=torch_dtype, device=torch_device)
        
        # Compute U_exp_s
        U_exp_s1 = beta_x1 + par_t[2] * self.time_slot_t
        U_exp_s2 = beta_x2 + par_t[12] * self.time_slot_t
        U_exp_s3 = beta_x3 + par_t[17] * self.time_slot_t
        
        # Work in log domain
        log_U_exp_exp_s1 = U_exp_s1 / par_t[5]
        log_U_exp_exp_s2 = U_exp_s2 / par_t[6]
        log_U_exp_exp_s3 = U_exp_s3 / par_t[7]
        
        # Compute utility_hx
        utility_hx1 = (par_t[20] + torch.sum(self.prob_slot_h_t * (beta_x01 + par_t[2] * self.time_slot0_t)) + 
                       0.5772 * par_t[5])
        utility_hx2 = (par_t[21] + torch.sum(self.prob_slot_h_t * (beta_x02 + par_t[12] * self.time_slot0_t)) + 
                       0.5772 * par_t[6])
        utility_hx3 = (par_t[22] + torch.sum(self.prob_slot_h_t * (beta_x03 + par_t[17] * self.time_slot0_t)) + 
                       0.5772 * par_t[7])
        
        # Compute log(V_slot * U_exp_exp_s) in log domain
        log_V_U1 = torch.zeros(self.V_slot_t.shape[0], dtype=torch_dtype, device=torch_device)
        log_V_U2 = torch.zeros(self.V_slot_t.shape[0], dtype=torch_dtype, device=torch_device)
        log_V_U3 = torch.zeros(self.V_slot_t.shape[0], dtype=torch_dtype, device=torch_device)
        
        for i in range(self.V_slot_t.shape[0]):
            mask = self.V_slot_t[i, :] > 0
            if torch.any(mask):
                log_vals_1 = log_U_exp_exp_s1[mask]
                log_vals_2 = log_U_exp_exp_s2[mask]
                log_vals_3 = log_U_exp_exp_s3[mask]
                
                max_1 = torch.max(log_vals_1)
                max_2 = torch.max(log_vals_2)
                max_3 = torch.max(log_vals_3)
                
                log_V_U1[i] = max_1 + torch.log(torch.sum(torch.exp(log_vals_1 - max_1)))
                log_V_U2[i] = max_2 + torch.log(torch.sum(torch.exp(log_vals_2 - max_2)))
                log_V_U3[i] = max_3 + torch.log(torch.sum(torch.exp(log_vals_3 - max_3)))
            else:
                log_V_U1[i] = torch.tensor(-float('inf'), device=torch_device)
                log_V_U2[i] = torch.tensor(-float('inf'), device=torch_device)
                log_V_U3[i] = torch.tensor(-float('inf'), device=torch_device)
        
        # Convert customer data to tensors if not already (cache them)
        if not hasattr(self, 'closest_distance_t'):
            self.closest_distance_t = torch.tensor(
                self.customer_data['closest_distance'].values, 
                dtype=torch_dtype, device=torch_device
            )
            self.n_home_A_t = torch.tensor(
                self.customer_data_A['n_home'].values, 
                dtype=torch_dtype, device=torch_device
            )
            self.n_station_A_t = torch.tensor(
                self.customer_data_A['n_station'].values, 
                dtype=torch_dtype, device=torch_device
            )
            self.n_home_B_t = torch.tensor(
                self.customer_data_B['n_home'].values, 
                dtype=torch_dtype, device=torch_device
            )
            self.index_has_station_t = torch.tensor(
                self.customer_data['n_station'].values > 0, 
                device=torch_device
            )
        
        # Compute utility_sx
        utility_sx1 = (par_t[4] + par_t[3] * self.closest_distance_t + 
                      torch.sum(self.prob_slot_s_t * log_V_U1) * par_t[5] + 0.5772 * par_t[5])
        utility_sx2 = (par_t[14] + par_t[13] * self.closest_distance_t + 
                      torch.sum(self.prob_slot_s_t * log_V_U2) * par_t[6] + 0.5772 * par_t[6])
        utility_sx3 = (par_t[19] + par_t[18] * self.closest_distance_t + 
                      torch.sum(self.prob_slot_s_t * log_V_U3) * par_t[7] + 0.5772 * par_t[7])
        
        # Split utilities for A and B
        utility_sx1_A = utility_sx1[self.index_has_station_t]
        utility_sx2_A = utility_sx2[self.index_has_station_t]
        utility_sx3_A = utility_sx3[self.index_has_station_t]
        utility_sx1_B = utility_sx1[~self.index_has_station_t]
        utility_sx2_B = utility_sx2[~self.index_has_station_t]
        utility_sx3_B = utility_sx3[~self.index_has_station_t]
        
        # 3D operations - convert pickup3d and deliver3d to tensors if needed (cache them)
        if not hasattr(self, 'pickup3d_t'):
            self.pickup3d_t = torch.tensor(self.pickup3d, dtype=torch_dtype, device=torch_device)
            self.deliver3d_t = torch.tensor(self.deliver3d, dtype=torch_dtype, device=torch_device)
        
        # Pickup operations
        pickup_1 = torch.einsum('ijk,k->ij', self.pickup3d_t, U_exp_s1)
        pickup_2 = torch.einsum('ijk,k->ij', self.pickup3d_t, U_exp_s2)
        pickup_3 = torch.einsum('ijk,k->ij', self.pickup3d_t, U_exp_s3)
        
        # Handle NaN in pickup (replace with 0)
        pickup_1 = torch.where(torch.isnan(pickup_1), torch.tensor(0.0, device=torch_device), pickup_1)
        pickup_2 = torch.where(torch.isnan(pickup_2), torch.tensor(0.0, device=torch_device), pickup_2)
        pickup_3 = torch.where(torch.isnan(pickup_3), torch.tensor(0.0, device=torch_device), pickup_3)
        
        pickup_sum_1 = torch.sum(pickup_1, dim=1) / par_t[5]
        pickup_sum_2 = torch.sum(pickup_2, dim=1) / par_t[6]
        pickup_sum_3 = torch.sum(pickup_3, dim=1) / par_t[7]
        
        # Deliver operations - clip to avoid overflow
        log_U_exp_exp_s1_clipped = torch.clamp(log_U_exp_exp_s1, -700, 700)
        log_U_exp_exp_s2_clipped = torch.clamp(log_U_exp_exp_s2, -700, 700)
        log_U_exp_exp_s3_clipped = torch.clamp(log_U_exp_exp_s3, -700, 700)
        
        U_exp_exp_s1_safe = torch.exp(log_U_exp_exp_s1_clipped)
        U_exp_exp_s2_safe = torch.exp(log_U_exp_exp_s2_clipped)
        U_exp_exp_s3_safe = torch.exp(log_U_exp_exp_s3_clipped)
        
        deliver_1 = torch.einsum('ijk,k->ij', self.deliver3d_t, U_exp_exp_s1_safe)
        deliver_2 = torch.einsum('ijk,k->ij', self.deliver3d_t, U_exp_exp_s2_safe)
        deliver_3 = torch.einsum('ijk,k->ij', self.deliver3d_t, U_exp_exp_s3_safe)
        
        # Take log, handling zeros (match CPU version: np.log(np.maximum(deliver_1, 1e-323)))
        # Don't replace NaN with 0, let nansum handle it (match CPU: np.nansum)
        log_deliver_1 = torch.log(torch.clamp(deliver_1, min=1e-323))
        log_deliver_2 = torch.log(torch.clamp(deliver_2, min=1e-323))
        log_deliver_3 = torch.log(torch.clamp(deliver_3, min=1e-323))
        
        # Use nansum to match CPU version (np.nansum ignores NaN)
        log_deliver_prod_1 = torch.nansum(log_deliver_1, dim=1)
        log_deliver_prod_2 = torch.nansum(log_deliver_2, dim=1)
        log_deliver_prod_3 = torch.nansum(log_deliver_3, dim=1)
        
        # Replace NaN/inf with very negative value (match CPU version)
        log_deliver_prod_1 = torch.where(torch.isfinite(log_deliver_prod_1), 
                                         log_deliver_prod_1, 
                                         torch.tensor(np.log(1e-323), device=torch_device))
        log_deliver_prod_2 = torch.where(torch.isfinite(log_deliver_prod_2), 
                                         log_deliver_prod_2, 
                                         torch.tensor(np.log(1e-323), device=torch_device))
        log_deliver_prod_3 = torch.where(torch.isfinite(log_deliver_prod_3), 
                                         log_deliver_prod_3, 
                                         torch.tensor(np.log(1e-323), device=torch_device))
        
        # Clip log_deliver_prod to prevent extreme values (match CPU version)
        min_log_deliver = -500  # log(1e-217) - reasonable minimum for numerical stability
        log_deliver_prod_1 = torch.clamp(log_deliver_prod_1, min=min_log_deliver)
        log_deliver_prod_2 = torch.clamp(log_deliver_prod_2, min=min_log_deliver)
        log_deliver_prod_3 = torch.clamp(log_deliver_prod_3, min=min_log_deliver)
        
        # Compute result_cus_A in log domain
        # Type 1 term
        # utility_hx1 is scalar, utility_sx1_A is array - need proper broadcasting
        utility_hx1_t = utility_hx1 if isinstance(utility_hx1, torch.Tensor) else torch.tensor(utility_hx1, device=torch_device)
        max_denom_1 = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx1_t), utility_sx1_A)
        log_denom_1 = (max_denom_1 + 
                      torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_1) + 
                               torch.exp(utility_hx1_t - max_denom_1) + 
                               torch.exp(utility_sx1_A - max_denom_1)))
        log_term1_A = (torch.log(par_t[8]) - self.n_day_t * log_denom_1 + 
                      self.n_home_A_t * utility_hx1_t + self.n_station_A_t * utility_sx1_A + 
                      pickup_sum_1 - log_deliver_prod_1)
        
        # Type 2 term
        utility_hx2_t = utility_hx2 if isinstance(utility_hx2, torch.Tensor) else torch.tensor(utility_hx2, device=torch_device)
        max_denom_2 = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx2_t), utility_sx2_A)
        log_denom_2 = (max_denom_2 + 
                      torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_2) + 
                               torch.exp(utility_hx2_t - max_denom_2) + 
                               torch.exp(utility_sx2_A - max_denom_2)))
        log_term2_A = (torch.log(par_t[9]) - self.n_day_t * log_denom_2 + 
                      self.n_home_A_t * utility_hx2_t + self.n_station_A_t * utility_sx2_A + 
                      pickup_sum_2 - log_deliver_prod_2)
        
        # Type 3 term
        utility_hx3_t = utility_hx3 if isinstance(utility_hx3, torch.Tensor) else torch.tensor(utility_hx3, device=torch_device)
        max_denom_3 = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx3_t), utility_sx3_A)
        log_denom_3 = (max_denom_3 + 
                      torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_3) + 
                               torch.exp(utility_hx3_t - max_denom_3) + 
                               torch.exp(utility_sx3_A - max_denom_3)))
        log_term3_A = (torch.log(1 - par_t[8] - par_t[9]) - self.n_day_t * log_denom_3 + 
                      self.n_home_A_t * utility_hx3_t + self.n_station_A_t * utility_sx3_A + 
                      pickup_sum_3 - log_deliver_prod_3)
        
        # Sum in log domain
        max_term_A = torch.maximum(torch.maximum(log_term1_A, log_term2_A), log_term3_A)
        result_cus_A_log = (max_term_A + 
                           torch.log(torch.exp(log_term1_A - max_term_A) + 
                                   torch.exp(log_term2_A - max_term_A) + 
                                   torch.exp(log_term3_A - max_term_A)))
        
        # Compute result_cus_B in log domain
        # Type 1 term
        max_denom_1_B = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx1_t), utility_sx1_B)
        log_denom_1_B = (max_denom_1_B + 
                        torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_1_B) + 
                                 torch.exp(utility_hx1_t - max_denom_1_B) + 
                                 torch.exp(utility_sx1_B - max_denom_1_B)))
        log_term1_B = (torch.log(par_t[8]) - self.n_day_t * log_denom_1_B + 
                      self.n_home_B_t * utility_hx1_t)
        
        # Type 2 term
        max_denom_2_B = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx2_t), utility_sx2_B)
        log_denom_2_B = (max_denom_2_B + 
                        torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_2_B) + 
                                 torch.exp(utility_hx2_t - max_denom_2_B) + 
                                 torch.exp(utility_sx2_B - max_denom_2_B)))
        log_term2_B = (torch.log(par_t[9]) - self.n_day_t * log_denom_2_B + 
                      self.n_home_B_t * utility_hx2_t)
        
        # Type 3 term
        max_denom_3_B = torch.maximum(torch.maximum(torch.tensor(0.0, device=torch_device), utility_hx3_t), utility_sx3_B)
        log_denom_3_B = (max_denom_3_B + 
                        torch.log(torch.exp(torch.tensor(0.0, device=torch_device) - max_denom_3_B) + 
                                 torch.exp(utility_hx3_t - max_denom_3_B) + 
                                 torch.exp(utility_sx3_B - max_denom_3_B)))
        log_term3_B = (torch.log(1 - par_t[8] - par_t[9]) - self.n_day_t * log_denom_3_B + 
                      self.n_home_B_t * utility_hx3_t)
        
        # Sum in log domain
        max_term_B = torch.maximum(torch.maximum(log_term1_B, log_term2_B), log_term3_B)
        result_cus_B_log = (max_term_B + 
                           torch.log(torch.exp(log_term1_B - max_term_B) + 
                                   torch.exp(log_term2_B - max_term_B) + 
                                   torch.exp(log_term3_B - max_term_B)))
        
        # Check for abnormally large log_term values (match CPU version)
        # Lower threshold to catch more problematic cases
        max_log_term = 100  # Maximum reasonable value for log_term (reduced from 500)
        if len(log_term1_A) > 0:
            log_term1_max = torch.max(log_term1_A).item()
            log_term2_max = torch.max(log_term2_A).item()
            log_term3_max = torch.max(log_term3_A).item()
            
            if torch.any(log_term1_A >= max_log_term) or torch.any(log_term2_A >= max_log_term) or torch.any(log_term3_A >= max_log_term):
                # If log_term values are too large, the parameters are in an invalid region
                # Return a large penalty to guide optimization away from this region
                result = torch.tensor(1e10, device=torch_device)
            else:
                # Final result
                result = -torch.sum(result_cus_A_log) - torch.sum(result_cus_B_log)
                
                # Additional check: if result_cus_A_log or result_cus_B_log have extreme values
                # This can happen even when log_term < max_log_term if there are many customers
                max_result_A = torch.max(result_cus_A_log).item() if len(result_cus_A_log) > 0 else 0
                max_result_B = torch.max(result_cus_B_log).item() if len(result_cus_B_log) > 0 else 0
                min_result_A = torch.min(result_cus_A_log).item() if len(result_cus_A_log) > 0 else 0
                min_result_B = torch.min(result_cus_B_log).item() if len(result_cus_B_log) > 0 else 0
                
                # If result_cus_A_log or result_cus_B_log are extremely large positive values,
                # the final result will be a large negative value, which is suspicious
                if max_result_A > 1000 or max_result_B > 1000:
                    result = torch.tensor(1e10, device=torch_device)
        else:
            # Final result
            result = -torch.sum(result_cus_A_log) - torch.sum(result_cus_B_log)
        
        # Check for NaN/Inf
        if torch.isnan(result) or not torch.isfinite(result):
            if self.args.verbose:
                n_of_nan_A = torch.sum(~torch.isfinite(result_cus_A_log)).item()
                n_of_nan_B = torch.sum(~torch.isfinite(result_cus_B_log)).item()
                print('[Warning] %d values are not finite (Condition A), %d (Condition B).' % (n_of_nan_A, n_of_nan_B))
            result = torch.tensor(1e10, device=torch_device)
        
        # Check for abnormally large values (match CPU version)
        if torch.abs(result) > 1e15:
            result = torch.tensor(1e10, device=torch_device)
        
        # Convert back to numpy and return
        result_np = result.item()
        
        self.nllloss.append(result_np)
        return result_np
 
    def _simulate_estdata(self):
        """Compute utility parameters for simulation using estimated parameters."""
        # Build beta coefficients for each type (pattern: [norm, norm] + [coef1]*6 + [coef2]*4 + [norm, norm])
        self.beta_x01 = np.array([self.norm1, self.norm1] + [self.par_est[0]]*6 + [self.par_est[1]]*4 + [self.norm1, self.norm1])
        self.beta_x02 = np.array([self.norm2, self.norm2] + [self.par_est[10]]*6 + [self.par_est[11]]*4 + [self.norm2, self.norm2])
        self.beta_x03 = np.array([self.norm3, self.norm3] + [self.par_est[15]]*6 + [self.par_est[16]]*4 + [self.norm3, self.norm3])

        beta_x1 = np.tile(self.beta_x01, 3)
        beta_x2 = np.tile(self.beta_x02, 3)
        beta_x3 = np.tile(self.beta_x03, 3)
        
        # Compute station utility components
        self.U_exp_s1 = beta_x1 + self.par_est[2] * self.time_slot
        self.U_exp_s2 = beta_x2 + self.par_est[12] * self.time_slot
        self.U_exp_s3 = beta_x3 + self.par_est[17] * self.time_slot
        
        self.U_exp_exp_s1 = np.exp(self.U_exp_s1 / self.par_est[5])
        self.U_exp_exp_s2 = np.exp(self.U_exp_s2 / self.par_est[6])
        self.U_exp_exp_s3 = np.exp(self.U_exp_s3 / self.par_est[7])
        
        # Compute home utility
        self.utility_hx1 = (self.par_est[20] + 
                           np.sum(self.prob_slot_h * (self.beta_x01 + self.par_est[2] * self.time_slot0)) + 
                           self.const1)
        self.utility_hx2 = (self.par_est[21] + 
                           np.sum(self.prob_slot_h * (self.beta_x02 + self.par_est[12] * self.time_slot0)) + 
                           self.const2)
        self.utility_hx3 = (self.par_est[22] + 
                           np.sum(self.prob_slot_h * (self.beta_x03 + self.par_est[17] * self.time_slot0)) + 
                           self.const3)
        
        # Compute station utility
        V_U1 = np.matmul(self.V_slot, self.U_exp_exp_s1)
        V_U2 = np.matmul(self.V_slot, self.U_exp_exp_s2)
        V_U3 = np.matmul(self.V_slot, self.U_exp_exp_s3)
        
        self.utility_sx1 = (self.par_est[4] + 
                           self.par_est[3] * self.customer_data['closest_distance'] + 
                           np.sum(self.prob_slot_s * np.log(V_U1)) * self.par_est[5] + 
                           self.const1)
        self.utility_sx2 = (self.par_est[14] + 
                           self.par_est[13] * self.customer_data['closest_distance'] + 
                           np.sum(self.prob_slot_s * np.log(V_U2)) * self.par_est[6] + 
                           self.const2)
        self.utility_sx3 = (self.par_est[19] + 
                           self.par_est[18] * self.customer_data['closest_distance'] + 
                           np.sum(self.prob_slot_s * np.log(V_U3)) * self.par_est[7] + 
                           self.const3)
        
 
    def _simulate_purchase(self, seed):
        """Simulate purchase decisions under different customer type assumptions."""
        np.random.seed(seed)
        customer_data_copy = self.customer_data.copy(deep=True)
        n_cus = customer_data_copy.shape[0]
        
        # Generate random utility shocks
        rand_s = np.random.gumbel(0, 1, n_cus)
        rand_h = np.random.gumbel(0, 1, n_cus)
        rand_np = np.random.gumbel(0, 1, n_cus)
        
        # Simulate customer types by estimated probability
        cus_type_ind = np.random.uniform(0, 1, n_cus)
        cus_type_1 = (cus_type_ind < self.par_est[8]).astype(int)
        cus_type_2 = (cus_type_ind > 1 - self.par_est[9]).astype(int)
        cus_type_3 = 1 - cus_type_1 - cus_type_2
        
        def simu_from_sample(utility_s, utility_h):
            """Simulate purchase decisions given station and home utilities."""
            customer_data_copy['utility_s'] = utility_s
            customer_data_copy['utility_h'] = utility_h
            customer_data_copy['utility_np'] = rand_np
            
            # Prior purchase: only home and outside option
            customer_data_copy['prior_purchase'] = (
                customer_data_copy['utility_np'] < customer_data_copy['utility_h']
            ).astype(int)
            
            # Post purchase: station (2), home (1), or outside (0)
            u_s = customer_data_copy['utility_s']
            u_h = customer_data_copy['utility_h']
            u_np = customer_data_copy['utility_np']
            
            station_best = (u_np < u_s) & (u_h < u_s)
            home_best = (u_np < u_h) & (u_s <= u_h)
            outside_best = (u_h <= u_np) & (u_s <= u_np)
            
            customer_data_copy['post_purchase'] = np.select(
                [station_best, home_best, outside_best], [2, 1, 0]
            )
            
            # Count purchases
            num_purchase_prior = customer_data_copy['prior_purchase'].sum()
            num_purchase_post = (customer_data_copy['post_purchase'] > 0).sum()
            num_purchase_post_sta = (customer_data_copy['post_purchase'] == 2).sum()
            num_purchase_post_home = (customer_data_copy['post_purchase'] == 1).sum()
            
            purchase_increase = (num_purchase_post - num_purchase_prior) / n_cus
            
            return [
                num_purchase_prior / n_cus,
                num_purchase_post / n_cus,
                num_purchase_post_sta / n_cus,
                num_purchase_post_home / n_cus,
                purchase_increase,
                np.mean(utility_h),
                np.mean(utility_s)
            ]
        
        # Simulation 0: Mixed types by estimated probability
        utility_s = (cus_type_1 * self.utility_sx1 + 
                    cus_type_2 * self.utility_sx2 + 
                    cus_type_3 * self.utility_sx3 + rand_s)
        utility_h = (cus_type_1 * self.utility_hx1 + 
                    cus_type_2 * self.utility_hx2 + 
                    cus_type_3 * self.utility_hx3 + rand_h)
        print('Simulating purchase: draw customer types and utilities by estimated probability')
        result0 = simu_from_sample(utility_s, utility_h)
        
        # Simulation 1-3: All customers are type 1, 2, or 3
        type_configs = [
            (1, self.utility_sx1, self.utility_hx1, 'type 1'),
            (2, self.utility_sx2, self.utility_hx2, 'type 2'),
            (3, self.utility_sx3, self.utility_hx3, 'type 3')
        ]
        
        results = [result0]
        for type_num, util_sx, util_hx, type_name in type_configs:
            utility_s = util_sx + rand_s
            utility_h = util_hx + rand_h
            print(f'Simulating purchase: assume everyone is {type_name}')
            results.append(simu_from_sample(utility_s, utility_h))
        
        result1, result2, result3 = results[1], results[2], results[3]
        
        # Create and print results table
        df = pd.DataFrame({
            'Estimates': [
                'Prior-station purchase',
                'Post-station purchase',
                'Purchase rate increase',
                'Post-station purchase with station delivery',
                'Post-station purchase with home delivery'
            ],
            'All': [result0[0], result0[1], result0[4], result0[2], result0[3]],
            'Low_type': [result2[0], result2[1], result2[4], result2[2], result2[3]],
            'Medium_type': [result3[0], result3[1], result3[4], result3[2], result3[3]],
            'High_type': [result1[0], result1[1], result1[4], result1[2], result1[3]]
        })
        df = df.round(3)
        print(df.to_latex(index=False, float_format='%.3f'))

    def _simulate_scenario_purchase(self, seed):
        """
        Simulate purchase with equal time preference (normalized slot values).
        Temporarily modifies parameters to use weighted average slot values.
        """
        # Save original state
        par_est_original = self.par_est.copy()
        norm_original = (self.norm1, self.norm2, self.norm3)
        
        try:
            # Calculate normalized slot values (weighted average) for equal time preference
            # Pattern: [0,0] + [coef1]*6 + [coef2]*4 + [0,0]
            type_configs = [
                (0, 1, 0, 1, 'norm1'),   # Type 1: update indices 0,1 using par_est[0], par_est[1]
                (10, 11, 10, 11, 'norm2'), # Type 2: update indices 10,11 using par_est[10], par_est[11]
                (15, 16, 15, 16, 'norm3')  # Type 3: update indices 15,16 using par_est[15], par_est[16]
            ]
            
            par_est_copy = self.par_est.copy()
            
            for idx1, idx2, est_idx1, est_idx2, norm_attr in type_configs:
                beta_array = np.array([0, 0] + [self.par_est[est_idx1]]*6 + 
                                     [self.par_est[est_idx2]]*4 + [0, 0])
                norm_value = np.sum(self.prob_slot_h * beta_array) / np.sum(self.prob_slot_h)
                setattr(self, norm_attr, norm_value)
                par_est_copy[idx1] = norm_value
                par_est_copy[idx2] = norm_value
            
            # Temporarily use modified parameters
            self.par_est = par_est_copy
            
            # Run simulation with equal time preference
            self._simulate_estdata()
            self._simulate_purchase(seed)
            
        finally:
            # Restore original parameters and norms
            self.par_est = par_est_original
            self.norm1, self.norm2, self.norm3 = norm_original
            self._simulate_estdata()

 
    def _calc_type(self):
        """Calculate customer type probabilities based on utility functions."""
        # Compute pickup and deliver sums for each type
        U_exp_s = [self.U_exp_s1, self.U_exp_s2, self.U_exp_s3]
        U_exp_exp_s = [self.U_exp_exp_s1, self.U_exp_exp_s2, self.U_exp_exp_s3]
        scale_params = [self.par_est[5], self.par_est[6], self.par_est[7]]
        
        pickup_sums = []
        deliver_prods = []
        
        for U_s, U_exp_s, scale in zip(U_exp_s, U_exp_exp_s, scale_params):
            pickup = np.einsum('ijk,k->ij', self.pickup3d, U_s)
            pickup_sum = np.nansum(pickup, axis=1) / scale
            pickup_sums.append(pickup_sum)
            
            deliver = np.einsum('ijk,k->ij', self.deliver3d, U_exp_s)
            deliver_prod = np.nanprod(deliver, axis=1)
            deliver_prod[deliver_prod == 0] = 1e-323
            deliver_prods.append(deliver_prod)
        
        # Split customers by station availability
        index_has_station = self.customer_data['n_station'] > 0
        index_no_station = ~index_has_station
        
        utility_hx = [self.utility_hx1, self.utility_hx2, self.utility_hx3]
        utility_sx = [self.utility_sx1, self.utility_sx2, self.utility_sx3]
        
        # Compute results for each type
        result_cus_A = []
        result_cus_B = []
        
        for i, (hx, sx, pickup_sum, deliver_prod) in enumerate(zip(
            utility_hx, utility_sx, pickup_sums, deliver_prods
        )):
            utility_sx_A = sx[index_has_station]
            utility_sx_B = sx[index_no_station]
            
            # Customers with stations
            denom_A = 1 + np.exp(hx) + np.exp(utility_sx_A)
            exp_term_A = (self.customer_data_A['n_home'].values * hx + 
                         self.customer_data_A['n_station'].values * utility_sx_A + 
                         pickup_sum)
            result_A = (denom_A ** (-self.n_day)) * np.exp(exp_term_A) / deliver_prod
            result_cus_A.append(result_A)
            
            # Customers without stations
            denom_B = 1 + np.exp(hx) + np.exp(utility_sx_B)
            exp_term_B = self.customer_data_B['n_home'].values * hx
            result_B = (denom_B ** (-self.n_day)) * np.exp(exp_term_B)
            result_cus_B.append(result_B)
        
        # Store results in customer_data
        for i in range(3):
            col_name = f'result_cus_{i+1}'
            self.customer_data.loc[index_has_station, col_name] = result_cus_A[i]
            self.customer_data.loc[index_no_station, col_name] = result_cus_B[i]
        
        # Calculate type probabilities
        result_sum = (self.customer_data['result_cus_1'] + 
                     self.customer_data['result_cus_2'] + 
                     self.customer_data['result_cus_3'])
        
        for i in range(1, 4):
            self.customer_data[f'p_type{i}'] = self.customer_data[f'result_cus_{i}'] / result_sum
        
        # Assign most likely type
        type_probs = self.customer_data[['p_type1', 'p_type2', 'p_type3']].values
        self.customer_data['type'] = np.argmax(type_probs, axis=1) + 1
        
        # Create binary type indicators
        for i in range(1, 4):
            self.customer_data[f'type{i}'] = (self.customer_data['type'] == i).astype(int)
        
        # Calculate type proportions
        n_total = self.customer_data.shape[0]
        return [self.customer_data[f'type{i}'].sum() / n_total for i in range(1, 4)]
        
    def _ext_type(self, seed):
        """
        Train LogisticRegression model to predict customer types.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        
        Returns
        -------
        model : LogisticRegression
            Trained multinomial logistic regression model.
        """
        # Split data: 80% training, 20% test
        df_train = self.customer_data.sample(frac=0.8, random_state=seed)
        df_test = self.customer_data.drop(df_train.index)
        
        # Feature columns
        feature_cols = ['month_order_amount', 'mean_item_quantity', 
                       'mean_max_item_pay', 'mean_avg_item_pay']
        
        # Prepare training data
        x_train = df_train[feature_cols]
        y_train = df_train['type']
        
        # Oversample to balance classes
        ros = RandomOverSampler(random_state=seed)
        x_over, y_over = ros.fit_resample(x_train, y_train)
        
        # Train model
        model = LogisticRegression(
            multi_class='multinomial',
            random_state=seed,
            solver='lbfgs',
            max_iter=1000,
            penalty='l2',
            C=1
        )
        model.fit(x_over, y_over)
        
        # Evaluate on test and training sets
        def evaluate_predictions(df, dataset_name):
            """Evaluate model predictions and print accuracy."""
            df['predict_type'] = model.predict(df[feature_cols])
            df['correct_type'] = (df['predict_type'] == df['type']).astype(int)
            accuracy = df['correct_type'].sum() / len(df)
            return df
        
        df_test = evaluate_predictions(df_test, 'test set')
        df_train = evaluate_predictions(df_train, 'training set')
        
        # Calculate type probabilities
        def calc_type_probs(df, col_name, name):
            """Calculate and print type probabilities."""
            probs = np.array([(df[col_name] == i).sum() / len(df) for i in range(1, 4)])
            return probs
        
        type_prob_pred = calc_type_probs(df_test, 'predict_type', 'Predicted')
        type_prob_baye = calc_type_probs(df_test, 'type', 'Bayesian calculated')
        
        # Confusion matrix
        cm = confusion_matrix(df_test['type'], df_test['predict_type'])
        
        return model


    def _read_data_city(self):
        """Load customer location data and station coordinates from simulated data."""
        # Load customer_info_loc.csv
        unified_file_path = join(self.args.projdir, 'synthetic_data', 'customer_info_loc.csv')
        
        self.customer_info_loc = pd.read_csv(unified_file_path, sep=',', header=0)
        
        # Load perturbed station coordinates
        coord_file_path = join(self.args.projdir, 'synthetic_data', 'station_coordinates_perturbed.csv')
        coord_df = pd.read_csv(coord_file_path)
        self.lnggrid_cursta_set = coord_df['lng_grid'].tolist()
        self.latgrid_cursta_set = coord_df['lat_grid'].tolist()
        self.station_num = len(coord_df)
        
        # Parse person_id to extract grid coordinates
        self.customer_info_loc[['lng_grid', 'lat_grid', 'id_val']] = (
            self.customer_info_loc['person_id'].str.split('_', expand=True)
        )
        self.customer_info_loc = self.customer_info_loc.replace(np.nan, 9999999, regex=True)
        self.customer_info_loc['lng_grid'] = self.customer_info_loc['lng_grid'].astype(int)
        self.customer_info_loc['lat_grid'] = self.customer_info_loc['lat_grid'].astype(int)
        
        # Calculate grid location and convert to radians
        self.customer_info_loc['grid_loc'] = (
            100000 * self.customer_info_loc['lng_grid'] + self.customer_info_loc['lat_grid']
        )
        self.customer_info_loc['lat'] = np.radians(41.19 + 0.0009 * self.customer_info_loc['lat_grid'])
        self.customer_info_loc['lng'] = np.radians(122.43 + 0.0012 * self.customer_info_loc['lng_grid'])
        
        # Predict customer types using logistic regression model
        feature_cols = ['month_order_amount', 'mean_item_quantity', 
                       'mean_max_item_pay', 'mean_avg_item_pay']
        X_features = self.customer_info_loc[feature_cols]
        
        self.customer_info_loc['predict_type'] = pl_model.predict(X_features)
        proba = pl_model.predict_proba(X_features)
        self.customer_info_loc['p_type1'] = proba[:, 0]
        self.customer_info_loc['p_type2'] = proba[:, 1]
        self.customer_info_loc['p_type3'] = proba[:, 2]
        
        # Calculate type probabilities
        n_total = len(self.customer_info_loc)
        self.type_prob = np.array([
            (self.customer_info_loc['predict_type'] == i).sum() / n_total 
            for i in range(1, 4)
        ])
        print('Predicted type percentage in extrapolated full set')
        print(self.type_prob)


    def _cusgrid_store(self):
        """
        Pre-process customer grid data by type for location counterfactual analysis.
        Groups customers by grid location and type, then calculates grid coordinates.
        """
        # Group customers by type and grid location
        cusgrid_dfs = []
        for type_id in range(1, 4):
            type_df = (
                self.customer_info_loc[self.customer_info_loc['predict_type'] == type_id]
                .groupby(['grid_loc'])
                .size()
                .reset_index(name=f'cus_type{type_id}')
            )
            cusgrid_dfs.append(type_df)
        
        # Merge all type dataframes
        self.cusgrid_type = cusgrid_dfs[0]
        for df in cusgrid_dfs[1:]:
            self.cusgrid_type = self.cusgrid_type.merge(df, how='outer', on='grid_loc')
        self.cusgrid_type = self.cusgrid_type.fillna(0)
        
        # Calculate latitude and longitude from grid_loc
        lat_grid = self.cusgrid_type['grid_loc'] % 100000
        lng_grid = np.floor(self.cusgrid_type['grid_loc'] / 100000)
        self.cusgrid_type['lat_store'] = np.radians(41.19 + 0.0009 * lat_grid)
        self.cusgrid_type['lng_store'] = np.radians(122.43 + 0.0012 * lng_grid)
        
        # Calculate total grid count and customer count
        self.n_grid = self.cusgrid_type.shape[0]
        self.n_cus_tot = sum(
            self.cusgrid_type[f'cus_type{i}'].sum() for i in range(1, 4)
        )
        return self.cusgrid_type  
    
    def _location_store(self):
        """
        Pre-compute distance matrix between all grid locations using Haversine formula.
        This speeds up the location counterfactual algorithm by storing distances.
        """
        EARTH_RADIUS_KM = 6373  # Earth radius in kilometers
        
        self.dis_store = np.zeros([self.n_grid, self.n_grid])
        print('Initializing distance matrix...')
        print(f'Computing distance matrix. Expected runtime: > 10 minutes...')
        
        lat_store = self.cusgrid_type['lat_store'].values
        lng_store = self.cusgrid_type['lng_store'].values
        
        for i in range(self.n_grid):
            # Calculate latitude and longitude differences
            dlat = lat_store[i] - lat_store
            dlng = lng_store[i] - lng_store
            
            # Haversine formula: a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlng/2)
            a = (
                np.sin(dlat / 2) ** 2 +
                np.cos(lat_store) * np.cos(lat_store[i]) * np.sin(dlng / 2) ** 2
            )
            
            # Distance = 2 * atan2(√a, √(1-a)) * R
            # Using atan2 for numerical stability
            self.dis_store[i] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * EARTH_RADIUS_KM
        
        return self.dis_store
    
    def _get_station_coordinates(self, best):
        """Get station coordinates based on best flag (0=current, 1=optimal)."""
        if best == 0:
            return self.latgrid_cursta_set, self.lnggrid_cursta_set
        else:  # best == 1
            latgrid = self.grid_best_set % 100000
            lnggrid = np.floor(self.grid_best_set / 100000)
            return latgrid, lnggrid
    
    def _compute_beta_arrays(self):
        """Compute beta coefficient arrays for three customer types."""
        # Base pattern: [0, 0] + [beta1]*6 + [beta2]*4 + [0, 0] = 12 elements
        base_pattern_type1 = [0, 0] + [self.par_est[0]] * 6 + [self.par_est[1]] * 4 + [0, 0]
        base_pattern_type2 = [0, 0] + [self.par_est[10]] * 6 + [self.par_est[11]] * 4 + [0, 0]
        base_pattern_type3 = [0, 0] + [self.par_est[15]] * 6 + [self.par_est[16]] * 4 + [0, 0]
        
        beta_x0 = [np.array(base_pattern_type1), np.array(base_pattern_type2), np.array(base_pattern_type3)]
        beta_x = [np.array(base_pattern_type1 * 3), np.array(base_pattern_type2 * 3), np.array(base_pattern_type3 * 3)]
        
        return beta_x0, beta_x
    
    def _compute_utilities(self, beta_x0, beta_x, dis_min_now, eff):
        """Compute utility values for all customer types."""
        EULER_GAMMA = 0.5772  # Euler-Mascheroni constant
        
        # Parameter indices for each type
        type_params = [
            {'time_coef': 2, 'scale': 5, 'intercept_h': 20, 'intercept_s': 4, 'dist_coef': 3, 'const': self.const1},
            {'time_coef': 12, 'scale': 6, 'intercept_h': 21, 'intercept_s': 14, 'dist_coef': 13, 'const': self.const2},
            {'time_coef': 17, 'scale': 7, 'intercept_h': 22, 'intercept_s': 19, 'dist_coef': 18, 'const': self.const3}
        ]
        
        # Compute expected utilities for store visits
        U_exp_s = []
        U_exp_exp_s = []
        for i, params in enumerate(type_params):
            u_exp = beta_x[i] + self.par_est[params['time_coef']] * self.time_slot
            U_exp_s.append(u_exp)
            U_exp_exp_s.append(np.exp(u_exp / self.par_est[params['scale']]))
        
        # Compute home utility (constant across grids)
        utility_hx = []
        for i, params in enumerate(type_params):
            u_h = (
                self.par_est[params['intercept_h']] +
                sum(self.prob_slot_h * (beta_x0[i] + self.par_est[params['time_coef']] * self.time_slot0)) +
                EULER_GAMMA * self.par_est[params['scale']]
            )
            utility_hx.append(u_h)
        
        # Compute store utility (varies by grid)
        utility_sx = []
        for i, params in enumerate(type_params):
            log_V_U = np.log(np.matmul(self.V_slot, U_exp_exp_s[i]))
            u_s = (
                self.par_est[params['intercept_s']] +
                self.par_est[params['dist_coef']] * dis_min_now +
                sum(self.prob_slot_s * log_V_U) * self.par_est[params['scale']] +
                params['const'] -
                10000 * eff * (dis_min_now > 1).astype(int)
            )
            utility_sx.append(u_s)
        
        return utility_hx, utility_sx
    
    def _compute_min_distances(self, latgrid_cursta_set, lnggrid_cursta_set):
        """Compute minimum distances from each grid to nearest station using Haversine formula."""
        EARTH_RADIUS_KM = 6373
        dis_min_now = np.full(self.n_grid, 1000.0)
        
        lat_store = self.cusgrid_type['lat_store'].values
        lng_store = self.cusgrid_type['lng_store'].values
        
        for i in range(self.station_num):
            lat = np.radians(41.19 + 0.0009 * latgrid_cursta_set[i])
            lng = np.radians(122.43 + 0.0012 * lnggrid_cursta_set[i])
            
            # Haversine formula
            dlat = lat - lat_store
            dlng = lng - lng_store
            a = (
                np.sin(dlat / 2) ** 2 +
                np.cos(lat_store) * np.cos(lat) * np.sin(dlng / 2) ** 2
            )
            dis_grids = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * EARTH_RADIUS_KM
            dis_min_now = np.minimum(dis_grids, dis_min_now)
        
        return dis_min_now
    
    def _compute_purchase_probability(self, utility_hx, utility_sx):
        """Compute purchase probability for each grid based on utilities."""
        prob_sum_grid = (
            self.cusgrid_type['cus_type1'] * (1 - 1 / (1 + np.exp(utility_hx[0]) + np.exp(utility_sx[0]))) +
            self.cusgrid_type['cus_type2'] * (1 - 1 / (1 + np.exp(utility_hx[1]) + np.exp(utility_sx[1]))) +
            self.cusgrid_type['cus_type3'] * (1 - 1 / (1 + np.exp(utility_hx[2]) + np.exp(utility_sx[2])))
        )
        return sum(prob_sum_grid) / self.n_cus_tot
    
    def _calloc_prob(self, best, eff):
        """
        Calculate purchase probability with given station locations.
        
        Parameters:
        -----------
        best : int
            0 for current locations, 1 for optimal locations
        eff : float
            Efficiency parameter affecting utility calculation
        
        Returns:
        --------
        float
            Overall purchase probability
        """
        # Get station coordinates
        latgrid_cursta_set, lnggrid_cursta_set = self._get_station_coordinates(best)
        
        # Compute minimum distances from each grid to nearest station
        self.dis_min_now = self._compute_min_distances(latgrid_cursta_set, lnggrid_cursta_set)
        
        # Compute beta arrays
        beta_x0, beta_x = self._compute_beta_arrays()
        
        # Compute utilities
        utility_hx, utility_sx = self._compute_utilities(beta_x0, beta_x, self.dis_min_now, eff)
        
        # Compute purchase probability
        self.prob_cur = self._compute_purchase_probability(utility_hx, utility_sx)
        return self.prob_cur
    
    def _calloc_prob_welfare(self, best, eff, q, n_sim, seed=0):
        """
        Calculate purchase probability and consumer welfare with given station locations.
        
        Parameters:
        -----------
        best : int
            0 for current locations, 1 for optimal locations
        eff : float
            Efficiency parameter affecting utility calculation
        q : float
            Quantile threshold for welfare calculation (if < 1, only consider bottom q quantile)
        n_sim : int
            Number of simulation iterations for welfare calculation
        seed : int
            Random seed for welfare simulation (default: 0)
        
        Returns:
        --------
        float
            Consumer welfare (scaled by 0.0004)
        """
        # Get station coordinates
        latgrid_cursta_set, lnggrid_cursta_set = self._get_station_coordinates(best)
        
        # Compute minimum distances from each grid to nearest station
        self.dis_min_now = self._compute_min_distances(latgrid_cursta_set, lnggrid_cursta_set)
        
        # Compute beta arrays
        beta_x0, beta_x = self._compute_beta_arrays()
        
        # Compute utilities
        utility_hx, utility_sx = self._compute_utilities(beta_x0, beta_x, self.dis_min_now, eff)
        
        # Simulate consumer welfare
        n_sample = int(self.n_grid)
        wg = np.zeros(int(self.n_cus_tot), dtype=np.float64)
        count = 0
        
        for i in range(n_sample):
            # Get customer counts for each type in this grid
            n_type1 = int(self.cusgrid_type['cus_type1'][i])
            n_type2 = int(self.cusgrid_type['cus_type2'][i])
            n_type3 = int(self.cusgrid_type['cus_type3'][i])
            n_total = n_type1 + n_type2 + n_type3
            
            # Initialize welfare arrays for this grid
            wg1 = np.zeros(n_type1, dtype=np.float64)
            wg2 = np.zeros(n_type2, dtype=np.float64)
            wg3 = np.zeros(n_type3, dtype=np.float64)
            
            # Monte Carlo simulation
            for j in range(n_sim):
                np.random.seed(seed + 100 * i + j)
                
                # Generate random utility shocks (Gumbel distribution)
                rand_s = np.random.gumbel(0, 1, n_total)  # Store visit shock
                rand_h = np.random.gumbel(0, 1, n_total)  # Home delivery shock
                rand_np = np.random.gumbel(0, 1, n_total)  # No purchase shock
                
                # Split shocks by customer type
                idx1_end = n_type1
                idx2_end = n_type1 + n_type2
                
                # Type 1 customers
                utility_np1 = rand_np[:idx1_end]
                utility_s1 = utility_sx[0][i] + rand_s[:idx1_end]
                utility_h1 = utility_hx[0] + rand_h[:idx1_end]
                
                # Type 2 customers
                utility_np2 = rand_np[idx1_end:idx2_end]
                utility_s2 = utility_sx[1][i] + rand_s[idx1_end:idx2_end]
                utility_h2 = utility_hx[1] + rand_h[idx1_end:idx2_end]
                
                # Type 3 customers
                utility_np3 = rand_np[idx2_end:]
                utility_s3 = utility_sx[2][i] + rand_s[idx2_end:]
                utility_h3 = utility_hx[2] + rand_h[idx2_end:]
                
                # Calculate welfare: max(store_utility - no_purchase, home_utility - no_purchase, 0)
                wg1_sim = np.maximum(
                    np.maximum(utility_s1 - utility_np1, utility_h1 - utility_np1),
                    0
                )
                wg2_sim = np.maximum(
                    np.maximum(utility_s2 - utility_np2, utility_h2 - utility_np2),
                    0
                )
                wg3_sim = np.maximum(
                    np.maximum(utility_s3 - utility_np3, utility_h3 - utility_np3),
                    0
                )
                
                wg1 += wg1_sim
                wg2 += wg2_sim
                wg3 += wg3_sim
            
            # Average over simulations
            wg1 /= n_sim
            wg2 /= n_sim
            wg3 /= n_sim
            
            # Store welfare values
            wg[count:count + n_type1] = wg1
            wg[count + n_type1:count + n_type1 + n_type2] = wg2
            wg[count + n_type1 + n_type2:count + n_total] = wg3
            count += n_total
        
        # Apply quantile filter if specified
        if q < 1:
            qq = np.quantile(wg, q)
            wg = wg[wg <= qq]
        
        # Calculate average welfare and scale
        self.welfare_location = np.mean(wg) / 0.0004
        
        # Also compute purchase probability (for consistency)
        self.prob_cur = self._compute_purchase_probability(utility_hx, utility_sx)
        
        return self.welfare_location
 
 
    
    def _location_counterfactual(self, eff):
        """
        Find optimal station locations by iteratively selecting the best grid location
        for each station to maximize purchase probability.
        
        Parameters:
        -----------
        eff : float
            parameter affecting stations' effective business distance
        
        Returns:
        --------
        tuple
            (grid_best_set, prob_best_set) - optimal grid locations and their probabilities
        """
        # Initialize
        dis_min = np.full(self.n_grid, 1000.0)
        self.grid_best_set = np.zeros(self.station_num)
        self.prob_best_set = np.zeros(self.station_num)
        
        # Compute beta arrays (home utility is constant, computed once)
        beta_x0, beta_x = self._compute_beta_arrays()
        utility_hx, _ = self._compute_utilities(beta_x0, beta_x, dis_min, eff)
        
        # Iteratively find best location for each station
        for station_idx in range(self.station_num):
            print(f'Running location counterfactual station {station_idx + 1}...')
            dis_min_pre = dis_min.copy()
            prob_best = 0
            grid_best = None
            
            # Try each grid location as a potential station site
            for grid_idx in range(self.n_grid):
                # Update minimum distance considering this grid as a station location
                dis_min_now = np.minimum(self.dis_store[grid_idx], dis_min_pre)
                
                # Compute utilities for all grids with this station location
                _, utility_sx = self._compute_utilities(beta_x0, beta_x, dis_min_now, eff)
                
                # Compute overall purchase probability
                prob_now = self._compute_purchase_probability(utility_hx, utility_sx)
                
                # Update best if this location is better
                if prob_now > prob_best:
                    prob_best = prob_now
                    grid_best = self.cusgrid_type['grid_loc'].iloc[grid_idx]
                    dis_min = dis_min_now.copy()
            
            # Store results for this station
            self.grid_best_set[station_idx] = grid_best
            self.prob_best_set[station_idx] = prob_best
                    
        # Save results to cache
        cache_dir = join(self.args.projdir, 'Results', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(join(cache_dir, 'grid_best_set_global_revision.pickle'), 'wb') as f:
            pickle.dump(self.grid_best_set, f)
        with open(join(cache_dir, 'prob_best_set_global_revision.pickle'), 'wb') as f:
            pickle.dump(self.prob_best_set, f)
        
        return self.grid_best_set, self.prob_best_set


    def _read_gaussian(self):
        """
        Load Gaussian mixture model parameters from simulated data.
        
        Returns:
        --------
        tuple
            (mu1, mu2, sigma_est1, sigma_est2, p_dis1, p_dis2) - Gaussian parameters
        """
        gaussian_params_path = join(self.args.projdir, 'synthetic_data', 'gaussian_params.csv')
        
        if not os.path.exists(gaussian_params_path):
            raise FileNotFoundError(
                f'Gaussian parameters file not found at {gaussian_params_path}. '
                'Please run create_simulated_customer_data.py to generate it.'
            )
        
        gaussian_df = pd.read_csv(gaussian_params_path, sep=',', header=0)
        params_dict = dict(zip(gaussian_df['parameter'], gaussian_df['value']))
        
        # Load parameters
        param_names = ['mu1', 'mu2', 'sigma_est1', 'sigma_est2', 'p_dis1', 'p_dis2']
        for param_name in param_names:
            setattr(self, param_name, params_dict[param_name])
        
        
        return self.mu1, self.mu2, self.sigma_est1, self.sigma_est2, self.p_dis1, self.p_dis2

    def _clear_gpu_cache(self):
        """
        Clear GPU cache by deleting cached tensors and emptying CUDA cache.
        This should be called before bootstrap or other operations that need fresh GPU memory.
        """
        # List of tensor attributes to clear
        tensor_attrs = [
            'time_slot_t', 'time_slot0_t', 'prob_slot_h_t', 'prob_slot_s_t',
            'V_slot_t', 'n_day_t', 'closest_distance_t', 'n_home_A_t',
            'n_station_A_t', 'n_home_B_t', 'index_has_station_t',
            'pickup3d_t', 'deliver3d_t'
        ]
        
        # Delete cached tensors
        for attr in tensor_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Empty CUDA cache
        torch.cuda.empty_cache()

    def _time_counterfactual(self):
        """
        Optimize delivery window allocation across customer types using linear programming.
        Maximizes overall purchase probability by redistributing delivery time slots.
        
        Returns:
        --------
        tuple
            (time_ctr_prob, type1_deliver, type2_deliver, type3_deliver)
            - time_ctr_prob: Average purchase probability after optimization
            - type1_deliver, type2_deliver, type3_deliver: Delivery allocations for each type
        """
        EULER_GAMMA = 0.5772
        N_TIME_SLOTS = 2
        
        print('Time counterfactual running...')
        
        # Calculate customer type proportions
        n_cus = self.customer_info_loc.shape[0]
        type_portion = np.array([
            len(self.customer_info_loc[self.customer_info_loc.predict_type == i]) / n_cus
            for i in range(1, 4)
        ])
        
        # Delivery probability and expected time for each time slot
        prob_upd = np.array([self.p_dis1, self.p_dis2])
        time_slot_upd = np.array([self.mu1, self.mu2])
        
        # Calculate probability weights for each time slot based on Gaussian distributions
        # pb: probability weights for morning slot (mu1, sigma_est1)
        # pb_: probability weights for afternoon slot (mu2, sigma_est2)
        def compute_prob_weights(mu, sigma):
            """Compute probability weights: [early, middle, late] time segments."""
            pb_early = norm.cdf(10, mu, sigma) + norm.sf(20, mu, sigma)  # <=10 or >=20
            pb_late = norm.sf(16, mu, sigma) - norm.sf(20, mu, sigma)    # 16-20
            pb_middle = 1 - pb_early - pb_late                            # 10-16
            return np.array([pb_early, pb_middle, pb_late])
        
        pb = compute_prob_weights(self.mu1, self.sigma_est1)  # Morning slot
        pb_ = compute_prob_weights(self.mu2, self.sigma_est2)  # Afternoon slot
        
        # Parameter indices for each customer type
        type_params = [
            {'norm': self.norm1, 'beta_idx': [0, 1], 'time_coef': 2, 'intercept': 20, 'scale': 5},
            {'norm': self.norm2, 'beta_idx': [10, 11], 'time_coef': 12, 'intercept': 21, 'scale': 6},
            {'norm': self.norm3, 'beta_idx': [15, 16], 'time_coef': 17, 'intercept': 22, 'scale': 7}
        ]
        
        # Compute updated beta coefficients for each type and time slot
        beta_upd = []
        for params in type_params:
            beta_slot1 = pb[0] * params['norm'] + pb[1] * self.par_est[params['beta_idx'][0]] + pb[2] * self.par_est[params['beta_idx'][1]]
            beta_slot2 = pb_[0] * params['norm'] + pb_[1] * self.par_est[params['beta_idx'][0]] + pb_[2] * self.par_est[params['beta_idx'][1]]
            beta_upd.append(np.array([beta_slot1, beta_slot2]))
        
        # Compute utility for each type and time slot
        utility_hx_each = []
        for i, params in enumerate(type_params):
            utility = (
                self.par_est[params['intercept']] +
                beta_upd[i] +
                self.par_est[params['time_coef']] * time_slot_upd +
                EULER_GAMMA * self.par_est[params['scale']]
            )
            utility_hx_each.append(utility)
        
        # Compute purchase probability for each type (sigmoid of utility)
        self.prob_buy1 = 1 / (1 + np.exp(-utility_hx_each[0]))
        self.prob_buy2 = 1 / (1 + np.exp(-utility_hx_each[1]))
        self.prob_buy3 = 1 / (1 + np.exp(-utility_hx_each[2]))
        
        # Prepare linear programming problem
        # Objective: maximize sum of purchase probabilities weighted by allocations
        prob_buy = np.concatenate([self.prob_buy1, self.prob_buy2, self.prob_buy3])
        
        # Equality constraints:
        # 1-3: Sum of allocations for each type equals type proportion
        # 4: Sum of morning allocations equals morning delivery probability
        A_eq0 = np.zeros([4, 6])
        for i in range(N_TIME_SLOTS):
            A_eq0[0, i] = 1      # Type 1, slot i
            A_eq0[1, i + 2] = 1  # Type 2, slot i
            A_eq0[2, i + 4] = 1  # Type 3, slot i
        
        A_eq0[3, 0] = 1  # Type 1, morning
        A_eq0[3, 2] = 1  # Type 2, morning
        A_eq0[3, 4] = 1  # Type 3, morning
        
        b_eq0 = np.append(type_portion, prob_upd[0] / prob_upd.sum())
        
        # Solve linear programming problem (minimize negative = maximize)
        res = linprog(-prob_buy, A_eq=A_eq0, b_eq=b_eq0)
        print(res)
        
        # Extract results
        self.type1_deliver = res.x[:2]
        self.type2_deliver = res.x[2:4]
        self.type3_deliver = res.x[4:]
        self.time_ctr_prob = -res.fun
        
        # Create results table
        table_6_panel_A = pd.DataFrame(
            [
                self.prob_buy2.tolist() + self.type2_deliver.tolist(),
                self.prob_buy3.tolist() + self.type3_deliver.tolist(),
                self.prob_buy1.tolist() + self.type1_deliver.tolist()
            ],
            columns=[
                'Purchase probability - Morning delivery window (i=0)',
                'Purchase probability - Afternoon delivery window (i=1)',
                'Delivery window allocation - Morning delivery window (i=0)',
                'Delivery window allocation - Afternoon delivery window (i=1)'
            ],
            index=['Low type (k=LT)', 'Medium type (k=MT)', 'High type (k=HT)']
        )
        
        print('Table 6 Delivery Window Counterfactuals, Panel A: Parameters and results')
        print(table_6_panel_A.to_latex(index=False, float_format="%.3f"))
        
        return self.time_ctr_prob, self.type1_deliver, self.type2_deliver, self.type3_deliver

    def _time_counterfactual_welfare(self, q, seed=0):
        """
        Calculate consumer welfare for time counterfactual analysis.
        First optimizes delivery window allocation, then simulates welfare using Monte Carlo.
        
        Parameters:
        -----------
        q : float
            Quantile threshold for welfare calculation (if < 1, only consider bottom q quantile)
        seed : int
            Random seed for Monte Carlo simulation (default: 0)
        
        Returns:
        --------
        tuple
            (utility_average, utility_before, utility_calculate, utility_cal_after, utility_cal_before)
            - utility_average: Average welfare after optimization (scaled by price coefficient)
            - utility_before: Average welfare before optimization (scaled by price coefficient)
            - utility_calculate: Calculated utility for each type and time slot
            - utility_cal_after: Calculated welfare after optimization
            - utility_cal_before: Calculated welfare before optimization
        """
        PRICE_COEFFICIENT = 0.0004
        EULER_GAMMA = 0.5772
        N_TIME_SLOTS = 2
        N_SAMPLE = 10000
        
        print('Time counterfactual running...')
        
        # Calculate customer type proportions
        n_cus = self.customer_info_loc.shape[0]
        type_portion = np.array([
            len(self.customer_info_loc[self.customer_info_loc.predict_type == i]) / n_cus
            for i in range(1, 4)
        ])
        
        # Delivery probability and expected time for each time slot
        prob_upd = np.array([self.p_dis1, self.p_dis2])
        time_slot_upd = np.array([self.mu1, self.mu2])
        
        # Calculate probability weights for each time slot
        def compute_prob_weights(mu, sigma):
            """Compute probability weights: [early, middle, late] time segments."""
            pb_early = norm.cdf(10, mu, sigma) + norm.sf(20, mu, sigma)
            pb_late = norm.sf(16, mu, sigma) - norm.sf(20, mu, sigma)
            pb_middle = 1 - pb_early - pb_late
            return np.array([pb_early, pb_middle, pb_late])
        
        pb = compute_prob_weights(self.mu1, self.sigma_est1)
        pb_ = compute_prob_weights(self.mu2, self.sigma_est2)
        
        # Parameter indices for each customer type
        type_params = [
            {'norm': self.norm1, 'beta_idx': [0, 1], 'time_coef': 2, 'intercept': 20, 'scale': 5},
            {'norm': self.norm2, 'beta_idx': [10, 11], 'time_coef': 12, 'intercept': 21, 'scale': 6},
            {'norm': self.norm3, 'beta_idx': [15, 16], 'time_coef': 17, 'intercept': 22, 'scale': 7}
        ]
        
        # Compute updated beta coefficients and utilities
        beta_upd = []
        utility_hx_each = []
        
        for params in type_params:
            beta_slot1 = pb[0] * params['norm'] + pb[1] * self.par_est[params['beta_idx'][0]] + pb[2] * self.par_est[params['beta_idx'][1]]
            beta_slot2 = pb_[0] * params['norm'] + pb_[1] * self.par_est[params['beta_idx'][0]] + pb_[2] * self.par_est[params['beta_idx'][1]]
            beta_upd.append(np.array([beta_slot1, beta_slot2]))
            
            utility = (
                self.par_est[params['intercept']] +
                beta_upd[-1] +
                self.par_est[params['time_coef']] * time_slot_upd +
                EULER_GAMMA * self.par_est[params['scale']]
            )
            utility_hx_each.append(utility)
        
        # Compute purchase probabilities
        self.prob_buy1 = 1 / (1 + np.exp(-utility_hx_each[0]))
        self.prob_buy2 = 1 / (1 + np.exp(-utility_hx_each[1]))
        self.prob_buy3 = 1 / (1 + np.exp(-utility_hx_each[2]))
        
        # Solve linear programming problem
        prob_buy = np.concatenate([self.prob_buy1, self.prob_buy2, self.prob_buy3])
        
        A_eq0 = np.zeros([4, 6])
        for i in range(N_TIME_SLOTS):
            A_eq0[0, i] = 1
            A_eq0[1, i + 2] = 1
            A_eq0[2, i + 4] = 1
        
        A_eq0[3, 0] = 1
        A_eq0[3, 2] = 1
        A_eq0[3, 4] = 1
        
        b_eq0 = np.append(type_portion, prob_upd[0] / prob_upd.sum())
        
        res = linprog(-prob_buy, A_eq=A_eq0, b_eq=b_eq0)
        print(res)
        
        self.type1_deliver = res.x[:2]
        self.type2_deliver = res.x[2:4]
        self.type3_deliver = res.x[4:]
        self.time_ctr_prob = -res.fun
        
        # Monte Carlo simulation for welfare calculation
        np.random.seed(seed)
        rand_h = np.random.gumbel(0, 1, N_SAMPLE)  # Home delivery shock
        rand_np = np.random.gumbel(0, 1, N_SAMPLE)  # No purchase shock
        
        # Calculate welfare for each type and time slot
        # Welfare = max(utility_home - utility_no_purchase, 0)
        utility_welfare = []
        for type_idx in range(3):
            welfare_slots = []
            for slot_idx in range(N_TIME_SLOTS):
                welfare = np.maximum(
                    utility_hx_each[type_idx][slot_idx] + rand_h - rand_np,
                    0
                )
                
                # Apply quantile filter if specified
                if q < 1:
                    qq = np.quantile(welfare, q)
                    welfare = welfare[welfare <= qq]
                
                welfare_slots.append(np.mean(welfare))
            utility_welfare.append(np.array(welfare_slots))
        
        self.utility_h1_avg = utility_welfare[0]
        self.utility_h2_avg = utility_welfare[1]
        self.utility_h3_avg = utility_welfare[2]
        self.p_dis = prob_upd
        
        # Calculate average welfare (after and before optimization)
        type_deliver = [self.type1_deliver, self.type2_deliver, self.type3_deliver]
        utility_average = sum(
            np.sum(type_deliver[i] * utility_welfare[i]) for i in range(3)
        ) / PRICE_COEFFICIENT
        
        utility_before = sum(
            type_portion[i] * np.sum(self.p_dis * utility_welfare[i]) for i in range(3)
        ) / PRICE_COEFFICIENT
        
        # Calculate theoretical utility using log-sum-exp formula
        utility_calculate = np.array([
            np.log(np.exp(utility_hx_each[i]) + 1) for i in range(3)
        ])
        
        utility_cal_after = sum(
            np.sum(type_deliver[i] * utility_calculate[i]) for i in range(3)
        )
        
        utility_cal_before = sum(
            type_portion[i] * np.sum(self.p_dis * utility_calculate[i]) for i in range(3)
        )
        
        return utility_average, utility_before, utility_calculate, utility_cal_after, utility_cal_before


    def _compare_time_ctr(self):
        """
        Compare purchase probabilities before and after time counterfactual optimization.
        Calculates probabilities with and without stations, using both point estimates and Bayesian averages.
        
        Returns:
        --------
        tuple
            (prob_wsta_avg, prob_nsta_avg) - average probabilities with and without stations
        """
        EULER_GAMMA = 0.5772
        N_TIME_SLOTS = 2
        
        customer_info_loc_copy = self.customer_info_loc.copy(deep=True)
        prob_buy = [self.prob_buy1, self.prob_buy2, self.prob_buy3]
        type_deliver = [self.type1_deliver, self.type2_deliver, self.type3_deliver]
        
        # Calculate expected purchase probability in each time slot
        for slot_idx in range(N_TIME_SLOTS):
            slot_prob = sum(
                customer_info_loc_copy[f'p_type{type_idx + 1}'] * prob_buy[type_idx][slot_idx]
                for type_idx in range(3)
            )
            customer_info_loc_copy[f'slot{slot_idx}_arr_p'] = slot_prob
        
        # Calculate expected purchase probability under arrangement for each type
        for type_idx in range(3):
            type_prob = sum(
                type_deliver[type_idx][slot_idx] * customer_info_loc_copy[f'slot{slot_idx}_arr_p']
                for slot_idx in range(N_TIME_SLOTS)
            ) / sum(type_deliver[type_idx])
            customer_info_loc_copy[f'type{type_idx + 1}_arr_p'] = type_prob
        
        # Assign arrangement probability based on predicted type
        customer_info_loc_copy['arr_p'] = np.select(
            [customer_info_loc_copy['predict_type'] == i for i in range(1, 4)],
            [customer_info_loc_copy[f'type{i}_arr_p'] for i in range(1, 4)],
            default=0
        )
        
        # Print statistics
        exp_purchase_p = np.mean(customer_info_loc_copy['arr_p'])
        purchase_p = [
            np.mean(customer_info_loc_copy[customer_info_loc_copy['predict_type'] == i + 1][f'type{i + 1}_arr_p'])
            for i in range(3)
        ]

        # Calculate probability weights for each time slot
        def compute_prob_weights(mu, sigma):
            """Compute probability weights: [early, middle, late] time segments."""
            pb_early = norm.cdf(10, mu, sigma) + norm.sf(20, mu, sigma)
            pb_late = norm.sf(16, mu, sigma) - norm.sf(20, mu, sigma)
            pb_middle = 1 - pb_early - pb_late
            return np.array([pb_early, pb_middle, pb_late])
        
        pb = compute_prob_weights(self.mu1, self.sigma_est1)  # Morning slot
        pb_ = compute_prob_weights(self.mu2, self.sigma_est2)  # Afternoon slot
        
        # Compute beta arrays for store utility calculation
        beta_x0, beta_x = self._compute_beta_arrays()
        
        # Compute expected utilities for store visits
        type_params = [
            {'time_coef': 2, 'scale': 5},
            {'time_coef': 12, 'scale': 6},
            {'time_coef': 17, 'scale': 7}
        ]
        U_exp_exp_s = []
        for i, params in enumerate(type_params):
            u_exp = beta_x[i] + self.par_est[params['time_coef']] * self.time_slot
            U_exp_exp_s.append(np.exp(u_exp / self.par_est[params['scale']]))
        
        # Compute beta updates and home utilities for each time slot
        type_params_full = [
            {'norm': self.norm1, 'beta_idx': [0, 1], 'time_coef': 2, 'intercept': 20, 'scale': 5, 'dist_coef': 3, 'intercept_s': 4, 'const': self.const1},
            {'norm': self.norm2, 'beta_idx': [10, 11], 'time_coef': 12, 'intercept': 21, 'scale': 6, 'dist_coef': 13, 'intercept_s': 14, 'const': self.const2},
            {'norm': self.norm3, 'beta_idx': [15, 16], 'time_coef': 17, 'intercept': 22, 'scale': 7, 'dist_coef': 18, 'intercept_s': 19, 'const': self.const3}
        ]
        
        utility_t_hx = []  # [time_slot][type]
        for slot_idx, (mu, pb_weights) in enumerate([(self.mu1, pb), (self.mu2, pb_)]):
            utilities = []
            for params in type_params_full:
                beta_upd = (
                    pb_weights[0] * params['norm'] +
                    pb_weights[1] * self.par_est[params['beta_idx'][0]] +
                    pb_weights[2] * self.par_est[params['beta_idx'][1]]
                )
                utility_h = (
                    self.par_est[params['intercept']] +
                    beta_upd +
                    self.par_est[params['time_coef']] * mu +
                    EULER_GAMMA * self.par_est[params['scale']]
                )
                utilities.append(utility_h)
            utility_t_hx.append(utilities)
        
        # Compute store utilities (varies by customer location)
        utility_sx = []
        for i, params in enumerate(type_params_full):
            log_V_U = np.log(np.matmul(self.V_slot, U_exp_exp_s[i]))
            u_s = (
                self.par_est[params['intercept_s']] +
                self.par_est[params['dist_coef']] * customer_info_loc_copy['closest_distance'] +
                sum(self.prob_slot_s * log_V_U) * self.par_est[params['scale']] +
                params['const']
            )
            utility_sx.append(u_s)
        # Helper function to compute purchase probability
        def compute_purchase_prob(utility_h, utility_s=None):
            """Compute purchase probability: 1 - 1/(1 + exp(utility_h) + exp(utility_s))."""
            if utility_s is None:
                return 1 - 1 / (1 + np.exp(utility_h))
            else:
                return 1 - 1 / (1 + np.exp(utility_h) + np.exp(utility_s))
        
        # Calculate probabilities with station (using point estimates)
        prob_wsta = np.zeros(len(customer_info_loc_copy))
        prob_nsta = np.zeros(len(customer_info_loc_copy))
        
        p_dis = np.array([self.p_dis1, self.p_dis2])
        for slot_idx in range(N_TIME_SLOTS):
            for type_idx in range(3):
                mask = customer_info_loc_copy['predict_type'] == (type_idx + 1)
                utility_h_scalar = utility_t_hx[slot_idx][type_idx]  # Scalar
                utility_s_vector = utility_sx[type_idx]  # Vector (per customer)
                
                # Broadcast scalar utility_h to match vector length
                utility_h_vector = np.full(len(customer_info_loc_copy), utility_h_scalar)
                
                # Compute probabilities for this type and slot
                prob_wsta_type = compute_purchase_prob(utility_h_vector, utility_s_vector)
                prob_nsta_type = compute_purchase_prob(utility_h_vector)
                
                prob_wsta[mask] += p_dis[slot_idx] * prob_wsta_type[mask]
                prob_nsta[mask] += p_dis[slot_idx] * prob_nsta_type[mask]
        
        customer_info_loc_copy['prob_wsta'] = prob_wsta
        customer_info_loc_copy['prob_nsta'] = prob_nsta
        
        self.prob_wsta_avg = prob_wsta.mean()
        self.prob_nsta_avg = prob_nsta.mean()
        
        
        # Calculate probabilities using Bayesian averages (weighted by type probabilities)
        prob_wsta_baye = np.zeros(len(customer_info_loc_copy))
        prob_nsta_baye = np.zeros(len(customer_info_loc_copy))
        
        for slot_idx in range(N_TIME_SLOTS):
            for type_idx in range(3):
                type_prob = customer_info_loc_copy[f'p_type{type_idx + 1}']
                utility_h_scalar = utility_t_hx[slot_idx][type_idx]
                utility_s_vector = utility_sx[type_idx]
                
                # Broadcast scalar utility_h to match vector length
                utility_h_vector = np.full(len(customer_info_loc_copy), utility_h_scalar)
                
                prob_wsta_baye += (
                    p_dis[slot_idx] *
                    type_prob *
                    compute_purchase_prob(utility_h_vector, utility_s_vector)
                )
                prob_nsta_baye += (
                    p_dis[slot_idx] *
                    type_prob *
                    compute_purchase_prob(utility_h_vector)
                )
        
        customer_info_loc_copy['prob_wsta_baye'] = prob_wsta_baye
        customer_info_loc_copy['prob_nsta_baye'] = prob_nsta_baye
        
        self.prob_wsta_avg_baye = prob_wsta_baye.mean()
        self.prob_nsta_avg_baye = prob_nsta_baye.mean()
        
        return self.prob_wsta_avg, self.prob_nsta_avg



def _build_parameter_table(par, std=None):
    """
    Build parameter table DataFrame for all customer types.
    
    Parameters:
    -----------
    par : array-like
        Parameter estimates array
    std : array-like, optional
        Standard errors array (if provided, formats as strings with parentheses)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with parameter estimates organized by type
    """
    # Parameter index mapping for each type
    # Order: [daytime, afterwork, scale, waiting, distance, home_const, station_const, type_prob]
    type_mappings = [
        {'name': 'Type1', 'indices': [0, 1, 5, 2, 3, 20, 4, 8]},
        {'name': 'Type2', 'indices': [10, 11, 6, 12, 13, 21, 14, 9]},
        {'name': 'Type3', 'indices': [15, 16, 7, 17, 18, 22, 19, None]}  # Type3 prob = 1 - Type1 - Type2
    ]
    
    # Parameter labels
    param_labels = [
        'Daytime receiving value',
        'After-work receiving value',
        'Scale parameter (uncertainty)',
        'Package waiting sensitivity',
        'Station distance sensitivity',
        'Home constant',
        'Station constant',
        'Type probability'
    ]
    
    # Extract values for each type
    data = {'Estimates': param_labels}
    
    for type_map in type_mappings:
        type_values = []
        for i, idx in enumerate(type_map['indices']):
            if idx is None:
                # Type3 probability: calculated value (1 - Type1 - Type2)
                if std is None:
                    type_values.append(f'{1 - par[8] - par[9]:.3f}')
                else:
                    type_values.append('--')
            else:
                # Single index
                if std is None:
                    type_values.append(f'{par[idx]:.3f}')
                else:
                    type_values.append(f'({std[idx]:.3f})')
        
        data[type_map['name']] = type_values
    
    df = pd.DataFrame(data)
    
    return df

def location_figures(obj_olm):
    """
    Generate location-related figures for the paper.
    
    Creates the following figures:
    - Figure 4(a): Population distribution and original station locations
    - Figure 4(b): Improved value of logistic flexibility (re-arranged vs original)
    - Figure 4(c): Purchase rate vs number of stations
    - Figure 4(d): Run fewer locations (first 25 re-arranged locations)
    - Global population density map
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # Constants
    FIG_SIZE = (4.5, 3.5)
    FONT_SIZE = 12
    GRID_BASE = 100000
    
    # Coordinate conversion parameters
    LNG_BASE = 122.43
    LNG_SCALE = 0.0012
    LAT_BASE = 41.19
    LAT_SCALE = 0.0009
    
    # View limits
    VIEW_LIMITS_ZOOM = {'xlim': [600, 1100], 'ylim': [400, 1000]}
    VIEW_LIMITS_GLOBAL = {'xlim': [400, 1300], 'ylim': [200, 1200]}
    
    # Scatter plot styles
    SCATTER_STYLE_DEFAULT = {'linewidths': 1, 'edgecolor': 'w', 's': 60}
    SCATTER_STYLE_ORIGINAL = {'linewidths': 1, 'edgecolor': 'w', 's': 30}
    
    # Helper function: Filter and prepare customer data
    def _prepare_customer_data(customer_info_loc, lng_range=(400, 1300), lat_range=(200, 1200)):
        """Filter customer data by grid coordinates and add lat/lng in radians."""
        mask = ((customer_info_loc['lng_grid'] >= lng_range[0]) & 
                (customer_info_loc['lng_grid'] <= lng_range[1]) &
                (customer_info_loc['lat_grid'] >= lat_range[0]) & 
                (customer_info_loc['lat_grid'] <= lat_range[1]))
        customer_select = customer_info_loc[mask].copy()
        customer_select['lng'] = np.radians(LNG_BASE + LNG_SCALE * customer_select['lng_grid'])
        customer_select['lat'] = np.radians(LAT_BASE + LAT_SCALE * customer_select['lat_grid'])
        return customer_select
    
    # Helper function: Compute best station grid coordinates
    def _compute_best_station_coords(grid_best_set):
        """Extract lat/lng grid coordinates from encoded grid_best_set."""
        latgrid_best = grid_best_set % GRID_BASE
        lnggrid_best = np.floor(grid_best_set / GRID_BASE).astype(int)
        lng_best = np.radians(LNG_BASE + LNG_SCALE * lnggrid_best)
        lat_best = np.radians(LAT_BASE + LAT_SCALE * latgrid_best)
        return latgrid_best, lnggrid_best, lat_best, lng_best
    
    # Helper function: Draw population density background
    def _draw_population_density(axes, customer_select, fig):
        """Draw 2D histogram of population density as background."""
        hist_data = axes.hist2d(
            customer_select['lng_grid'], 
            customer_select['lat_grid'], 
            bins=(100, 100), 
            cmap=plt.cm.Greys, 
            norm=LogNorm()
        )
        cbar = fig.colorbar(hist_data[3], ax=axes)
        cbar.set_label("Consumer population")
        return hist_data
    
    # Helper function: Draw station locations
    def _draw_stations(axes, lng_coords, lat_coords, color, label, marker, **scatter_kwargs):
        """Draw station locations as scatter points."""
        axes.scatter(lng_coords, lat_coords, color=color, label=label, 
                    marker=marker, **scatter_kwargs)
    
    # Helper function: Set up map axes
    def _setup_map_axes(axes, title, xlim, ylim, fontsize=FONT_SIZE):
        """Configure map axes with labels, title, and limits."""
        axes.set_xlabel('#Longitudinal grid', fontsize=fontsize)
        axes.set_ylabel('#Latitudinal grid', fontsize=fontsize)
        axes.set_title(title, fontsize=fontsize)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
    
    # Prepare data
    obj_olm._calloc_prob(0, 0)
    customer_select = _prepare_customer_data(obj_olm.customer_info_loc)
    
    # Compute best station coordinates
    (obj_olm.latgrid_best_set, obj_olm.lnggrid_best_set, 
     obj_olm.lat_best_set, obj_olm.lng_best_set) = _compute_best_station_coords(obj_olm.grid_best_set)
    
    kwargs = {'fontsize': FONT_SIZE}
    
    # Figure 4(c): Purchase rate curve
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE)
    station_counts = range(1, obj_olm.station_num + 1)
    axes.plot(station_counts, obj_olm.prob_best_set * 100, marker='o', 
             color='r', label="Re-arranged")
    axes.axhline(y=obj_olm.prob_cur * 100, ls='--', color='b', label="Original")
    axes.set_xlabel("Station quantity", **kwargs)
    axes.set_ylabel("Average purchase rate (%)", **kwargs)
    axes.legend(labels=("Re-arranged", "Original"), loc="lower right")
    axes.set_title('Figure 4(c) Purchase Rate vs Number of Stations', **kwargs)
    fig.tight_layout()
    fig.savefig('Figure_4c_Purchase Rate and Picked Locations.pdf')
    
    # Figure 4(b): Improved value map (re-arranged vs original)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE)
    _draw_population_density(axes, customer_select, fig)
    _draw_stations(axes, obj_olm.lnggrid_cursta_set, obj_olm.latgrid_cursta_set,
                  color='b', label="Original locations", marker='o', **SCATTER_STYLE_DEFAULT)
    _draw_stations(axes, obj_olm.lnggrid_best_set, obj_olm.latgrid_best_set,
                  color='r', label="Re-arranged locations", marker='^', **SCATTER_STYLE_DEFAULT)
    axes.legend(labels=("Original", "Re-arranged"), ncol=2)
    _setup_map_axes(axes, 'Figure 4(b) Map: Improve Value of Logistic Flexibility',
                    VIEW_LIMITS_ZOOM['xlim'], VIEW_LIMITS_ZOOM['ylim'])
    fig.tight_layout()
    fig.savefig('Figure_4b_Map_Improve Value of Logistic Flexibility.pdf')
    
    # Figure 4(d): Run fewer locations (first 25 stations)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE)
    _draw_population_density(axes, customer_select, fig)
    _draw_stations(axes, obj_olm.lnggrid_cursta_set, obj_olm.latgrid_cursta_set,
                  color='b', label="Original locations", marker='o', **SCATTER_STYLE_DEFAULT)
    _draw_stations(axes, obj_olm.lnggrid_best_set[:25], obj_olm.latgrid_best_set[:25],
                  color='r', label="Re-arranged locations", marker='^', **SCATTER_STYLE_DEFAULT)
    axes.legend(labels=("Original", "Re-arranged"), ncol=2)
    _setup_map_axes(axes, 'Figure 4(d) Map: Run Fewer Locations',
                    VIEW_LIMITS_ZOOM['xlim'], VIEW_LIMITS_ZOOM['ylim'])
    fig.tight_layout()
    fig.savefig('Figure_4d_Map_Run Fewer Locations.pdf')
    
    # Global population density map
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE)
    _draw_population_density(axes, customer_select, fig)
    _setup_map_axes(axes, '', VIEW_LIMITS_GLOBAL['xlim'], VIEW_LIMITS_GLOBAL['ylim'])
    fig.tight_layout()
    fig.savefig('location_population_global_map_rand_p.pdf')
    
    # Figure 4(a): Original locations map
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE)
    _draw_population_density(axes, customer_select, fig)
    _draw_stations(axes, obj_olm.lnggrid_cursta_set, obj_olm.latgrid_cursta_set,
                  color='b', label="Original locations", marker='o', **SCATTER_STYLE_ORIGINAL)
    axes.legend(labels=["Original"], loc="lower right")
    _setup_map_axes(axes, 'Figure 4(a) Map: Population Distribution and Station Locations',
                    VIEW_LIMITS_GLOBAL['xlim'], VIEW_LIMITS_GLOBAL['ylim'])
    fig.tight_layout()
    fig.savefig('Figure_4a_Map_Population and Stations in the City.pdf')

def print_par(par_final_3set):
    """
    Print parameter estimates table in LaTeX format.
    
    Parameters:
    -----------
    par_final_3set : array-like
        Final parameter estimates for all three customer types
    """
    df = _build_parameter_table(par_final_3set)
    print(df.to_latex(index=False, float_format='%.3f'))


def print_est(par_final_3set, std_final_3set):
    """
    Print parameter estimates with standard errors in LaTeX format.
    Prints three tables: estimates, standard errors, and combined.
    
    Parameters:
    -----------
    par_final_3set : array-like
        Final parameter estimates for all three customer types
    std_final_3set : array-like
        Standard errors for all parameters
    """
    # Build estimates table
    df_est = _build_parameter_table(par_final_3set)
    
    # Build standard errors table
    df_std = _build_parameter_table(par_final_3set, std_final_3set)
    df_std['Estimates'] = [''] * len(df_std)  # Empty labels for std table
    
    # Combine estimates and standard errors in alternating rows
    # Row order: [0, 1, 5, 2, 3, 7, 4, 6] - interleaves estimates and std errors
    row_order = [0, 1, 5, 2, 3, 7, 4, 6]
    df_combined = pd.concat([
        pd.concat([df_est.iloc[[i]], df_std.iloc[[i]]]) for i in row_order
    ], ignore_index=True)
    
    # Format numeric columns to 3 decimal places before printing
    for col in ['Type1', 'Type2', 'Type3']:
        if col in df_combined.columns:
            # Convert numeric values to 3 decimal places, keep strings as is
            def format_value(x):
                if isinstance(x, str):
                    # If it's already a formatted string (like "(33.588)" or "--"), keep it
                    if x == '--' or (x.startswith('(') and x.endswith(')')):
                        return x
                    # Try to parse and reformat if it's a numeric string
                    try:
                        return f'{float(x):.3f}'
                    except (ValueError, TypeError):
                        return x
                elif isinstance(x, (int, float, np.number)):
                    return f'{float(x):.3f}'
                else:
                    return x
            
            df_combined[col] = df_combined[col].apply(format_value)
    
    print(df_combined.to_latex(index=False))


def _generate_random_initial_point():
    """
    Generate a random initial parameter point with overflow-aware ranges.
    
    Returns:
    --------
    np.ndarray
        Random initial parameter vector of length 23
    """
    par = np.zeros(23)
    
    # Set probability parameters first (must sum to < 1)
    par[8] = np.random.uniform(0.01, 1)
    par[9] = np.random.uniform(0.01, 1 - par[8])
    
    # Parameter ranges based on overflow analysis
    param_ranges = {
        'scale': ([5, 6, 7], (0.1, 50)),  # Must be > 0.1 to avoid exp overflow
        'beta': ([0, 1, 10, 11, 15, 16], (-50, 50)),  # Beta coefficients
        'time': ([2, 12, 17], (-5, 5)),  # Time preference coefficients
        'distance': ([3, 13, 18], (-5, 5)),  # Distance coefficients
        'station_intercept': ([4, 14, 19], (-50, 50)),  # Station intercepts
        'home_intercept': ([20, 21, 22], (-50, 50))  # Home delivery intercepts
    }
    
    for param_type, (indices, (low, high)) in param_ranges.items():
        for idx in indices:
            par[idx] = np.random.uniform(low, high)
    
    return par


def _validate_initial_point(par):
    """
    Validate initial parameter point.
    
    Parameters:
    -----------
    par : np.ndarray
        Parameter vector to validate
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    # Check scale parameters are positive
    if any(par[i] <= 0.001 for i in [5, 6, 7]):
        return False
    
    # Check probability parameters are valid
    if (par[8] <= 0.001 or par[9] <= 0.001 or 
        par[8] + par[9] >= 1.0):
        return False
    
    return True


def _run_single_optimization(obj_olm, par_init_candidate):
    """
    Run optimization for a single initial point.
    
    Parameters:
    -----------
    obj_olm : OptimizeLatentModel
        Optimization object
    par_init_candidate : np.ndarray
        Initial parameter vector
    
    Returns:
    --------
    dict or None
        Result dictionary with keys: 'result', 'obj_value', 'par_init', 'par_opt'
        Returns None if optimization failed
    """
    # Set up optimization with candidate initial point
    obj_olm.experiment = 'vanilla'
    obj_olm.par_init = par_init_candidate.copy()
    obj_olm._optimization_setup()
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj_olm._optimization_run()
        
        # Check if optimization succeeded
        if (obj_olm.result.success or 
            (obj_olm.result.fun is not None and np.isfinite(obj_olm.result.fun))):
            return {
                'result': obj_olm.result,
                'obj_value': obj_olm.result.fun,
                'par_init': par_init_candidate.copy(),
                'par_opt': obj_olm.result.x.copy()
            }
    except Exception:
        pass
    
    return None


def find_optimal_multi_start(obj_olm, n_valid_points=200, max_attempts=100000, 
                             random_seed=None, use_gpu=False):
    """
    Multi-start optimization strategy: search among n_valid_points initial points and pick the best.
    
    This function performs a broad search across multiple initial points to find the global optimum.
    
    **Note**: This function illustrates one approach to finding optimal solutions. In practice,
    practitioners should iteratively refine solutions by perturbing previously identified optima
    and using them as new starting points across multiple rounds, rather than relying on a single
    round of random initialization. This enables more efficient local refinement around promising 
    regions.
    
    Parameters:
    -----------
    obj_olm : OptimizeLatentModel
        Optimization object
    n_valid_points : int
        Number of valid initial points to find and optimize (default: 200)
    max_attempts : int
        Maximum number of attempts to find valid points (default: 100000)
    random_seed : int, optional
        Random seed for reproducibility
    use_gpu : bool
        Whether to use GPU-accelerated loglikelihood function (default: False)
        
    Returns:
    --------
    tuple
        (best_result, best_par_init, all_results)
        - best_result: Best optimization result (scipy.optimize.OptimizeResult)
        - best_par_init: Best initial point
        - all_results: List of all optimization results
    """
    # Set up GPU if requested
    if use_gpu:
        if not torch.cuda.is_available():
            print('Warning: GPU requested but not available. Falling back to CPU.')
            use_gpu = False
        else:
            print(f'Using GPU acceleration (device: {torch_device})')
    
    # Store and temporarily replace loglikelihood method
    original_loglikelihood = obj_olm._loglikelihood
    if use_gpu:
        obj_olm._loglikelihood = obj_olm._loglikelihood_gpu
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Print header
    print('=' * 60)
    print('MULTI-START OPTIMIZATION: Broad Search')
    print('=' * 60)
    print(f'Searching for {n_valid_points} valid initial points (max attempts: {max_attempts})...')
    print(f'Using {"GPU" if use_gpu else "CPU"} for loglikelihood computation\n')
    
    # Initialize tracking variables
    all_results = []
    valid_count = 0
    attempt_count = 0
    best_obj_value = np.inf
    best_result = None
    best_par_init = None
    
    # Main optimization loop
    while valid_count < n_valid_points and attempt_count < max_attempts:
        attempt_count += 1
        
        # Generate and validate initial point
        par_init_candidate = _generate_random_initial_point()
        
        if not _validate_initial_point(par_init_candidate):
            if attempt_count % 1000 == 0:
                print(f'  [Attempt {attempt_count}] Still searching... '
                      f'{valid_count}/{n_valid_points} valid points found', flush=True)
            continue
        
        valid_count += 1
        print(f'  [Valid point {valid_count}/{n_valid_points}] '
              f'Found valid initial point (attempt {attempt_count}), '
              f'starting optimization...', flush=True)
        
        # Run optimization
        opt_result = _run_single_optimization(obj_olm, par_init_candidate)
        
        if opt_result is not None:
            obj_value = opt_result['obj_value']
            is_best = obj_value < best_obj_value
            
            if is_best:
                best_obj_value = obj_value
                best_result = opt_result['result']
                best_par_init = opt_result['par_init'].copy()
            
            all_results.append(opt_result)
            
            status = "✓ SUCCESS" if opt_result['result'].success else "✓ FINISHED"
            best_marker = " [NEW BEST!]" if is_best else ""
            print(f'    {status}: Objective = {obj_value:.6f}{best_marker}', flush=True)
        else:
            print('    ✗ FAILED: Optimization did not converge or returned invalid result', 
                  flush=True)
    
    # Print summary
    print(f'\n{"=" * 60}')
    print('Optimization Summary:')
    print(f'  Total attempts: {attempt_count}')
    print(f'  Valid initial points found: {valid_count}/{n_valid_points}')
    print(f'  Successful optimizations: {len(all_results)}')
    if all_results:
        print(f'  Best objective value: {best_obj_value:.6f}')
    print('=' * 60)
    
    # Restore original loglikelihood method
    obj_olm._loglikelihood = original_loglikelihood
    
    if not all_results:
        print('Warning: No successful optimizations!')
        return None, None, []
    
    # Sort results by objective value
    all_results.sort(key=lambda x: x['obj_value'])
    
    # Print top 10 results
    print('\nTop 10 results:')
    for idx, res in enumerate(all_results[:10]):
        marker = " ← BEST" if idx == 0 else ""
        print(f'  {idx+1:2d}. Objective value: {res["obj_value"]:.6f}{marker}')
    
    return best_result, best_par_init, all_results



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxiter_num', default=100000, type=int)
    parser.add_argument('--maxfev_num', default=100000, type=int)
    parser.add_argument('--xatol_num', default=0.001, type=float)
    parser.add_argument('--fatol_num', default=0.001, type=float) 
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--method', default='Nelder-Mead', type=str)
    parser.add_argument('--gpu_id', default=None, type=int, 
                       help='GPU device ID to use (e.g., 0, 1, 2, 3). If not specified, uses CUDA_DEVICE environment variable or defaults to 0.')
    
    return parser.parse_args()






if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Initialize project directory
    projdir = os.path.dirname(os.path.abspath(__file__))
    args.projdir = projdir
    os.chdir(projdir)
    
    # Configure GPU device
    if args.gpu_id is not None:
        set_gpu_device(args.gpu_id)
    else:
        set_gpu_device()
    
    # Verify GPU availability (required for this script)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required but not available. Please run on a machine with CUDA support.')
    
    if torch_device is None or torch_device.type != 'cuda':
        raise RuntimeError(f'GPU device not properly initialized. Expected CUDA device, got {torch_device}')
    
    print(f'Using GPU-accelerated optimization (device: {torch_device})')
    
    ############# Read data ##################
    print('Reading data...')
    
    # Define data directory
    data_dir = join(projdir, 'synthetic_data')
    
    # Load customer data
    customer_data = pd.read_pickle(join(data_dir, 'customer_data.pickle'))
    customer_info = pd.read_csv(join(data_dir, 'customer_all.csv'), sep=',', header=0)
    
    # Merge customer info into customer data
    info_cols = ["person_id", "month_order_amount", "mean_item_quantity", 
                 "mean_max_item_pay", "mean_avg_item_pay"]
    customer_data = customer_data.merge(customer_info[info_cols], how="inner", on="person_id")
    
    # Load perturbed probabilities
    prob_df = pd.read_csv(join(data_dir, 'prob_slot.csv'), sep=',', header=0)
    prob_slot_h = prob_df['prob_slot_h'].values  # Home delivery hour probabilities (perturbed)
    prob_slot_s = prob_df['prob_slot_s'].values  # Station delivery hour probabilities (perturbed)
    
    # Define time slots
    time_slot = np.concatenate((np.arange(8, 22), np.arange(32, 46), np.arange(56, 70)))
    time_slot0 = np.arange(8, 22)
    n_time_slot0 = len(time_slot0)  
    n_time_slot = len(time_slot)     
    V_slot = np.zeros((n_time_slot0, n_time_slot), dtype=int)
    for i in range(n_time_slot0):
        V_slot[i, i:] = 1
    
    # Set number of days
    n_day = 31
    
    ############# Initialization ##################
    # Initialize parameter arrays
    par_init = [0.01] * 23
    for idx in [3, 13, 18]: 
        par_init[idx] = -0.01
    par_est = par_init.copy()  
    
    # Create optimization model instance
    self = OptimizeLatentModel(
        args,
        prob_slot_h,
        prob_slot_s,
        V_slot,
        par_init,
        par_est,
        n_day
    )

    ############# Estimate structural model ##################
    print('Starting structural model estimation...')
    self.experiment = 'vanilla'
    self._optimization_setup()
    
    # Run single optimization with initial parameters
    self._optimization_run()
    
    # Check optimization results
    if self.result.success:
        self.par_est = self.result['x'].tolist()
        print_par(self.par_est)
        print(f'Finish estimation...')
    else:
        print(f'Warning: Optimization did not converge successfully.')
        print(f'Message: {self.result.message}')
        # Still use the result even if not fully converged
        self.par_est = self.result['x'].tolist()
        print_par(self.par_est)
        print(f'Using best result found despite convergence warning.')

    ############# Bootstrap standard error ##################
    print('\nStarting bootstrap standard error calculation...')
    n_bootstrap = 100
    self.experiment = "bootstrap"
    
    # Clear GPU cache before bootstrap
    self._clear_gpu_cache()
    print(f'Using GPU ({torch_device}) for bootstrap optimization', flush=True)
    
    # Run bootstrap samples
    for i in range(n_bootstrap):
        print(f'\nBootstrap sample {i+1}/{n_bootstrap}...', flush=True)
        self._optimization_setup(seed=i)
        self._optimization_run()
        print(f'Completed bootstrap sample {i+1}/{n_bootstrap}', flush=True)
    print('Finish bootstrap...')

    ############# Display structural estimation results ##################
    # Load bootstrap results and calculate standard errors
    bootstrap_results_path = join(projdir, 'Results', 'bootstrap', 'bootstrap_results_full.csv')
    bootstrap_results = pd.read_csv(bootstrap_results_path, sep=',', header=0).values
    bootstrap_params = bootstrap_results[:, :23].astype(float)  # Extract parameter estimates
    std_final_3set = np.std(bootstrap_params, axis=0)
    
    # Print estimation results table
    print('---Table 3: Estimation Results---')
    print_est(self.par_est, std_final_3set)
    
    ############# Counterfactual analysis setup ##################
    # Set random seed for reproducibility
    seed = 0

    GMV = 50000 # perturbed GMV
    price_coeff = 0.0004

    print('Setting up counterfactual analysis...')
    self._simulate_estdata()
    
    ## Table 4: Simulations from the Structural Model
    # Simulate purchase decisions with different customer type assumptions
    print('Simulating purchase behavior...')
    print('---Table 4: Simulation 1: Overall Performance---')
    self._simulate_purchase(seed)
    print('---Table 4: Simulation 2: Value of Flexibility---')
    self._simulate_scenario_purchase(seed)
    
    # Calculate customer types and build prediction model
    print('Calculating customer types and building prediction model...')
    self._calc_type()
    pl_model = self._ext_type(seed)
    
    # Load city-wide customer location data
    print('Loading city-wide customer location data...')
    self._read_data_city() 

    ############# Location counterfactual analysis ##################
    print('\nStarting location counterfactual analysis...')

    self._cusgrid_store()
    self.dis_store = self._location_store()
    
    
    #### Table 5, Panel A: Without Effective Distance ####
    print('\nComputing Table 5, Panel A (without effective distance). Expected runtime: > 1 hour...')
    self._location_counterfactual(0)
    
    # Calculate metrics for Panel A
    print(f'Computing consumer welfare. Expected runtime: > 1 hour...')
    after_purchase_rate = self._calloc_prob(1, 0)
    after_95_percent = self._calloc_prob_welfare(1, 0, 0.95, 1)
    after_75_percent = self._calloc_prob_welfare(1, 0, 0.75, 1)
    after_daily_cons = self._calloc_prob_welfare(1, 0, 1, 1)
    
    before_purchase_rate = self._calloc_prob(0, 0)
    before_95_percent = self._calloc_prob_welfare(0, 0, 0.95, 1)
    before_75_percent = self._calloc_prob_welfare(0, 0, 0.75, 1)
    before_daily_cons = self._calloc_prob_welfare(0, 0, 1, 1)
    
    # Create Table 5 Panel A 
    print("---Table 5: Performance of Location Counterfactuals, Panel A: Without Effective Distance---")
    table_5_panel_A_1 = pd.DataFrame([
        [before_purchase_rate, after_purchase_rate]
    ], columns=['Before', 'After'], index=['Purchase rate'])
    table_5_panel_A_1["Improvement"] = table_5_panel_A_1["After"] - table_5_panel_A_1["Before"]   
    print(table_5_panel_A_1.to_latex(index=True, float_format='%.3f'))

    table_5_panel_A_2 = pd.DataFrame([
        [GMV, GMV * after_purchase_rate / before_purchase_rate]
    ], columns=['Before', 'After'], index=['Annual GMV (million)'])
    table_5_panel_A_2["Improvement"] = table_5_panel_A_2["After"] - table_5_panel_A_2["Before"]   
    print(table_5_panel_A_2.to_latex(index=True, float_format='%.0f'))

    table_5_panel_A_3 = pd.DataFrame([
        [before_daily_cons, after_daily_cons],
        [before_95_percent, after_95_percent],
        [before_75_percent, after_75_percent]
    ], columns=['Before', 'After'], index=['Daily consumer welfare', '95% group daily welfare', '75% group daily welfare'])
    table_5_panel_A_3["Improvement"] = table_5_panel_A_3["After"] - table_5_panel_A_3["Before"]
    print(table_5_panel_A_3.to_latex(index=True, float_format='%.2f'))
    
    #### Table 5, Panel B: 1km Effective Distance ####
    print('\nComputing Table 5, Panel B (1km effective distance). Expected runtime: > 1 hour...')
    self._location_counterfactual(1)
    
    # Calculate metrics for Panel B
    print(f'Computing consumer welfare. Expected runtime: > 1 hour...')
    before_purchase_rate = self._calloc_prob(0, 1)
    before_95_percent = self._calloc_prob_welfare(0, 1, 0.95, 1)
    before_75_percent = self._calloc_prob_welfare(0, 1, 0.75, 1)
    before_daily_cons = self._calloc_prob_welfare(0, 1, 1, 1)
    
    after_purchase_rate = self._calloc_prob(1, 1)
    after_95_percent = self._calloc_prob_welfare(1, 1, 0.95, 1)
    after_75_percent = self._calloc_prob_welfare(1, 1, 0.75, 1)
    after_daily_cons = self._calloc_prob_welfare(1, 1, 1, 1)
    
    # Create Table 5 Panel B
    print("---Table 5: Performance of Location Counterfactuals, Panel B: 1km Effective Distance---")
    table_5_panel_B_1 = pd.DataFrame([
        [before_purchase_rate, after_purchase_rate]
    ], columns=['Before', 'After'], index=['Purchase rate'])
    table_5_panel_B_1["Improvement"] = table_5_panel_B_1["After"] - table_5_panel_B_1["Before"]
    print(table_5_panel_B_1.to_latex(index=True, float_format='%.3f'))

    table_5_panel_B_2 = pd.DataFrame([
        [GMV, GMV * after_purchase_rate / before_purchase_rate]
    ], columns=['Before', 'After'], index=['Annual GMV (million)'])
    table_5_panel_B_2["Improvement"] = table_5_panel_B_2["After"] - table_5_panel_B_2["Before"]
    print(table_5_panel_B_2.to_latex(index=True, float_format='%.0f'))

    table_5_panel_B_3 = pd.DataFrame([
        [before_daily_cons, after_daily_cons],
        [before_95_percent, after_95_percent],
        [before_75_percent, after_75_percent]
    ], columns=['Before', 'After'], index=['Daily consumer welfare', '95% group daily welfare', '75% group daily welfare'])
    table_5_panel_B_3["Improvement"] = table_5_panel_B_3["After"] - table_5_panel_B_3["Before"]
    print(table_5_panel_B_3.to_latex(index=True, float_format='%.2f'))

    #### Figure 4(a)-4(d) Picking Better Locations of Pick-up Stations ####
    print('\nComputing Figure 4(a)-4(d) (picking better locations of pick-up stations)...')
    location_figures(self) 
    print('Finish Figure 4(a)-4(d)...')
    

    ############# Delivery windows counterfactual analysis ##################
    print('\nStarting delivery windows counterfactual analysis...')
    
    # Set random seed and load Gaussian parameters
    seed = 0
    self._read_gaussian()
    
    #### Table 6: Delivery Windows Counterfactuals, Panel A: Parameters and Results ####
    print('\nComputing Table 6, Panel A (parameters and results)...')
    time_ctr_prob, type1_deliver, type2_deliver, type3_deliver = self._time_counterfactual()
    
    #### Table 6: Delivery Windows Counterfactuals, Panel B: Comparison of Performance ####
    print('\nComputing Table 6, Panel B (comparison of performance)...')
    utility_average, utility_before, utility_calculate, utility_cal_after, utility_cal_before = self._time_counterfactual_welfare(q=1)
    prob_wsta_avg, prob_nsta_avg = self._compare_time_ctr()  # wsta: with station (After), nsta: without station (Before)
    
    # Create Table 6 Panel B
    print("---Table 6: Delivery Windows Counterfactuals, Panel B: Comparison of Performance---")
    table_6_panel_B_1 = pd.DataFrame([
        [prob_nsta_avg, time_ctr_prob]
    ], columns=['Before', 'After'], 
       index=['Purchase rate'])
    table_6_panel_B_1["Improvement"] = table_6_panel_B_1["After"] - table_6_panel_B_1["Before"]
    print(table_6_panel_B_1.to_latex(index=True, float_format='%.3f'))

    table_6_panel_B_2 = pd.DataFrame([
        [GMV, GMV * time_ctr_prob / prob_nsta_avg]
    ], columns=['Before', 'After'], 
       index=['Annual GMV (million)'])
    table_6_panel_B_2["Improvement"] = table_6_panel_B_2["After"] - table_6_panel_B_2["Before"]
    print(table_6_panel_B_2.to_latex(index=True, float_format='%.0f'))

    table_6_panel_B_3 = pd.DataFrame([
        [utility_cal_before / price_coeff, utility_cal_after / price_coeff]
    ], columns=['Before', 'After'], 
       index=['Daily consumer welfare'])
    table_6_panel_B_3["Improvement"] = table_6_panel_B_3["After"] - table_6_panel_B_3["Before"]
    print(table_6_panel_B_3.to_latex(index=True, float_format='%.2f'))

