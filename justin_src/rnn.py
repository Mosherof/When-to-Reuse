#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:37:40 2026

@author: dinc, ju5tinz
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from viz import viz_R, animate_R

class FlipFlopTask:
    def __init__(self, batch_size, T=100, dt=1.0, flip_prob=0.05, bits=2):
        self.batch_size = batch_size
        self.T = T
        self.dt = dt
        self.flip_prob = flip_prob
        self.bits = bits
        self.U = None
        self.O = None
        
    def generate(self):
        self.U = torch.zeros(self.batch_size, self.T, self.bits)
        self.O = torch.zeros(self.batch_size, self.T, self.bits)
        
        for b in range(self.batch_size):
            for ch in range(self.bits):
                self.U[b, 0, ch] = np.random.choice([-1, 1])
                self.O[b, 0, ch] = self.U[b, 0, ch]
            
            for t in range(1, self.T):
                for ch in range(self.bits):
                    if np.random.rand() < self.flip_prob:
                        self.U[b, t, ch] = np.random.choice([-1, 1])
                
                for ch in range(self.bits):
                    if self.U[b, t, ch] != 0:
                        self.O[b, t, ch] = self.U[b, t, ch]
                    else:
                        self.O[b, t, ch] = self.O[b, t-1, ch]
        
        return self.U, self.O
    
    def get(self):
        if self.U is None or self.O is None:
            return self.generate()
        return self.U, self.O

class RNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, output_size=2, tau=2.0, dt=1.0, use_bias=False):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.use_bias = use_bias
        
        std = 1.0 / np.sqrt(hidden_size)
        
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * std)
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) * std)
        
        if use_bias:
            self.b_rec = nn.Parameter(torch.zeros(hidden_size))
            self.b_out = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_buffer('b_rec', torch.zeros(hidden_size))
            self.register_buffer('b_out', torch.zeros(output_size))
    
    def reinitialize_weights(self, weights=None):
        """
        Reinitialize specified weight matrices.
        
        Parameters:
        -----------
        weights : list of str or None
            List of weight names to reinitialize. Options: 'W_rec', 'W_in', 'W_out', 'b_rec', 'b_out'.
            If None, reinitializes all weights.
        """
        print(f"Reinitializing weights: {weights if weights is not None else 'all'}")
        if weights is None:
            weights = ['W_rec', 'W_in', 'W_out']
            if self.use_bias:
                weights.extend(['b_rec', 'b_out'])
        
        std = 1.0 / np.sqrt(self.hidden_size)
        
        for weight in weights:
            if weight == 'W_rec':
                self.W_rec.data = torch.randn(self.hidden_size, self.hidden_size) * std
            elif weight == 'W_in':
                self.W_in.data = torch.randn(self.hidden_size, self.input_size) * std
            elif weight == 'W_out':
                self.W_out.data = torch.randn(self.output_size, self.hidden_size) * std
            elif weight == 'b_rec' and self.use_bias:
                self.b_rec.data = torch.zeros(self.hidden_size)
            elif weight == 'b_out' and self.use_bias:
                self.b_out.data = torch.zeros(self.output_size)
            elif weight in ['b_rec', 'b_out'] and not self.use_bias:
                print(f"Warning: Cannot reinitialize {weight} when use_bias=False")
            else:
                raise ValueError(f"Unknown weight: {weight}")
            
    def get_hidden_states(self, U):
        """
        Runs a forward pass and returns only the hidden states.
        
        Args:
            U: Input tensor of shape (batch_size, T, input_size)
            
        Returns:
            R: Hidden states tensor of shape (batch_size, T + 1, hidden_size)
        """
        # Run the forward pass
        with torch.no_grad():
            R, _ = self.forward(U)
        
        # Optional: If you want to exclude the random initial state at t=0
        # and only keep the states driven by inputs, you can return R[:, 1:, :] instead.
        return R
        
    def forward(self, U):
        batch_size, T, _ = U.shape
        
        r = torch.randn(batch_size, self.hidden_size) * np.sqrt(0.1)
        
        R = torch.zeros(batch_size, T + 1, self.hidden_size)
        R[:, 0, :] = r
        
        O = torch.zeros(batch_size, T, self.output_size)
        
        for t in range(T):
            u_t = U[:, t, :]
            
            activation = torch.matmul(r, self.W_rec.t()) + \
                           torch.matmul(u_t, self.W_in.t()) + \
                           self.b_rec
                        
            r = (1 - self.alpha) * r + self.alpha * torch.tanh(activation)
            
            R[:, t + 1, :] = r
            
            o_t = torch.matmul(r, self.W_out.t()) + self.b_out
            O[:, t, :] = o_t
        
        return R, O


def loss_function(O_pred, O_target):
    """
    Mean squared error loss as defined in Eq. (24):
    L(Θ) = (1/(T*bits*B)) * Σ_i Σ_j Σ_k (Ô_ijk - O_ijk)²
    
    Parameters:
    -----------
    O_pred : torch.Tensor
        Predicted outputs of shape (B, T, bits)
    O_target : torch.Tensor
        Target outputs of shape (B, T, bits)
    """
    B = O_pred.shape[0]
    T = O_pred.shape[1]
    bits = O_pred.shape[2]
    mse = torch.sum((O_pred - O_target) ** 2) / (T * bits * B)
    return mse


def train_rnn(rnn, task, n_epochs=2000, learning_rate=1e-3, batch_size=128):
    """
    Train the RNN on the flip flop task.
    
    Parameters:
    -----------
    rnn : RNN
        The RNN model to train
    task : FlipFlopTask
        Task generator
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for ADAM optimizer
    batch_size : int
        Batch size for training
    """
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    
    losses = []
    
    print("Training RNN on flip flop task...")
    print(f"Epochs: {n_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")
    print("-" * 60)
    
    # Generate new batch of data
    task.batch_size = batch_size
    U, O_target = task.get()
    
    R_list = []
    weight_snapshots = []
    
    for epoch in range(n_epochs):
        
        
        # Forward pass
        optimizer.zero_grad()
        R, O_pred = rnn(U)
        
        # Compute loss
        loss = loss_function(O_pred, O_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        R_list.append(R.detach().cpu())
        weight_snapshots.append({
            'W_rec': rnn.W_rec.detach().cpu().clone(),
            'b_rec': rnn.b_rec.detach().cpu().clone(),
        })
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    print("-" * 60)
    print("Training complete!")
    
    return R_list, losses, weight_snapshots

def inference_rnn(rnn, task):
    """
    Run inference on the trained RNN and compute final loss and accuracy.
    
    Parameters:
    -----------
    rnn : RNN
        The trained RNN model
    task : FlipFlopTask
        Task generator
    """
    U_test, O_test = task.get()

    print("Running inference...")
    print("-" * 60)

    with torch.no_grad():
        R_list = []
        losses = []
        weight_snapshots = []
        for epoch in range(1000):
            R, O_pred_test = rnn(U_test)
            R_list.append(R.detach().cpu())
            loss = loss_function(O_pred_test, O_test)
            losses.append(loss.item())
            weight_snapshots.append({
                'W_rec': rnn.W_rec.detach().cpu().clone(),
                'b_rec': rnn.b_rec.detach().cpu().clone(),
            })

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.6f}")

    print("-" * 60)
    print(f"Inference complete! Mean loss: {sum(losses)/len(losses):.6f}")

    return R_list, losses, weight_snapshots

# Main execution
#if __name__ == "__main__":
