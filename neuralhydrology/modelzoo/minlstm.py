from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class MinLSTM(BaseModel):
    """MinLSTM model class, which relies on PyTorch's LSTM class.
    
    Following Feng, L., Tung, F., Ahmed, M.O., Bengio, Y., Hajimirsadeghi, 
    H., 2024. Were RNNs All We Needed? https://doi.org/10.48550/arXiv.2410.01201

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MinLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        
        self.forget_gate = nn.Linear(self.embedding_net.output_size, cfg.hidden_size)
        self.input_gate = nn.Linear(self.embedding_net.output_size, cfg.hidden_size)
        self.candidate_h = nn.Linear(self.embedding_net.output_size, cfg.hidden_size)
        
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.forget_gate.bias.data[:] = self.cfg.initial_forget_bias
            
    def _parallel_scan_log(log_coeffs, log_values): 
        # log_coeffs: (batch_size, seq_len, input_size)
        # log_values: (batch_size, seq_len + 1, input_size) 
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0)) 
        log_h0_plus_b_star = torch.logcumsumexp(
            log_values - a_star, dim=1
            ) 
        log_h = a_star + log_h0_plus_b_star 
        
        return torch.exp(log_h)[:, 1:]
    
    @staticmethod
    def _g(x: torch.Tensor) -> torch.Tensor:
        """g(x) activation from the minGRU/minLSTM paper (ensures positive candidate values)."""
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))
    
    @staticmethod
    def _log_g(x: torch.Tensor) -> torch.Tensor:
        """Numerically-stable log(g(x))."""
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
    
    @staticmethod
    def _parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """Parallel log-space scan (Heinsen, 2023), as used in the minLSTM paper.

        Parameters
        ----------
        log_coeffs : torch.Tensor
            Shape [batch_size, seq_len, hidden_size].
        log_values : torch.Tensor
            Shape [batch_size, seq_len + 1, hidden_size] (includes initial state at t=0).

        Returns
        -------
        torch.Tensor
            Shape [batch_size, seq_len, hidden_size].
        """
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star

        return torch.exp(log_h)[:, 1:]

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        
        x_d_batch_first = x_d.transpose(0, 1)  # [batch_size, seq_len, input_size]  
    
        batch_size = x_d_batch_first.shape[0]

        # Initial hidden state:
        # [batch_size, hidden_size]
        h_0 = x_d_batch_first.new_zeros(
            batch_size,
            self.cfg.hidden_size
            )
        

        diff = F.softplus(-self.forget_gate(x_d_batch_first)) / -F.softplus(-self.input_gate(x_d_batch_first))
        log_f = -F.softplus(diff) 
        log_i = -F.softplus(-diff)
        log_h_0 = torch.log(h_0) 
        log_tilde_h = self._log_g(self.candidate_h(x_d_batch_first))
        log_values = torch.cat([log_h_0.unsqueeze(1), log_i + log_tilde_h], dim=1)
        hidden_states = self._parallel_scan_log(
            log_coeffs=log_f, log_values=log_values
        )
        

        h_n = hidden_states[:, -1:, :]

        pred = {'lstm_output': hidden_states, 'h_n': h_n}
        pred.update(self.head(self.dropout(hidden_states)))

        return pred
