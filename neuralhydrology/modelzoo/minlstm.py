from typing import Dict

import torch
import torch.nn as nn

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
            self.forget_gate.bias.data = self.cfg.initial_forget_bias
            self.input_gate.bias.data = self.cfg.initial_forget_bias
            self.candidate_h.bias.data = self.cfg.initial_forget_bias

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
        """
        
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        
        sequence_length = x_d.shape[0]
        batch_size = x_d.shape[1]

        # Compute all input-dependent quantities at once.
        # Each has shape:
        # [sequence_length, batch_size, hidden_size]
        f = torch.sigmoid(self.forget_gate(x_d))
        i = torch.sigmoid(self.input_gate(x_d))
        candidate_h = self.candidate_h(x_d)

        # Initial hidden state:
        # [batch_size, hidden_size]
        h_t = x_d.new_zeros(
            batch_size,
            self.cfg.hidden_size
        )

        hidden_states = []
        
        # minLSTM recurrence
        for t in range(sequence_length):
            h_t = (
                f[t] * h_t
                + i[t] * candidate_h[t]
            )

        hidden_states.append(h_t)

        # [sequence_length, batch_size, hidden_size]
        lstm_output = torch.stack(hidden_states, dim=0)

        # [batch_size, sequence_length, hidden_size]
        lstm_output = lstm_output.transpose(0, 1)

        # Match the shape used by the original model:
        # [batch_size, 1, hidden_size]
        h_n = h_t.unsqueeze(1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
