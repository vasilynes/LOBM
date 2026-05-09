import torch 
import torch.nn as nn

class LOB_NN(nn.Module):
    """
        Processes a tensor of dim (Batch, Channels, Time, Levels), 
        where Channels are configured to be AskP, AskQ, BidP, BidQ, 
        Time is configured to be 100, Levels are 20 LOB levels.

        In forward, the tensor is passed through multiple convolutions,
        resulting in four tensors of dim (Batch, 64, Time, 1).
        To preserve the Time dimension magnitude as is (100), 
        tensors are causally padded at the beginning of the sequence.
        The results are concatenated along the second dimension
        and squeezed with permutation to a tensor (Batch, Time, 256).
        Engineered features are added to the tensor, resulting in 
        (Batch, Time, 256 + num_global) dimensions.

        The output is fed to the 1-Layer (64) GRU network, whose  
        hidded states are then passed through temporal pooling Linear 
        layer (64 -> 1) and softmaxed to obtain weights for a linear 
        combination of GRU hidden states, which is then passed through 
        Linear layers (64 -> 32 -> output_dim), where output_dim 
        defaults to 1. The resulting tensor is of dim (Batch, output_dim).
    """
    def __init__(self, output_dim=1, num_global=11):
        super(LOB_NN, self).__init__()

        # Calculate microstructure features for each Level
        self.conv_micro = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Calculate macrostructure features along all Levels
        self.conv_macro = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 20)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Convolve along Time dimension with kernels of different sizes
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=(0, 0)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 4, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Max pool features and convolve 32 -> 64 for dimension compatibility
        self.inp_pool = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 0)), 
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

        self.cnn_norm = nn.LayerNorm(256)

        hidden_size = 64

        self.gru = nn.GRU(
            input_size=256 + num_global, 
            hidden_size=hidden_size,
            num_layers=1, 
            batch_first=True
        )

        self.gru_norm = nn.LayerNorm(hidden_size)

        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64, 32), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim)
        )

        self._fwd_stats: dict[str, float] = {}
        self._grad_norms: dict[str, float] = {}
        self._register_grad_hooks() 
    
    def _register_grad_hooks(self):
        """
        Register backward hooks on the weights of CNN and GRU, 
        saving L2 norms into self._grad_norms.
        """
        tracked = {
            'cnn_macro/weight': self.conv_macro[0].weight,
            'gru/weight_ih': self.gru.weight_ih_l0,
            'gru/weight_hh': self.gru.weight_hh_l0
        }
        for k, p in tracked.items():
            p.register_hook(
                lambda g, k=k: self._grad_norms.update({k: g.norm().item()})
            )

    def _record_fwd_stats(self, cnn_out: torch.Tensor, h: torch.Tensor):
        """
        Compute and cache forward-pass stats for CNN output 
        and GRU hidden state tensors. 
        """
        c = cnn_out.detach()
        g = h.detach()
        self._fwd_stats = {
            'cnn_out/mean':    c.mean().item(),
            'cnn_out/std':     c.std().item(),
            'gru_h/mean':      g.mean().item(),
            'gru_h/std':       g.std().item(),
        }

    def diagnose(self) -> dict[str, float]:
        """
        Return stats of the last forward-pass output stats
        and the last backward-pass gradient stats. If backward 
        pass is not complete, gradient entries are absent.
        """
        return {**self._fwd_stats, **self._grad_norms}

    def forward(self, x_lob, x_global):
        x = self.conv_micro(x_lob)
        x = self.conv_macro(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x_pool = self.inp_pool(x)

        # (Batch, 256, Time, 1)
        x_cat = torch.cat([x_inp1, x_inp2, x_inp3, x_pool], dim=1)
        # (Batch, Time, 256)
        cnn_out = x_cat.squeeze(-1).permute(0, 2, 1)
        cnn_out = self.cnn_norm(cnn_out)
        # (Batch, Time, 256 + num_global)
        x = torch.cat([cnn_out, x_global], dim=-1)

        h, _ = self.gru(x)
        # Record stats before normalization, so that
        # LN doesn't distort them
        self._record_fwd_stats(cnn_out, h)

        h_norm = self.gru_norm(h)
        attn = torch.softmax(self.attn(h_norm), dim=1)
        h_comb = (attn * h).sum(dim=1) 
        pred = self.fc(h_comb)
  
        return pred, attn