from einops import repeat, rearrange

import torch
import torch.nn as nn
    

class NormCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Normalizes the coordinates to be strictly inside [-1, 1], along the second dimension
        :param z: tensor of coordinates, of shape W x N x 4
        """
        
        # Ignore the nan values
        # NB : nan means that the coordinates are unknown (due to an occlusion for instance)
        #z = z.to(torch.float64)  # NB: this is needed for torch.where
        z_min_nonan = torch.where(torch.isnan(z), float('inf'), z)
        z_max_nonan = torch.where(torch.isnan(z), -float('inf'), z)
        
        # Find the extrema using torch.min and torch.max
        xmin = torch.min(z_min_nonan[:, :, 0], dim=0, keepdim=True).values   # 1 x N        
        ymin = torch.min(z_min_nonan[:, :, 1], dim=0, keepdim=True).values   # 1 x N
        xmax = torch.max(z_max_nonan[:, :, 2], dim=0, keepdim=True).values   # 1 x N
        ymax = torch.max(z_max_nonan[:, :, 3], dim=0, keepdim=True).values   # 1 x N
        
        minimum = torch.stack((xmin, ymin, xmin, ymin), axis=2)              # 1 x N x 4
        maximum = torch.stack((xmax, ymax, xmax, ymax), axis=2)              # 1 x N x 4
                
        # Normalize inside [-1, 1]
        z = 2 * (z - minimum) / (maximum - minimum) - 1                      # W x N x 4

        return z#.to(torch.float32)


class TemporalPositionEncoding(nn.Module):
    
    def __init__(self, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
    def forward(self, framesP, framesF, fps):
        """
        Encode the temporal positional encoding on each observation
        :param framesP : tensor with the frames of past tracklets, of shape WP x NP x 1
        :param framesF : tensor with the frames of future tracklets, of shape WF x NF x 1
        :param fps: frame per second of the video
        """
        WP, NP, _ = framesP.shape
        WF, NF, _ = framesF.shape

        # Combine frames to get pairs of tracklets
        framesP_rep = repeat(framesP, 'w m d -> w (m n) d', n=NF)  # WP x (NP*NF) x 1
        framesF_rep = repeat(framesF, 'w n d -> w (m n) d', m=NP)  # WF x (NP*NF) x 1
        frames_rep = torch.cat((framesP_rep, framesF_rep), dim=0)  # (WP+WF) x (NP*NF) x 1

        # Normalize such that the first frame of the future tracklet is 0
        first_framesF = framesF[0]#.to(torch.float64)  # NB: this is needed for torch.where       # NF x 1
        framesF_min_nonan = torch.where(torch.isnan(first_framesF), float('inf'), first_framesF) # NF x 1
        temporal_shift = torch.min(framesF_min_nonan, dim=0, keepdims=True).values               # 1 x 1
        
        frames_rep = (frames_rep - temporal_shift) / fps            # unit in seconds, (WP+WF) x (NP*NF) x 1

        # Apply a cos/sin positional encoding
        temporal_encoding = torch.zeros((WP+WF, NP*NF, self.d_model)).to(self.device)                 # (WP+WF) x (NP*NF) x D

        div_term = 1 / ((10**4) ** (torch.arange(0, self.d_model, dtype=torch.float)/self.d_model)).to(self.device)  # D
        tables = frames_rep * div_term                                 # (WP+WF) x (NP*NF) x 1

        temporal_encoding[:, :, 0::2] = torch.sin(tables[:, :, 0::2])  # (WP+WF) x (NP*NF) x D
        temporal_encoding[:, :, 1::2] = torch.cos(tables[:, :, 1::2])  # (WP+WF) x (NP*NF) x D

        return temporal_encoding   # (WP+WF) x (NP*NF) x D
    

class TWiX(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout, inter_pair=True, device='cuda'):
        super().__init__()

        self.device = device
        self.inter_pair = inter_pair
        
        self.norm_coordinates = NormCoords()
        self.projection = nn.Linear(4, d_model)
        self.temporalEncoding = TemporalPositionEncoding(d_model=d_model, device=device)
        self.cls_pair_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer Encoder applied on a pair of tracklet
        self.encoder_intra = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers, enable_nested_tensor=False
        )

        # Transformer Encoder applied on set of pair of tracklets
        if self.inter_pair:
            self.encoder_inter = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers, enable_nested_tensor=False
            )

        self.mlp = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()
         
    def forward(self, coordsP, framesP, coordsF, framesF, fps):
        """
        Compute a similarity matrix between the tracklets of the past and those of the future
        :param coordsP : coordinates of the past tracklets, shape (WP x NP x 4)
        :param framesP : timestamp of the past tracklets, shape (WP x NP x 1)
        """

        # Size of the window
        assert len(coordsP) == len(framesP)
        assert len(coordsF) == len(framesF)

        WP, NP, _ = framesP.shape
        WF, NF, _ = framesF.shape

        if (NP == 0) or (NF == 0):
            return torch.zeros((NP, NF)).to(self.device)

        # Create pairs of tracklets
        coordsP_rep = repeat(coordsP, 'w m d -> w (m n) d', n=NF)  # WP x (NP*NF) x 4
        coordsF_rep = repeat(coordsF, 'w n d -> w (m n) d', m=NP)  # WF x (NP*NF) x 4      
        x = torch.cat((coordsP_rep, coordsF_rep), dim=0)           # (WP+WF) x (NP*NF) x 4

        # Normalize the spatial coordinates inside each pair
        x = self.norm_coordinates(x)                     # (WP+WF) x (NP*NF) x 4

        # Replace NaN from padding values. Otherwise, softmax return NaN
        x[x != x] = 0                                    # (WP+WF) x (NP*NF) x 4

        # Linear projection
        x = self.projection(x)                           # (WP+WF) x (NP*NF) x D

        # Add the temporal positional encoding : the timestamp for each observation (t=0 is the first one of the future tracklet)
        x = x + self.temporalEncoding(framesP, framesF, fps)             # (WP+WF) x (NP*NF) x D

        # Replace NaN from padding values. Otherwise, softmax return NaN
        x[x != x] = 0                                                    # (WP+WF) x (NP*NF) x D

        # Padding mask in the attention mechanism for the first encoder
        framesP_rep = repeat(framesP, 'w m d -> w (m n) d', n=NF)             # WP x (NP*NF) x 1
        framesF_rep = repeat(framesF, 'w n d -> w (m n) d', m=NP)             # WF x (NP*NF) x 1
        frames_rep = torch.cat((framesP_rep, framesF_rep), dim=0)             # (WP+WF) x (NP*NF) x 1

        padmask = rearrange(torch.isnan(frames_rep), 'w n () -> n w')                         # (NP*NF) x (WP+WF)
        false_pad_cls = repeat(torch.tensor([False]), '() -> k ()', k=NP*NF).to(self.device)  # (NP*NF) x 1
        padmask = torch.cat((false_pad_cls, padmask), dim=1)                                  # (NP*NF) x (WP+WF+1)

        # Add a CLS token for each pair of tracklet
        repeat_cls_token = repeat(self.cls_pair_token, '() () d -> () n d', n=NP*NF)  # 1 x (NP*NF) x D
        x = torch.cat((repeat_cls_token, x), dim=0)                                   # (WP+WF+1) x (NP*NF) x D

        # Transformer Encoder applied on each pair
        # NB : an attention mask is applied to take into account the padding 
        x = self.encoder_intra(x, src_key_padding_mask=padmask)                       # (WP+WF+1) x (NP*NF) x D
        x = x[0]                                                                      # (NP*NF) x D

        x = rearrange(x, 'b d -> b () d')             # (NP*NF) x 1 x D

        # Skip connection + Inter-Pair Transformer Encoder
        if self.inter_pair:
            x = x + self.encoder_inter(x)                 # (NP*NF) x 1 x D
           
        # Linear linear and tanh
        x = self.mlp(x)                               # (NP*NF) x 1 x 1
        x = self.tanh(x)                              # (NP*NF) x 1 x 1
        x = rearrange(x, '(m n) () () -> m n', m=NP)  # NP x NF

        return x