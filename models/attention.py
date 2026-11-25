import torch
from torch import nn

part_6_names = ["head", "hand", "arm", "hip", "leg", "foot"]

skeleton_part_6 = {
    "head": [10, 11],
    "hand": [14, 15, 16, 19, 20, 21],
    "arm": [12, 13, 17, 18],
    "hip": [8, 9],
    "leg": [0, 1, 4, 5],
    "foot": [1, 2, 3, 5, 6, 7]
}

class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim=[128,128], hidden_dim=128):
        super(CrossAttentionFusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_q = nn.Linear(self.input_dim[0], hidden_dim)
        self.input_k = nn.Linear(self.input_dim[1], hidden_dim)
        self.input_v = nn.Linear(self.input_dim[1], hidden_dim)

    def forward(self, input_a, input_b):

        bs = input_a.size(0)

        q = self.input_q(input_a)
        k = self.input_k(input_b)
        v = self.input_v(input_b)
        score = torch.einsum("bid,bjd->bij", q, k) / (self.hidden_dim ** 0.5)
        att_weight = nn.functional.softmax(score, dim=-1)
        output = torch.bmm(att_weight, v)

        return output
