import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def lambda_init_fn(depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiheadFusionAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        depth,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Linear projections for left-hand and right-hand features
        self.q_proj_left = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj_right = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_left = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_right = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj_left = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj_right = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Initialize learnable parameters for weighting
        # self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_init = 0.1
        self.lambda_q_left = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_left = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_right = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_right = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def lambda_init_fn(self,depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, left_features, right_features, attn_mask=None):
        # x_self = x_self.permute(0, 2, 1)  # (16. 30, 2048)
        # x_other = x_other.permute(0, 2, 1)  # (16. 30, 2048)
        left_features = left_features.permute(0, 2, 1)  # (16. 30, 2048)
        right_features = right_features.permute(0, 2, 1)  # (16. 30, 2048)
        bsz, tgt_len, embed_dim = left_features.size()
        src_len = tgt_len

        # Project left and right features separately
        q_left = self.q_proj_left(left_features)
        k_left = self.k_proj_left(left_features)
        v_left = self.v_proj_left(left_features)

        q_right = self.q_proj_right(right_features)
        k_right = self.k_proj_right(right_features)
        v_right = self.v_proj_right(right_features)

        # Reshape for multi-head attention
        q_left = q_left.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, tgt_len, head_dim)
        k_left = k_left.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, src_len, head_dim)
        v_left = v_left.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, src_len, head_dim)

        q_right = q_right.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, tgt_len, head_dim)
        k_right = k_right.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, src_len, head_dim)
        v_right = v_right.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (bsz, num_heads, src_len, head_dim)

        # Attention scaling
        q_left *= self.scaling
        q_right *= self.scaling

        # Compute attention weights for left and right features
        attn_weights_left = torch.matmul(q_left, k_left.transpose(-1, -2))  # Shape: (bsz, num_heads, tgt_len, src_len)
        attn_weights_right = torch.matmul(q_right, k_right.transpose(-1, -2))  # Shape: (bsz, num_heads, tgt_len, src_len)

        # Apply attention mask if provided
        if attn_mask is None:
            attn_mask = torch.triu(torch.zeros([tgt_len, src_len]).float().fill_(float("-inf")), 1)
        attn_mask = attn_mask.to(attn_weights_left.device)#tocheck
        
        # Expand attn_mask to match the shape of attn_weights_left and attn_weights_right
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, tgt_len, src_len)
        attn_mask = attn_mask.expand(bsz, self.num_heads, tgt_len, src_len)  # Shape: (bsz, num_heads, tgt_len, src_len)

        attn_weights_left += attn_mask
        attn_weights_right += attn_mask

        # Softmax to normalize attention weights
        attn_weights_left = F.softmax(attn_weights_left, dim=-1)
        attn_weights_right = F.softmax(attn_weights_right, dim=-1)

        # Compute lambda values for modulation
        lambda_left = torch.exp(torch.sum(self.lambda_q_left * self.lambda_k_left, dim=-1)).type_as(q_left)
        lambda_right = torch.exp(torch.sum(self.lambda_q_right * self.lambda_k_right, dim=-1)).type_as(q_right)
        lambda_full = lambda_left - lambda_right + self.lambda_init

        # Weighted sum of values for left and right
        attn_left = torch.matmul(attn_weights_left, v_left)  # Shape: (bsz, num_heads, tgt_len, head_dim)
        attn_right = torch.matmul(attn_weights_right, v_right)  # Shape: (bsz, num_heads, tgt_len, head_dim)

        # Combine left and right attention outputs using learned lambda_full weights
        attn_combined = attn_left - lambda_full.unsqueeze(-1).unsqueeze(-1) * attn_right  # Shape: (bsz, num_heads, tgt_len, head_dim)

        # Normalize and output
        attn_combined = attn_combined.transpose(1, 2).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)  # Shape: (bsz, tgt_len, embed_dim)
        attn_combined = self.subln(attn_combined)
        attn_combined = self.out_proj(attn_combined)
        attn_combined = self.mlp(attn_combined)
        
        # 维度转换
        # x_fused = x_fused.permute(0, 2, 1)  # (16. 30, 2048)
        attn_combined = attn_combined.permute(0, 2, 1)

        return attn_combined
if __name__ == '__main__':
    # Example usage
    embed_dim = 30
    num_heads = 8
    depth = 12
    model = MultiheadFusionAttn(embed_dim, num_heads, depth)

    # Dummy left and right hand features
    left_features = torch.randn(2, 2048, embed_dim)  # Batch size of 16, sequence length of 128
    right_features = torch.randn(2, 2048, embed_dim)

    # Forward pass
    output = model(left_features, right_features)  # Assuming rel_pos is provided as needed
    print(output.shape)  # Expected output shape: (16, 128, embed_dim)
