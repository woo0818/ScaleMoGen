import torch
import torch.nn as nn
import torch.nn.init as init
from model.blocks import Resnet1D, SimpleConv1dLayer, Conv1dLayer


def length_to_mask(length, max_len=None, device: torch.device = None):
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)
    
    if max_len is None:
        max_len = max(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def reparametrize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)

class GlobalRegressor(nn.Module):
    def __init__(self, dim_in, dim_latent, dim_out):
        super().__init__()
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv1d(dim_in, dim_latent, 3, 1, 1),
                nn.LeakyReLU(0.2)
                )
        )
        
        layers.append(Resnet1D(dim_latent, n_depth=3, dilation_growth_rate=3, reverse_dilation=True))
        # layers.append(Resnet1D(dim_latent, n_depth=2, dilation_growth_rate=3, reverse_dilation=True))
        layers.append(nn.Conv1d(dim_latent, dim_out, 3, 1, 1))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)



    def forward(self, input):
        input = input.permute(0, 2, 1)
        return self.layers(input).permute(0, 2, 1)
    


############################################
################# VQ Model #################
############################################

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x, keep_shape=False):
        x = self.model(x)
        if keep_shape:
            return x
        else:
            return x.permute(0, 2, 1)
        

class EncoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None):
        super().__init__()

        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.embed = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn=use_attn))
        self.outproj = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        # blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        # self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x, m_lens=None):
        x = self.embed(x)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens = m_lens//2
            x = attn_block(x, m_lens)
        return self.outproj(x)


class DecoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn = False,
                 norm=None):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn))

        self.outproj = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )
        self.apply(init_weights)

    def forward(self, x, m_lens=None, keep_shape=False):
        x = self.embed(x)

        # m_lens //= 2**len(self.res_blocks)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens *= 2
            x = attn_block(x, m_lens)

        x = self.outproj(x)

        if keep_shape:
            return x
        else:
            return x.permute(0, 2, 1)


def make_attn(in_channels, use_attn=True):
    return AttnBlock(in_channels) if use_attn else MultiInputIdentity()


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(in_channels, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, m_lens):
        x = x.permute(0, 2, 1)
        key_mask = length_to_mask(m_lens, x.shape[1])

        attn_out, _ = self.attn_block(
            self.norm(x), self.norm(x), self.norm(x), key_padding_mask = ~key_mask
        )

        x = x + attn_out
        return x.permute(0, 2, 1)
    

class MultiInputIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m_lens=None):
        return x

############################################
######## KL Regulerized AutoEncoder ########
############################################
class AutoEncoderKL(nn.Module):
    def __init__(self, nfeats, latent_dim, down_t, width, depth, dilation_growth_rate=3, scale=None, shift=None):
        super().__init__()

        # self.encoder = AEEncoder(input_emb_width=nfeats, 
        #                          output_emb_width=latent_dim*2,
        #                          down_t=down_t,
        #                          width=width,
        #                          depth=depth,
        #                          dilation_growth_rate=dilation_growth_rate,
        #                          activation="leakyrelu")
        # self.decoder = AEDecoder(input_emb_width=latent_dim,
        #                          output_emb_width=nfeats,
        #                          down_t=down_t,
        #                          width=width,
        #                          depth=3,
        #                          dilation_growth_rate=3,
        #                          activation="leakyrelu")
        en_channels = [nfeats, 384, 512]
        self.encoder = AEEncoder(en_channels, output_size=latent_dim)
        de_channels = [latent_dim, 512, 384]
        self.decoder = AEDecoder(de_channels, output_size=nfeats)
        self.scale_factor = 2**down_t
        # if (shift is not None) and (scale is not None):
        #     self.scale = scale
        #     self.shift = shift
        #     self.rescale = True
        # else:
        #     self.rescale = False

        self.rescale = False

    def encode(self, x, m_length=None, sample_mean=True):
        # x = x.permute(0, 2, 1)
        if m_length is not None:
            mask = length_to_mask(m_length, x.shape[1], device=x.device)
            x = torch.where(mask.unsqueeze(-1), x, 0.)
        
        x = x.permute(0, 2, 1)
        mu, logvar = self.encoder(x).chunk(2, 1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        if sample_mean:
            return_vec = mu
        else:
            return_vec = reparametrize(mu, logvar)
        return_vec = return_vec.permute(0, 2, 1)

        if self.rescale:
            return_vec = (return_vec - self.shift) / self.scale
        return return_vec


    def decode(self, x, m_length=None):
        if self.rescale:
            x = x * self.scale + self.shift
        
        if m_length is not None:
            mask = length_to_mask(m_length//self.scale_factor, x.shape[1], device=x.device)
            x = torch.where(mask.unsqueeze(-1), x, 0.)
        
        x = x.permute(0, 2, 1)
        output = self.decoder(x)
        return output.permute(0, 2, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu, logvar = self.encoder(x).chunk(2, 1)
        logvar = torch.clamp(logvar, -10.0, 10.0)

        latent = reparametrize(mu, logvar)
        output = self.decoder(latent)
        return output.permute(0, 2, 1), mu, logvar


class AEEncoder(nn.Module):
    def __init__(self, channels, output_size):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(channels) - 1
        # self.ToMidPoint = []
        model = []
        for i in range(1, n_down+1):
            model.append(
                Conv1dLayer(channels[i-1], channels[i], kernel_size=3, drop_prob=0.2, downsample=True)
            )
        # if vae_encoder:
        #     model.append(Conv1dLayer(channels[-1], output_size*2, kernel_size=1, activate=False))
        # else:
        model.append(Conv1dLayer(channels[-1], output_size*2, kernel_size=1, activate=False))
        self.model = nn.Sequential(*model)
        # self.vae_encoder = vae_encoder

    def forward(self, input):
        output = self.model(input)
        return output
        # return sp_mu, sp_logvar


class AEDecoder(nn.Module):
    def __init__(self, channels, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.n_up = len(channels) - 1
        # 32 -> 64 -> 128 -> 256
        # 512 -> 1024 -> 512 -> 263
        n_up = len(channels) - 1
        model = []
        model.append(Conv1dLayer(channels[0], channels[0], kernel_size=3, downsample=False))
        for i in range(n_up):
            model.append(SimpleConv1dLayer(channels[i], channels[i+1], upsample=True))

        model.append(Conv1dLayer(channels[-1], output_size, kernel_size=1, activate=False, downsample=False))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    

# class AEEncoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()

#         blocks = []
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())

#         for i in range(down_t):
#             input_dim = width
#             block = nn.Sequential(
#                 nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
#                 Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, output_emb_width, 1, 1, 0))
#         self.model = nn.Sequential(*blocks)
#         self.apply(init_weights)

#     def forward(self, x):
#         return self.model(x)


# class AEDecoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()
#         blocks = []

#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         for i in range(down_t):
#             out_dim = width
#             block = nn.Sequential(
#                 Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv1d(width, out_dim, 3, 1, 1)
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, output_emb_width, 1, 1, 0))
#         self.model = nn.Sequential(*blocks)
#         self.apply(init_weights)

#     def forward(self, x):
#         x = self.model(x)
#         return x


# class MotionEncoder(nn.Module):
#     def __init__(self, channels, output_size, vae_encoder=False):
#         super().__init__()
#         # channels = [263, 512, 1024, 1024]
#         # scale = [96, 48, 24, 12]
#         n_down = len(channels) - 1
#         # self.ToMidPoint = []
#         model = []
#         for i in range(1, n_down+1):
#             model.append(
#                 Conv1dLayer(channels[i-1], channels[i], kernel_size=3, drop_prob=0.2, downsample=True)
#             )
#         if vae_encoder:
#             model.append(Conv1dLayer(channels[-1], output_size*2, kernel_size=1, activate=False))
#         else:
#             model.append(Conv1dLayer(channels[-1], output_size, kernel_size=1, activate=False))
#         self.model = nn.Sequential(*model)
#         self.vae_encoder = vae_encoder

#     def forward(self, input):
#         output = self.model(input)
#         if self.vae_encoder:
#             mean, logvar = output.chunk(2, 1)
#             return reparametrize(mean, logvar), mean, logvar
#         else:
#             return output, None, None
#         # return sp_mu, sp_logvar


# class MotionDecoder(nn.Module):
#     def __init__(self, channels, output_size):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         # self.n_up = len(channels) - 1
#         # 32 -> 64 -> 128 -> 256
#         # 512 -> 1024 -> 512 -> 263
#         n_up = len(channels) - 1
#         model = []
#         model.append(Conv1dLayer(channels[0], channels[0], kernel_size=3, downsample=False))
#         for i in range(n_up):
#             model.append(SimpleConv1dLayer(channels[i], channels[i+1], upsample=True))

#         model.append(Conv1dLayer(channels[-1], output_size, kernel_size=1, activate=False, downsample=False))
#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)

# class MotionVAE(nn.Module):
#     def __init__(self, dim_pose, output_size, vae_encoder=True):
#         super().__init__()
#         self.vae_encoder = vae_encoder

#         # Encoder
#         en_channels = [dim_pose, 384, 512]
#         self.encoder = MotionEncoder(en_channels, output_size=output_size, vae_encoder=vae_encoder)

#         # Decoder
#         de_channels = [output_size, 512, 384]
#         self.decoder = MotionDecoder(de_channels, output_size=dim_pose)

#     def forward(self, input):
#         input = input.transpose(1, 2)
#         z, mean, logvar = self.encoder(input)
#         output = self.decoder(z).transpose(1, 2)
#         if self.vae_encoder:
#             return output, mean, logvar
#         return output