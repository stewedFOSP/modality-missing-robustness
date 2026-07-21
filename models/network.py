import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from models.stft_decoder import STFTDecoder
from models.stft_encoder import STFTEncoder
from models.complex_utils import new_complex_like
from models.get_layer_from_string import get_layer
# from models.TransformerEncoderCross import TransformerEncoderCross

EPS = 1e-5

class ModalFusionBlock(nn.Module):
    def __init__(
        self,
        chan,
        n_freqs,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        self.lnQ = nn.Linear(n_freqs, n_freqs)
        self.lnK = nn.Linear(n_freqs, n_freqs)
        self.lnV = nn.Linear(n_freqs, n_freqs)
        self.proj = nn.Sequential(
            nn.Linear(n_freqs, n_freqs),
            get_layer(activation)(),
            LayerNormalization4DCF((chan, n_freqs), eps=eps),
        )
        
    def forward(self, x):
        '''
        x: [B, C, T, F] C=3, aud 2 and vis 1
        '''
        q = self.lnQ(x) # [B, C, T, F]
        k = self.lnK(x) # [B, C, T, F]
        v = self.lnV(x) # [B, C, T, F]
        q = q.transpose(1, 2) # [B, T, C, F]
        k = k.transpose(1, 2) # [B, T, C, F]
        v = v.transpose(1, 2) # [B, T, C, F]
        attn_mat = q @ k.transpose(2, 3) # [B, T, C, C]
        attn_mat = F.softmax(attn_mat, dim=-1) # [B, T, C, C]
        out = attn_mat @ v # [B, T, C, F]
        out = out.transpose(1, 2) # [B, C, T, F]
        out = self.proj(out)
        
        return out

class TF_gridnet_attentionblock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        n_freqs,
        n_head,
        approx_qk_dim,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"


        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        # assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, E, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DCF(
                (n_head, E // n_head, n_freqs), eps=eps
            ),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(E, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.n_head = n_head

    def forward(self, batch, aux):
        """GridNetV2Block Forward.

        Args:
            batch: [B, C, T, Q]
            aux: [B, C, T, Q]
            out: [B, C, T, Q]
        """

        B, _, old_T, old_Q = batch.shape
        # print(batch.shape)
        aux_T = aux.shape[-2]
        # print('vis', vis.shape)

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, H, E, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](aux))  # [B, H, E, t, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](aux))  # [B, H, C/H, t, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [BHE, T, Q]
        K = K.view(-1, *K.shape[2:])  # [BHE, t, Q]
        V = V.view(-1, *V.shape[2:])  # [BC, t, Q]

        Q = Q.transpose(1, 2) # [BHE, Q, T]
        Q = Q.flatten(start_dim=2)  # [BHE, Q, T]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, aux_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        # V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.reshape([old_shape[0],old_T,old_shape[-2],old_shape[-1]])
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        return batch

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    # @amp.float_function
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    # @amp.float_function
    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    # @amp.float_function
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

class VisualConv1D(nn.Module):
    def __init__(self, V=512, H=1024):
        super(VisualConv1D, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(V)
        conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(H)
        dsconv = nn.Conv1d(H, H, 3, stride=1, padding=1,dilation=1, groups=H, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(H)
        pw_conv = nn.Conv1d(H, V, 1, bias=False)
        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class Model(nn.Module):
    """Offline separator network.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of separator blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        # input_dim,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        use_builtin_complex=False,
        ref_channel=-1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel

        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        
        self.l_conv = nn.Conv1d(512,65,1)
        self.L_TCN = nn.ModuleList([])
        for _ in range(5):
            self.L_TCN.append(VisualConv1D(V=65, H=130))
        # self.mf = ModalFusionBlock(2*n_imics+1, 65)

        self.f_conv = nn.Conv1d(512,65,1)
        self.F_TCN = nn.ModuleList([])
        for _ in range(5):
            self.F_TCN.append(VisualConv1D(V=65, H=130))

        # self.att_aux = TF_gridnet_attentionblock(emb_dim=2, n_freqs=n_freqs, n_head=4, approx_qk_dim=attn_approx_qk_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics + 1 + 1, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, emb_dim, ks, padding=padding),
        #     nn.GroupNorm(1, emb_dim, eps=eps),
        # )
        # self.att_av = TF_gridnet_attentionblock(emb_dim = 128, n_freqs = n_freqs, n_head = 4, approx_qk_dim = attn_approx_qk_dim)
        # self.att_va = TF_gridnet_attentionblock(emb_dim = 2, n_freqs = n_freqs, n_head = 1, approx_qk_dim = attn_approx_qk_dim)
        # self.lip_conv = nn.Linear(256, n_freqs)

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)        

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        lip_emb: torch.Tensor,
        face_emb: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            lip_emb (torch.Tensor): [B, T', E]
            face_emb (torch.Tensor): [B, T', E']
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization
        
        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        # batch = batch0.mag
        n_batch, _, n_frames, n_freqs = batch.shape
        # batch = self.conv(batch)  # [B, -1, T, F]
        #batch_enc = batch

        # face_emb = F.interpolate(face_emb.transpose(1, 2), size=(n_frames), mode='linear') # [B, E, T]
        face_emb = face_emb.unsqueeze(-1).expand(-1, -1, n_frames) # [B, E, T]
        face_emb = self.f_conv(face_emb.squeeze(1)) # [B, F, T]
        for ii in range(5):
            face_emb = self.F_TCN[ii](face_emb) # [B, F, T]
        face_emb = face_emb.transpose(1, 2).contiguous().unsqueeze(1) # [B, 1, T, F]
        
        lip_emb = F.interpolate(lip_emb.transpose(1, 2), size=(n_frames), mode='linear') # [B, E, T]
        lip_emb = self.l_conv(lip_emb.squeeze(1)) # [B, F, T]
        for ii in range(5):
            lip_emb = self.L_TCN[ii](lip_emb) # [B, F, T]
        lip_emb = lip_emb.transpose(1, 2).contiguous().unsqueeze(1) # [B, 1, T, F]
        
        batch = torch.cat([batch, lip_emb, face_emb], dim=1) # [B, 4, T, F]
        # batch = self.mf(batch) # [B, 3, T, F]
        batch = self.conv(batch) # [B, C, T, F]
        
        # lip_emb = lip_emb.repeat(1,batch.shape[1],1,1) # [B, 2*M, T, E]
        # batch = torch.cat((batch, lip_emb), dim=-1)
        # batch = self.ln(batch)

        for ii in range(self.n_layers):
            # batch = torch.cat((batch, lip_emb), dim=-1) # [B, 2*M, T, E+F]
            # batch = self.norm3[ii](self.norm1[ii](self.ln1[ii](lip_emb)*batch) + self.norm2[ii](self.ln2[ii](lip_emb))) #FiLM
            # batch = self.ln[ii](batch)
            batch = self.blocks[ii](batch, lip_emb)  # [B, -1, T, F], [B, newT, E]

        #batch = batch * batch_enc
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        # batch = new_complex_like(batch0, (batch.squeeze(1), batch0.phase))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization

        # batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks
        

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1), # [B, E, T, Q]
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1), # [B, E, T, Q]
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1), # [B, C/H, T, Q]
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        # self.E = E
        # self.approx_qk_dim = approx_qk_dim
        # self.ln = nn.Linear(2*6*)

    def forward(self, x, lip_emb):
        """GridNetBlock Forward.

        Args:
            C = channel dim
            T = n frames
            Q = n freq bins
            x: [B, C, T, Q]
            lip_emb: [B, 1, T, E]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
        # lip_emb = F.pad(lip_emb, (0, self.E*Q - self.approx_qk_dim, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, E, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, E, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C/H, T, Q]
        Q = torch.cat(all_Q, dim=0)  # [HB, E, T, Q]
        K = torch.cat(all_K, dim=0)  # [HB, E, T, Q]
        V = torch.cat(all_V, dim=0)  # [HB, C/H, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [HB, T, EQ]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [HB, T, EQ]
        V = V.transpose(1, 2)  # [HB, T, C/H, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [HB, T, C/H*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [HB, T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [HB, T, T]
        V = torch.matmul(attn_mat, V)  # [HB, T, C/H*Q]

        V = V.reshape(old_shape)  # [HB, T, C/H, Q]
        V = V.transpose(1, 2)  # [HB, C/H, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [H, B, C/H, T, Q])
        batch = batch.transpose(0, 1)  # [B, H, C/H, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class AllHeadPReLULayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x