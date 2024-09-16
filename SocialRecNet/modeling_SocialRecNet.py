import math
from typing import List, Optional, Tuple, Union
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, LlamaForCausalLM
from transformers import LlamaConfig
import torch.nn.functional as F
try:
    from configuration import SocialRecNetConfig
except:
    from configuration import SocialRecNetConfig



def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask
def pad_or_trim(tensor, target_length):
    current_length = tensor.size(1)
    if current_length < target_length:
        # 如果長度小於 30，填充到 30
        padding = (0, target_length - current_length)
        tensor = F.pad(tensor, padding)
    elif current_length > target_length:
        # 如果長度大於 30，裁剪到 30
        tensor = tensor[:, :target_length]
    return tensor


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)
device = 'cpu'


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,dropout=0.1)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, embedding_A, embedding_B):
        # 確保嵌入維度一致
        assert embedding_A.size(-1) == self.multihead_attention.embed_dim, \
            f"Embedding dimension of A ({embedding_A.size(-1)}) does not match the model's expected dimension ({self.multihead_attention.embed_dim})"
        assert embedding_B.size(-1) == self.multihead_attention.embed_dim, \
            f"Embedding dimension of B ({embedding_B.size(-1)}) does not match the model's expected dimension ({self.multihead_attention.embed_dim})"

        # 創建 Q, K, V
        query = self.linear_q(embedding_A)  # shape: [batch_size, seq_len_A, dim]
        key = self.linear_k(embedding_B)    # shape: [batch_size, seq_len_B, dim]
        value = self.linear_v(embedding_B)  # shape: [batch_size, seq_len_B, dim]

        # 轉置輸入以符合 PyTorch MultiheadAttention 的要求
        query = query.transpose(0, 1)  # shape: [seq_len_A, batch_size, dim]
        key = key.transpose(0, 1)      # shape: [seq_len_B, batch_size, dim]
        value = value.transpose(0, 1)  # shape: [seq_len_B, batch_size, dim]

        # 計算 multihead cross attention
        attention_output, _ = self.multihead_attention(query, key, value)  # shape: [seq_len_A, batch_size, dim]

        # 將輸出轉置回原來的形狀
        attention_output = attention_output.transpose(0, 1)  # shape: [batch_size, seq_len_A, dim]
        return attention_output
# Model parameters

class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
    ):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return residual + x

class SocialRecNet(PreTrainedModel):
#     config_class = SocialRecNet
# Config
#     base_model_prefix = "blsp"

    def __init__(self, config: SocialRecNetConfig):
        super().__init__(config)
        self.llama_config = LlamaConfig(**config.llama_config)
        self.llama_model = LlamaForCausalLM(self.llama_config)

        out_d = self.llama_config.hidden_size

        self.subsampler_final = Conv1dSubsampler(
            512,
            2 *512,
            out_d,
            [int(k) for k in config.conv_kernel_sizes.split(",")],
        )

        self.speech_cross_attention = CrossAttention(embed_dim=768, num_heads=4)
        self.text_cross_attention = CrossAttention(embed_dim=4096, num_heads=4)
        self.gru = nn.GRU(input_size=5632, hidden_size=512, num_layers=2, batch_first=True)
        self.final_ln = torch.nn.LayerNorm(out_d, 1e-5, True)
        self.final_adapter = Adapter(out_d, config.adapter_inner_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        suffix_labels: Optional[torch.LongTensor] = None,
        turn1_x1:Optional[torch.FloatTensor] = None,
        turn1_x2: Optional[torch.FloatTensor] = None,
        turn1_text_1: Optional[torch.LongTensor] = None,
        turn1_text_2:Optional[torch.LongTensor] = None,
        turn2_x1:Optional[torch.FloatTensor] = None,
        turn2_x2: Optional[torch.FloatTensor] = None,
        turn2_text_1: Optional[torch.LongTensor] = None,
        turn2_text_2:Optional[torch.LongTensor] = None,
        turn3_x1:Optional[torch.FloatTensor] = None,
        turn3_x2: Optional[torch.FloatTensor] = None,
        turn3_text_1: Optional[torch.LongTensor] = None,
        turn3_text_2:Optional[torch.LongTensor] = None,
        turn4_x1:Optional[torch.FloatTensor] = None,
        turn4_x2: Optional[torch.FloatTensor] = None,
        turn4_text_1: Optional[torch.LongTensor] = None,
        turn4_text_2:Optional[torch.LongTensor] = None,
        turn5_x1:Optional[torch.FloatTensor] = None,
        turn5_x2: Optional[torch.FloatTensor] = None,
        turn5_text_1: Optional[torch.LongTensor] = None,
        turn5_text_2:Optional[torch.LongTensor] = None,
    ):

        turn_input_all =[]
        text_input_all =[]  

        for i in range(1, 6):  # 5 turn
            x1 = locals().get(f'turn{i}_x1', None)
            x2 = locals().get(f'turn{i}_x2', None)
            text_1 = locals().get(f'turn{i}_text_1', None)
            text_2 = locals().get(f'turn{i}_text_2', None)

            if x1 is not None:
                x1 = torch.tensor(x1) if isinstance(x1, list) else x1
                x1 = x1.to('cuda').squeeze(dim=0)

            if x2 is not None:
                x2 = torch.tensor(x2) if isinstance(x2, list) else x2
                x2 = x2.to('cuda').squeeze(dim=0)

            if text_1 is not None:

                text_1 = torch.tensor(text_1) if isinstance(text_1, list) else text_1
                text_1 = text_1.to('cuda').squeeze(dim=0)
                text_1  = self.llama_model.get_input_embeddings()(text_1)
                text_1 = text_1.unsqueeze(0)

            if text_2 is not None:
                text_2 = torch.tensor(text_2) if isinstance(text_2, list) else text_2
                text_2 = text_2.to('cuda').squeeze(dim=0)
                text_2  = self.llama_model.get_input_embeddings()(text_2)
                text_2 = text_2.unsqueeze(0)

            text_embeds = torch.cat([text_1,text_2],dim=1)
            text_reciprocity_embed =self.get_text_reciprocity(text_1,text_2)
            text_reciprocity_embed = text_reciprocity_embed.mean(dim=1, keepdim=True)
        
            speech_embeds = torch.cat([x1,x2],dim=1)
            speech_embeds = speech_embeds.mean(dim=1, keepdim=True)
            speech_reciprocity_embed = self.get_speech_reciprocity(x1,x2)
            speech_reciprocity_embed = speech_reciprocity_embed.mean(dim=1, keepdim=True)

            turn_embeds = torch.cat([speech_embeds,text_reciprocity_embed,speech_reciprocity_embed],dim=-1)
            turn_input_all.append(turn_embeds)
            text_input_all.append(text_embeds)

        turn_embeds = torch.cat(turn_input_all,dim=1)
        text_all_embeds = torch.cat(text_input_all,dim=1)
        self.gru.flatten_parameters()
        turn_embeds, _  = self.gru(turn_embeds)
        self.subsampler_final=self.subsampler_final.to(torch.float32)
        self.subsampler_final=self.subsampler_final.to(turn_embeds.device)
        turn_lengths = torch.tensor([turn_embeds.size(1)] * turn_embeds.size(0), device=turn_embeds.device)  # B x T
        turn_embeds, turn_lengths = self.subsampler(turn_embeds, turn_lengths)
        turn_embeds = turn_embeds.transpose(0,1) 
        self.final_adapter=self.final_adapter.to(torch.float32)
        self.final_adapter=self.final_adapter.to(turn_embeds.device)
        self.final_ln=self.final_ln.to(torch.float32)
        self.final_ln=self.final_ln.to(turn_embeds.device)
        turn_embeds = self.final_adapter(turn_embeds)
        turn_embeds = self.final_ln(turn_embeds)

        turn_embeds_lengths = torch.tensor([turn_embeds.size(1)] * turn_embeds.size(0), device=turn_embeds.device)
        turn_padding_mask = lengths_to_padding_mask(turn_embeds_lengths)  # B x T
        turn_atts = ~turn_padding_mask # B x T
        turn_labels = torch.LongTensor(turn_embeds.size(0), turn_embeds.size(1)).fill_(-100).to(turn_embeds.device)
        text_all_embeds_lengths = torch.tensor([text_all_embeds.size(1)] * text_all_embeds.size(0), device=text_all_embeds.device)
        text_all_padding_mask = lengths_to_padding_mask(text_all_embeds_lengths)  # B x T
        text_all_atts = ~text_all_padding_mask # B x T
        text_all_labels = torch.LongTensor(text_all_embeds.size(0), text_all_embeds.size(1)).fill_(-100).to(text_all_embeds.device)
        ### 2. forward llama
        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        
        inputs_embeds = torch.cat([prefix_embeds, text_all_embeds,turn_embeds, suffix_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, text_all_atts,turn_atts, suffix_attention_mask], dim=1)
        labels = torch.cat([labels, text_all_labels,turn_labels, suffix_labels], dim=1)

        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        
    def get_speech_reciprocity(self, x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        self.speech_cross_attention.to(x1.device)
        self.speech_cross_attention.to(torch.float32)
        speech_reciprocity_embeds1 = self.speech_cross_attention(x2,x1)
        speech_reciprocity_embeds2 = self.speech_cross_attention(x1,x2)
        speech_reciprocity_embeds = torch.cat([speech_reciprocity_embeds1,speech_reciprocity_embeds2], dim=1)

        return speech_reciprocity_embeds
    def get_text_reciprocity(self, x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        self.text_cross_attention.to(x1.device)
        self.text_cross_attention.to(torch.float32)
        text_reciprocity_embeds1 = self.text_cross_attention(x2,x1)
        text_reciprocity_embeds2 = self.text_cross_attention(x1,x2)
        text_reciprocity_embeds = torch.cat([text_reciprocity_embeds1,text_reciprocity_embeds2], dim=1)
        
        return text_reciprocity_embeds

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        suffix_input_ids,
        turn1_x1=None,
        turn1_x2=None,
        turn1_text_1=None,
        turn1_text_2=None,
        turn2_x1=None,
        turn2_x2=None,
        turn2_text_1=None,
        turn2_text_2=None,
        turn3_x1=None,
        turn3_x2=None,
        turn3_text_1=None,
        turn3_text_2=None,
        turn4_x1=None,
        turn4_x2=None,
        turn4_text_1=None,
        turn4_text_2=None,
        turn5_x1=None,
        turn5_x2=None,
        turn5_text_1=None,
        turn5_text_2=None,
        generation_config=None
    ):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)
        turn_embeds_all=[]
        text_embeds_all=[]

        for i in range(1, 6):  # 5 turn
            x1 = locals().get(f'turn{i}_x1', None)
            x2 = locals().get(f'turn{i}_x2', None)
            text_1 = locals().get(f'turn{i}_text_1', None)
            text_2 = locals().get(f'turn{i}_text_2', None)

            if x1 is not None:
                x1 = torch.tensor(x1) if isinstance(x1, list) else x1
                x1 = x1.to('cuda')

            if x2 is not None:
                x2 = torch.tensor(x2) if isinstance(x2, list) else x2
                x2 = x2.to('cuda')

            if text_1 is not None:
                text_1 = torch.tensor(text_1) if isinstance(text_1, list) else text_1
                text_1 = text_1.to('cuda')
                text_1 = self.llama_model.get_input_embeddings()(text_1)
                text_1 = text_1.unsqueeze(0)
            if text_2 is not None:
                text_2 = torch.tensor(text_2) if isinstance(text_2, list) else text_2
                text_2 = text_2.to('cuda')
                text_2 = self.llama_model.get_input_embeddings()(text_2)
                text_2 = text_2.unsqueeze(0)
                
            text_embeds = torch.cat([text_1,text_2],dim=1)
            text_reciprocity_embed =self.get_text_reciprocity(text_1,text_2)
            text_reciprocity_embed = text_reciprocity_embed.mean(dim=1, keepdim=True)
            
            speech_embeds = torch.cat([x1,x2],dim=1)
            speech_embeds = speech_embeds.mean(dim=1, keepdim=True)
            speech_reciprocity_embed = self.get_speech_reciprocity(x1,x2)
            speech_reciprocity_embed = speech_reciprocity_embed.mean(dim=1, keepdim=True)

            turn_embeds = torch.cat([speech_embeds,text_reciprocity_embed,speech_reciprocity_embed],dim=-1)
            turn_embeds_all.append(turn_embeds)
            text_embeds_all.append(text_embeds)
        turn_embeds=torch.cat(turn_embeds_all,dim=1)
        text_all_embeds=torch.cat(text_embeds_all,dim=1)
        self.gru.flatten_parameters()
        turn_embeds, _  = self.gru(turn_embeds)

        self.subsampler_final=self.subsampler_final.to(torch.float32)
        self.subsampler_final=self.subsampler_final.to(turn_embeds.device)
        turn_lengths = torch.tensor([turn_embeds.size(1)] * turn_embeds.size(0), device=turn_embeds.device)  # B x T
        turn_embeds, turn_lengths = self.subsampler(turn_embeds, turn_lengths)
        turn_embeds = turn_embeds.transpose(0,1) 
        self.final_adapter=self.final_adapter.to(torch.float32)
        self.final_adapter=self.final_adapter.to(turn_embeds.device)
        self.final_ln=self.final_ln.to(torch.float32)
        self.final_ln=self.final_ln.to(turn_embeds.device)
        turn_embeds = self.final_adapter(turn_embeds)
        turn_embeds = self.final_ln(turn_embeds)

        turn_embeds_lengths = torch.tensor([turn_embeds.size(1)] * turn_embeds.size(0), device=turn_embeds.device)
        turn_embeds_padding_mask = lengths_to_padding_mask(turn_embeds_lengths)  # B x T
        turn_embeds_atts = ~turn_embeds_padding_mask # B x T
        text_all_embeds_lengths = torch.tensor([text_all_embeds.size(1)] * text_all_embeds.size(0), device=text_all_embeds.device)
        text_all_padding_mask = lengths_to_padding_mask(text_all_embeds_lengths)  # B x T
        text_all_atts = ~text_all_padding_mask # B x T
        
        inputs_embeds.append(text_all_embeds)
        attention_mask.append(text_all_atts)
        inputs_embeds.append(turn_embeds)
        attention_mask.append(turn_embeds_atts)
        
        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0), suffix_embeds.size(1), dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
 
