from typing import Sequence, Optional

import torch
from torch import nn, Tensor

from eathformer.baselines.PixelShuffle import PixelShuffle
from earthformer.layers.ReversiblePredictiveModule import ReversiblePredictiveModule
from earthformer.layers.i_RevNet_Block import i_RevNet_Block

__all__ = ["CrevNet"]


class CrevNet(nn.Module):

    # def __init__(self, in_channels: int = 1, channels_list: Optional[Sequence[int]] = None, n_layers: int = 6):
    def __init__(self, num_layers, num_hidden, x_dim,y_dim,enc_in,enc_out):
        super(CrevNet, self).__init__()
        #
        # if channels_list is None:
        #     channels_list = [2, 8, 32]
        self.n = 2
        channels_list = [2, 8, 32] #每次翻4倍 第一个2固定
        self.configs = configs
        self.in_channels = x_dim * y_dim * enc_in
        self.channels_list = channels_list
        self.n_blocks = len(channels_list)
        self.num_layers = num_layers

        self.auto_encoder = nn.ModuleList([])
        for i in range(self.n_blocks):
            self.auto_encoder.append(i_RevNet_Block(channels_list[i]))

        self.rpm = ReversiblePredictiveModule(channels=channels_list[-1], n_layers=self.num_layers,x_dim=x_dim,y_dim=y_dim)

        self.pixel_shuffle = PixelShuffle(self.n)

    # noinspection PyUnboundLocalVariable
    def forward(self, inputs, mask_true) -> Tensor:
        device = inputs.device
        batch, sequence, channel, height, width = inputs.shape

        h = []  # 存储隐藏层
        c = []  # 存储cell记忆
        pred = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.num_layers):
            zero_tensor_h = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                                        width // 2 ** self.n_blocks).to(device)
            zero_tensor_c = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                                        width // 2 ** self.n_blocks).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        m = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                        width // 2 ** self.n_blocks).to(device)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for s in range(self.configs.total_length - 1):

            if s < self.configs.input_length:
                x = inputs[:, s]
            else:
                time_diff = s - self.configs.input_length
                x = mask_true[:, time_diff] * inputs[:, s] + (1 - mask_true[:, time_diff]) * x_pred

            x = self.pixel_shuffle.forward(x)        #[8,4,64,64]
            x = torch.split(x, x.size(1) // 2, dim=1)  #[[8,2,64,64],[8,2,64,64]]

            for i in range(self.n_blocks - 1):
                x = self.auto_encoder[i].forward(x) # [8,2,64,64],[8,2,64,64]]
                x = [self.pixel_shuffle.forward(t) for t in x] #每次运行维度 * (2*2),翻了4倍
            x = self.auto_encoder[-1].forward(x)

            x, h, c, m = self.rpm(x, h, c, m)

            for i in range(self.n_blocks - 1):
                x = self.auto_encoder[-1 - i].inverse(x)
                x = [self.pixel_shuffle.inverse(t) for t in x]

            x = self.auto_encoder[0].inverse(x)

            x = torch.cat(x, dim=1)

            x_pred = self.pixel_shuffle.inverse(x)

            # if s >= sequence:
            #     pred.append(x_pred)
            pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     net = CrevNet(in_channels=1, channels_list=[2, 8, 32, 128]).to("cuda")
#     inputs = torch.ones(2, 10, 1, 128, 128).to("cuda")
#     result = net(inputs, out_len=12)
#     print(result.shape)
#     mse = torch.nn.MSELoss()(result, result)
#     mse.backward()