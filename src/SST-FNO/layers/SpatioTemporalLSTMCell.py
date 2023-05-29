import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, hidden_dim, x_dim,y_dim):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.input_channels = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = [5,5]
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        #self.device = configs.device
        self.sr_size = 4
        self.patch_size = 1
        self.width = x_dim // self.patch_size // self.sr_size // 2
        self.height = y_dim // self.patch_size // self.sr_size // 2

        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim * 7, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.LayerNorm([hidden_dim * 7, self.height, self.width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.LayerNorm([hidden_dim * 4, self.height, self.width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.LayerNorm([hidden_dim * 3, self.height, self.width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.LayerNorm([hidden_dim, self.height, self.width])
        )
        self.conv_last = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)  # hidden 扩展了7倍
        h_concat = self.conv_h(h_t)  # hidden 扩展了4倍
        m_concat = self.conv_m(m_t)  # hidden 扩展了3倍
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.hidden_dim, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_dim, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


# if __name__ == '__main__':
#     from configs.radar_train_configs import configs

#     parse = configs()
#     configs = parse.parse_args()
#     print(configs.num_hidden)
#     model = SpatioTemporalLSTMCell(64, configs.num_hidden, configs).cuda()
#     print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))