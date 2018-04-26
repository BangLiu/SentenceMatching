import torch
from torch import nn
from torch.autograd import Variable


class SemMatchNet_malstm(torch.nn.Module):
    def __init__(self):
        super(SemMatchNet_malstm, self).__init__()
        self.model_name = 'based on saimese lstm'
        self.depth = 2
        self.width = 4
        self.lstm_hidden_dim = 50
        self.num_layers = 1
        self.step_size0 = 1
        self.step_size1 = 4
        self.step_size2 = 16
        self.embedding_dim = 300
        self.h1 = 300
        self.h2 = 300

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.lstm_hidden_dim,
                            self.num_layers,
                            bidirectional=False)


        self.regressor = nn.Sequential(
           nn.Linear(1+1+1+3*21, self.h2),
            nn.Tanh(),
            nn.Linear(self.h2, 1),
            nn.Sigmoid())

    def forward(self, x):
        xl = x[:, :, 0:self.embedding_dim]
        xr = x[:, :, self.embedding_dim:2 * self.embedding_dim]
        input1l = xl[:, 0: self.step_size0]
        input2_l = xl[:,  self.step_size0: self.step_size0+self.step_size1]
        input3_l = xl[:, self.step_size0+self.step_size1:self.step_size0+self.step_size1+self.step_size2]
        input1r = xr[:, 0:1]
        input2_r = xr[:, 1:5]
        input3_r = xr[:, 5:21]
        fea = x[:, :, 600:603]
        fea = fea.contiguous().view(-1, 21*3)

        batch, seq, embed = input1l.size()
        input1_l = input1l.contiguous().view(seq, batch, embed)
        input1_r = input1r.contiguous().view(seq, batch, embed)
        xl1_o, (xl1_h, xl1_c) = self.lstm(input1_l)
        xr1_o, (xr1_h, xr1_c) = self.lstm(input1_r)
        xl1_h = xl1_h.view(-1, self.lstm_hidden_dim)
        xr1_h = xr1_h.view(-1, self.lstm_hidden_dim)
        # print xl1_h.size()
        out1 = torch.exp(torch.sum(torch.abs(torch.add(xl1_h, -xr1_h)), dim=1, keepdim=True))

        batch, seq, embed = input2_l.size()
        input2_l = input2_l.contiguous().view(seq, batch, embed)
        input2_r = input2_r.contiguous().view(seq, batch, embed)
        xl2_o, (xl2_h, xl2_c) = self.lstm(input2_l)
        xr2_o, (xr2_h, xr2_c) = self.lstm(input2_r)
        xl2_h = xl2_h.view(-1, self.lstm_hidden_dim)
        xr2_h = xr2_h.view(-1, self.lstm_hidden_dim)
        out2 = torch.exp(-torch.sum(torch.abs(torch.add(xl2_h, -xr2_h)), dim=1, keepdim=True))

        batch, seq, embed = input3_l.size()
        input3_l = input3_l.contiguous().view(seq, batch, embed)
        input3_r = input3_r.contiguous().view(seq, batch, embed)
        xl3_o, (xl3_h, xl3_c) = self.lstm(input3_l)
        xr3_o, (xr3_h, xr3_c) = self.lstm(input3_r)
        xl3_h = xl3_h.view(-1, self.lstm_hidden_dim)
        xr3_h = xr3_h.view(-1, self.lstm_hidden_dim)
        out3 = torch.exp(-torch.sum(torch.abs(torch.add(xl3_h, -xr3_h)), dim=1, keepdim=True))

        out = [out1, out2, out3, fea]
        out = torch.cat(out,1)
        pre = self.regressor(out)
        return pre



class SemMatchNet_hcti(torch.nn.Module):

    def __init__(self):
        super(SemMatchNet_hcti, self).__init__()
        self.model_name = 'based on hcti'
        self.depth = 2
        self.width = 4
        self.features = []
        self.num_filter = 300
        self.step_size0 = 1
        self.step_size1 = 4
        self.step_size2 = 16
        self.vector_size = 300
        self.window_size = 1
        self.h1 = 300
        self.h2 = 300
        self.n1 = 10 
        self.n2 = 2
        self.n3 = 2

        self.conv_1d = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(self.window_size, self.vector_size)),
            nn.ReLU())
        self.max0 = nn.MaxPool2d(kernel_size=(self.step_size0 - self.window_size + 1, 1))
        self.max1 = nn.MaxPool2d(kernel_size=(self.step_size1 - self.window_size + 1, 1))
        self.max2 = nn.MaxPool2d(kernel_size=(self.step_size2 - self.window_size + 1, 1))
        self.regressor1 = nn.Sequential(
                 nn.Linear(self.num_filter * 3, self.n1),
                 nn.Softmax())
        self.regressor2 = nn.Sequential(
                 nn.Linear(self.num_filter * 2, self.n2),
                 nn.Softmax())
        self.regressor3 = nn.Sequential(
                 nn.Linear(self.num_filter * 2, self.n3),
                 nn.Softmax())

        self.regressor = nn.Sequential(
           nn.Linear(self.n1+self.n2+self.n3+3*21, self.h2),
            nn.Tanh(),
            nn.Linear(self.h2, 1),
            nn.Sigmoid())

    def forward(self, x):
        xl = x[:, :, 0:300]
        xr = x[:, :, 300:600]
        input1l = xl[:, 0:1]
        input2_l = xl[:, 1:5]
        input3_l = xl[:, 5:21]
        input1r = xr[:, 0:1]
        input2_r = xr[:, 1:5]
        input3_r = xr[:, 5:21]
        fea = x[:, :, 600:603]
        fea = fea.contiguous().view(-1, 21*3)

        batch, seq, embed = input1l.size()
        input1_l = input1l.contiguous().view(batch, 1, seq, embed)
        input1_r = input1r.contiguous().view(batch, 1, seq, embed)
        x_l_1 = self.conv_1d(input1_l)
        x_r_1 = self.conv_1d(input1_r)
        x_l_1 = self.max0(x_l_1)
        x_r_1 = self.max0(x_r_1)
        x_l_1 = x_l_1.view(-1, self.num_filter)
        x_r_1 = x_r_1.view(-1, self.num_filter)
        x_mul_1 = torch.mul(x_l_1, x_r_1)
        x_dif_1 = torch.abs(torch.add(x_l_1, -x_r_1))
        x_dif_12 = torch.abs(torch.add(input1l, -input1r)).view(-1,300)
        x_md_1 = torch.cat([x_mul_1, x_dif_1, x_dif_12], 1)
        out1 = self.regressor1(x_md_1)

        batch, seq, embed = input2_l.size()
        input2_l = input2_l.contiguous().view(batch, 1, seq, embed)
        input2_r = input2_r.contiguous().view(batch, 1, seq, embed)
        x_l_2 = self.conv_1d(input2_l)
        x_r_2 = self.conv_1d(input2_r)
        x_l_2 = self.max1(x_l_2)
        x_r_2 = self.max1(x_r_2)
        x_l_2 = x_l_2.view(-1, self.num_filter)
        x_r_2 = x_r_2.view(-1, self.num_filter)
        x_mul_2 = torch.mul(x_l_2, x_r_2)
        x_dif_2 = torch.abs(torch.add(x_l_2, -x_r_2))
        x_md_2 = torch.cat([x_mul_2, x_dif_2], 1)
        out2 = self.regressor2(x_md_2)

        batch, seq, embed = input3_l.size()
        input3_l = input3_l.contiguous().view(batch, 1, seq, embed)
        input3_r = input3_r.contiguous().view(batch, 1, seq, embed)
        x_l_3 = self.conv_1d(input3_l)
        x_r_3 = self.conv_1d(input3_r)
        x_l_3 = self.max2(x_l_3)
        x_r_3 = self.max2(x_r_3)
        x_l_3 = x_l_3.view(-1, self.num_filter)
        x_r_3 = x_r_3.view(-1, self.num_filter)
        x_mul_3 = torch.mul(x_l_3, x_r_3)
        x_dif_3 = torch.abs(torch.add(x_l_3, -x_r_3))
        x_md_3 = torch.cat([x_mul_3, x_dif_3], 1)
        out3 = self.regressor3(x_md_3)
        out = [out1, out2, out3, fea]
        out = torch.cat(out,1)
        pre = self.regressor(out)
        return pre
