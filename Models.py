import torch
import torch.nn as nn
import math


class ProcessLLD(nn.Module):
    def __init__(self):
        super(ProcessLLD, self).__init__()
        self.prj_lld = nn.Linear(70, 128)
        self.act_lld = nn.ReLU()
        self.lstm_lld = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=False)
        self.attn_lld = Attention2(d_k=128)

    def forward(self, x):
        # processing lld via lstm
        out = self.prj_lld(x.permute(0, 2, 1))
        out = self.lstm_lld(out)[0]
        out = out[:, -1, :]
        out = self.mh_attn_lld(out)
        out = self.attn_lld(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.max_p1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.1)
        self.cnn1 = nn.Sequential(self.conv1, self.conv11, self.bn1, self.act1, self.max_p1)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv22 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.max_p2 = nn.MaxPool2d((2, 1))
        self.dropout2 = nn.Dropout(0.1)
        self.cnn2 = nn.Sequential(self.conv2, self.conv22, self.bn2, self.act2, self.max_p2)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv33 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.max_p3 = nn.MaxPool2d((2, 1))
        self.dropout3 = nn.Dropout(0.1)
        self.cnn3 = nn.Sequential(self.conv3, self.conv33, self.bn3, self.act3, self.max_p3)

        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv44 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.max_p4 = nn.MaxPool2d((2, 1))
        self.dropout4 = nn.Dropout(0.1)
        self.cnn4 = nn.Sequential(self.conv4, self.conv44, self.bn4, self.act4)

        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv55 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        self.max_p5 = nn.MaxPool2d((2, 1))
        self.dropout5 = nn.Dropout(0.1)
        self.cnn5 = nn.Sequential(self.conv5, self.conv55, self.bn5, self.act5)

        self.skip13 = nn.Conv2d(32, 128, (1, 1))
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        self.max_p13 = nn.MaxPool2d((4, 1))
        self.skip1 = nn.Sequential(self.skip13, self.bn13, self.act13, self.max_p13)

        self.skip35 = nn.Conv2d(128, 512, (1, 1))
        self.bn35 = nn.BatchNorm2d(512)
        self.act35 = nn.ReLU()
        self.max_p35 = nn.MaxPool2d((2, 1))
        self.skip2 = nn.Sequential(self.skip35, self.bn35, self.act35)

    def forward(self, x):
        out = self.cnn1(x)
        skip1 = self.skip1(out)
        out = self.cnn3(self.cnn2(out))
        out = out + skip1
        skip2 = self.skip2(out)
        out = self.cnn5(self.cnn4(out))
        out = out + skip2
        return out


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x: [bs, num_trunks, ch, F, T_trunk]
        num_trunks, ch, fmel, t_trunk = x.size()[-4:]
        x = x.contiguous().view(-1, ch, fmel, t_trunk)
        out = self.module(x)
        ch, fmel, t_trunk = out.size()[-3:]
        out = out.contiguous().view(-1, num_trunks, ch, fmel, t_trunk)
        return out


class DistributedCNN(nn.Module):
    def __init__(self):
        super(DistributedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.max_p1 = nn.MaxPool2d((2, 2))
        self.conv1, self.conv11, self.bn1, self.act1, self.max_p1 = [TimeDistributed(m) for m in [self.conv1, self.conv11, self.bn1, self.act1, self.max_p1]]
        self.cnn1 = nn.Sequential(self.conv1, self.conv11, self.bn1, self.act1, self.max_p1)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv22 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.max_p2 = nn.MaxPool2d((2, 1))
        self.conv2, self.conv22, self.bn2, self.act2, self.max_p2 = [TimeDistributed(m) for m in [self.conv2, self.conv22, self.bn2, self.act2, self.max_p2]]
        self.cnn2 = nn.Sequential(self.conv2, self.conv22, self.bn2, self.act2, self.max_p2)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv33 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.max_p3 = nn.MaxPool2d((2, 1))
        self.conv3, self.conv33, self.bn3, self.act3, self.max_p3 = [TimeDistributed(m) for m in [self.conv3, self.conv33, self.bn3, self.act3, self.max_p3]]
        self.cnn3 = nn.Sequential(self.conv3, self.conv33, self.bn3, self.act3, self.max_p3)

        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv44 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.max_p4 = nn.MaxPool2d((2, 1))
        self.conv4, self.conv44, self.bn4, self.act4 = [TimeDistributed(m) for m in [self.conv4, self.conv44, self.bn4, self.act4]]
        self.cnn4 = nn.Sequential(self.conv4, self.conv44, self.bn4, self.act4)

        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv55 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        self.max_p5 = nn.MaxPool2d((2, 1))
        self.conv5, self.conv55, self.bn5, self.act5 = [TimeDistributed(m) for m in [self.conv5, self.conv55, self.bn5, self.act5]]
        self.cnn5 = nn.Sequential(self.conv5, self.conv55, self.bn5, self.act5)

        self.skip13 = nn.Conv2d(32, 128, (1, 1))
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        self.max_p13 = nn.MaxPool2d((4, 1))
        self.skip13, self.bn13, self.act13, self.max_p13 = [TimeDistributed(m) for m in [self.skip13, self.bn13, self.act13, self.max_p13]]
        self.skip1 = nn.Sequential(self.skip13, self.bn13, self.act13, self.max_p13)

        self.skip35 = nn.Conv2d(128, 512, (1, 1))
        self.bn35 = nn.BatchNorm2d(512)
        self.act35 = nn.ReLU()
        self.max_p35 = nn.MaxPool2d((2, 1))
        self.skip35, self.bn35, self.act35 = [TimeDistributed(m) for m in [self.skip35, self.bn35, self.act35]]
        self.skip2 = nn.Sequential(self.skip35, self.bn35, self.act35)

    def forward(self, x):
        out = self.cnn1(x)
        skip1 = self.skip1(out)
        out = self.cnn3(self.cnn2(out))
        out = out + skip1
        skip2 = self.skip2(out)
        out = self.cnn5(self.cnn4(out))
        out = out + skip2
        return out


class PositionalEncoder(nn.Module):
    def __init__(self, max_len, d_model, scale=None):
        super(PositionalEncoder, self).__init__()
        pe_flat = [math.sin(pos / (1e+4 ** (i / d_model))) if i % 2 == 0
                   else math.cos(pos / (1e+4 ** ((i - 1) / d_model)))
                   for pos in range(max_len) for i in range(d_model)]
        pe = torch.reshape(torch.FloatTensor(pe_flat), (max_len, d_model)).cuda()
        self.pe = pe.unsqueeze(0)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(d_model)

    def forward(self, x):
        seq_len = x.shape[1]  # x: [bs, seq_len, embed_dim]
        x += self.pe[:, :seq_len] * self.scale
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=2560, nhead=4, dim_feedforward=128, dropout=0.5, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.pe = PositionalEncoder(max_len=321, d_model=2560)
        self.out_dense = nn.Linear(2560, 128)

    def forward(self, x):
        # bs, T, F
        out = self.pe(x)
        # out = x
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = self.out_dense(out)
        return out


class Attention1(nn.Module):
    def __init__(self):
        super(Attention1, self).__init__()
        self.do = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.do(x)
        d = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d)
        att_weights = self.softmax(scores)
        out = torch.matmul(att_weights, x).sum(1)
        return out


class Attention2(nn.Module):
    def __init__(self, d_k):
        super(Attention2, self).__init__()
        self.W = nn.Parameter(torch.Tensor(d_k, d_k))
        self.u = nn.Parameter(torch.Tensor(d_k, 1))

        nn.init.uniform_(self.W, -0.1, -0.1)
        nn.init.uniform_(self.u, -0.1, -0.1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        u = torch.tanh(torch.matmul(x, self.W))
        scores = torch.matmul(u, self.u)
        att_weights = self.softmax(scores)
        out = torch.sum(x * att_weights, dim=1)
        return out


class SelfAttentionMultihead(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super(SelfAttentionMultihead, self).__init__()
        self.self_attn_multihead = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

    def forward(self, x):
        out = x.permute(1, 0, 2)  # T, bs, E
        out = self.self_attn_multihead(out, out, out)
        out = out[0].permute(1, 0, 2)  # bs, T, E
        out = torch.sum(out, dim=1)
        return out


class CRNN(nn.Module):
    def __init__(self, conv_net='cnn', recurrent_net='lstm_attn'):
        super(CRNN, self).__init__()

        self.conv_net = conv_net
        self.recurrent_net = recurrent_net

        if conv_net == 'td-cnn':
            self.cnn = DistributedCNN()
            self.dense_vor_rnn = nn.Linear(512 * 5 * 4, 512 * 5)
        else:
            self.cnn = CNN()

        if recurrent_net == 'transformer':
            self.rnn = Transformer()
        elif recurrent_net == 'lstm_attn':
            self.rnn = nn.LSTM(512 * 5, 128, num_layers=1, batch_first=True, bidirectional=False)
            # self.attn = Attention2(d_k=128)
            self.mh_attn = SelfAttentionMultihead(d_model=128, num_heads=4)
        elif recurrent_net == 'lstm':
            self.rnn = nn.LSTM(512 * 5, 128, num_layers=1, batch_first=True, bidirectional=False)
        else:
            self.linear = nn.Linear(512 * 5, 128)

        self.fc1 = nn.Linear(128, 64)  # 128 + 32, 64
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.5)   # 0.5
        self.fc2 = nn.Linear(64, 2)
        self.process_lld = ProcessLLD()

    def forward(self, x):   # lld
        # processing spectrum
        out = self.cnn(x)
        if self.conv_net == 'td-cnn':
            bs, trunks, ch, f, t = out.size()
            out = out.view(bs, trunks, -1)
            out = self.dense_vor_rnn(out)
        else:
            bs, ch, f, t = out.size()
            out = out.view(bs, ch * f, t).permute(0, 2, 1)  # bs, T, E

        if self.recurrent_net == 'transformer':
            # T, bs, E
            out = self.rnn(out)
            out = torch.mean(out, dim=0)
        elif self.recurrent_net == 'lstm':
            out = self.rnn(out)[0]
            out = out[:, -1, :]
            # out = self.attn(out)
        elif self.recurrent_net == 'lstm_attn':
            out = self.rnn(out)[0]
            out = self.mh_attn(out)
        else:
            # bs, ts, fea = out.shape
            out = torch.mean(out, dim=1)
            out = self.linear(out)

        # cascade lld output and spectrum output
        # lld_out = self.process_lld(lld)
        # out = torch.cat([out, lld_out], dim=-1)

        out = self.act(self.fc1(out))
        features = out
        out = self.fc2(self.do(out))
        return out, features


if __name__ == '__main__':
    crnn = CRNN(conv_net='cnn', recurrent_net='lstm_attn')
    dummy = torch.zeros(4, 1, 40, 321)
    output, features_ = crnn(dummy)

    print(features_.shape)
    print(output.shape)

