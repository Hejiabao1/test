import torch.nn as nn
from collections import OrderedDict
class Net(nn.Module):
    def __init__(self, hidden_size=32, seq_len=750, is_bidirectional=True, network="LSTM",channels_size=3):#表隐藏层的大小、序列长度、是否使用双向RNN以及选择的网络类型（默认为LSTM）
        super(Net, self).__init__()#调用了父类nn.Module的初始化方法，确保正确地初始化了SleepNet类。
        self.padding_edf = {
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }#四个键值对，分别对应不同的卷积和池化层的填充值。这些值用于指定每个层的填充大小。
        first_filter_size = int(100 / 2.0)#定义第一个滤波器的大小
        first_filter_stride = int(100 / 20.0)#定义第一个滤波器的步长

        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1 这一行使用ConstantPad1d对输入进行一维常数填充，填充大小由padding_edf['conv1']指定
            nn.Sequential(OrderedDict([
                ('conv1',
                 nn.Conv1d(in_channels=channels_size, out_channels=150, kernel_size=first_filter_size, stride=first_filter_stride,
                           bias=False))#定义了一个一维卷积层Conv1d，指定输入通道数为2，输出通道数为128，滤波器大小为first_filter_size，步长为first_filter_stride，且不使用偏置。
            ])),
            nn.BatchNorm1d(num_features=150, eps=0.001, momentum=0.01),#定义了一个一维批量归一化层，指定特征数为128，设置了批量归一化的参数eps和momentum
            nn.ReLU(inplace=True),#定义了一个ReLU激活函数层，inplace=True表示原地操作，节省内存
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1   这一行使用ConstantPad1d对输入进行一维常数填充，填充大小由padding_edf['max_pool1']指定
            nn.MaxPool1d(kernel_size=8, stride=8),#定义了一个一维最大池化层，指定池化核大小为8，步长为8
            nn.Dropout(p=0.6),#定义了一个Dropout层，设置了丢弃概率为0.5。
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=150, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3', nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=150, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=150, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),#将连续的维度范围展开为张量
            nn.Dropout(p=0.6),
        )

        if network == "LSTM":
            self.rnn = nn.LSTM(input_size=channels_size, hidden_size=hidden_size, num_layers=1, batch_first=True,
                               bidirectional=is_bidirectional)#创建一个LSTM模型，指定输入特征大小为2048，隐藏单元大小为hidden_size，层数为1，batch维度在第一维
        elif network == "GRU":
            self.rnn = nn.GRU(input_size=channels_size, hidden_size=hidden_size, num_layers=1, batch_first=True,
                              bidirectional=is_bidirectional)
        elif network == "Attention":
            # encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
            # 这里其实用transformer_encoder也可以，但是如果叠加太多层，会跑不起来，而且发现使用attention的效果很一般
            self.qtrans = nn.Sequential(nn.Linear(channels_size, 256), nn.ReLU(inplace=True))
            self.ktrans = nn.Sequential(nn.Linear(channels_size, 256), nn.ReLU(inplace=True))
            self.vtrans = nn.Sequential(nn.Linear(channels_size, 256), nn.ReLU(inplace=True))
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.rnn_dropout = nn.Dropout(p=0.6)#创建一个Dropout层，设置丢弃概率为0.5
        if network != "Attention":
            # 使用双向RNN的时候，会有前向和后向的两个输出进行拼接作为最后的输出，因此大小需要×2
            self.fc = nn.Linear(hidden_size * 2, 1) if is_bidirectional else nn.Linear(hidden_size, 1)
        else:
            self.fc = nn.Linear(256, 1)


        self.is_bidirectional = is_bidirectional
        self.hidden_size = hidden_size
        self.network = network
        self.seq_len = seq_len#将参数保存到模型中，包括是否双向、隐藏单元大小、网络类型和序列长度
        self.expand_dim = nn.Linear(750,  channels_size* 750)  # 假设您希望在每个时间步上增加到2个特征维度

    def forward(self, x):#定义了一个前向传播函数，该函数接收输入x作为参数。
        x = x.reshape(-1, 3, 750)#将输入xreshape成形状为(-1, 1, 3000)的张量，将其形状调整为(batch_size, 2, 3000)。
        x = self.cnn(x)
        # x = self.last_conv_layer(x)#将调整后的输入x通过卷积神经网络（CNN）模块self.cnn进行处理
        # print("After CNN processing, the tensor shape is:", x.shape)
        x = x.unsqueeze(-1)  # 假设原始为(batch_size, seq_len)，增加维度变为(batch_size, seq_len, 1)
        x = self.expand_dim(x.view(-1, 750))  # 扩展特征维度，需要先展平序列维度
        x = x.view(x.shape[0], self.seq_len, -1)  # 重塑为(batch_size, seq_len, features)

        if self.network != "Attention":
            x, _ = self.rnn(x)#将调整后的张量x通过RNN模型self.rnn进行处理，返回输出张量x和一个不需要的变量_。
            if self.is_bidirectional:
                # 提取最后一个时间步的输出，并考虑双向（合并前向和后向的隐藏状态）
                x = x[:, -1, :self.hidden_size * 2]  # 或者 x[:, -1, :] 如果你只想取其中一个方向的输出
            else:
                x = x[:, -1, :]  # 取序列的最后一时刻的输出

                # 动态地基于当前批次大小重塑张量形状
            x = x.reshape(x.shape[0], -1)  # 这里保留了原始的batch_size维度，确保输出维度为(batch_size, hidden_size * num_directions)


        x = self.rnn_dropout(x)
        x = self.fc(x)  # 确保fc层的输入维度与处理后的x匹配

        return x



# if __name__ == '__main__':
#     from torchsummaryX import summary
#     #
#     # model = SleepNet(network="LSTM", seq_len=750)
#     # # torch可以自己计算出隐单元输入的大小，这里就不自己预先定义了
#     # # state = (torch.zeros(size=(1, 15, 128)),
#     # #          torch.zeros(size=(1, 15, 128)))
#     # # torch
#     # # state = (state[0].to(self.device), state[1].to(self.device))
#     # y = model(torch.randn(size=(20 * 32, 1, 3000)))
#     import torch
#
#     # 实例化模型，这里我们使用LSTM为例
#     model = Net(hidden_size=32, seq_len=750, is_bidirectional=True, network="LSTM")
#
#     # 创建随机模拟数据，假设有batch_size=4的样本，每个样本有2个通道，序列长度为3000
#     batch_size = 4
#     input_data = torch.randn(batch_size, 3, 750)
#
#     # 将数据送入模型进行前向传播
#     output = model(input_data)
#
#     # 输出模型的前向传播结果
#     print("Model output shape:", output.shape)
#
#     # 检查输出是否符合预期
#     assert output.shape == (batch_size, 1), "Output shape does not match expected shape"
#     print("Model runs successfully!")

# import torch
# from torch.autograd import Variable
#
#
# def test_model(model):
#     # 假设输入数据为随机生成的张量，形状为(batch_size, channels, seq_len)
#     batch_size = 5  # 示例中的批次大小
#     input_channels = 2  # 输入通道数，与模型设计匹配
#     seq_length = model.seq_len  # 序列长度，从模型中获取
#
#     # 生成随机输入数据
#     input_data = torch.randn(batch_size, input_channels, seq_length)
#
#     # 将输入数据包裹在Variable中，以便自动求导（如果需要的话）
#     input_data = Variable(input_data)
#
#     # 通过模型进行前向传播
#     output = model(input_data)
#
#     # 打印输出形状
#     print(f"Model output shape: {output.shape}")
#
#     # 打印部分输出数值（例如，前几个样本的第一个元素）
#     print(f"Sample outputs: {output[:3].detach().numpy()}")
#
#
# # 实例化模型（以LSTM为例）
# model_instance = Net(hidden_size=32, seq_len=750, is_bidirectional=False, network="LSTM")
#
# # 测试模型
# test_model(model_instance)





#     # summary(model, torch.randn(size=(20 * 32, 1, 3000)))  # batch_size = 20, seq_len = 32

