from torch import nn


class BaseModel(nn.Module):
    def __init__(self, state_shape):
        super(BaseModel, self).__init__()
        c, w, h = state_shape
        self.conv_net = nn.Sequential(nn.Conv2d(c, 32, kernel_size=(8, 8), stride=(4, 4)),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                                      nn.ReLU(),
                                      ).apply(self.init_convs)

        conv1_out_w = self.conv_out_size(w, 8, 4)
        conv1_out_h = self.conv_out_size(h, 8, 4)
        conv2_out_w = self.conv_out_size(conv1_out_w, 4, 2)
        conv2_out_h = self.conv_out_size(conv1_out_h, 4, 2)
        conv3_out_w = self.conv_out_size(conv2_out_w, 3, 1)
        conv3_out_h = self.conv_out_size(conv2_out_h, 3, 1)
        self.flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(self.flatten_size, 512)
        nn.init.orthogonal_(self.fc.weight, 1.)
        self.fc.bias.data.zero_()

    @staticmethod
    def init_convs(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            layer.bias.data.zero_()

    def forward(self, inputs):
        raise NotImplementedError

    def get_qvalues(self, x): # noqa
        raise NotImplementedError

    @staticmethod
    def conv_out_size(input_size, kernel_size, stride=1, padding=0):
        return (input_size + 2 * padding - kernel_size) // stride + 1
