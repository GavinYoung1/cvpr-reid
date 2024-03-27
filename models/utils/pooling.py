import mindspore as ms
from mindspore import ops
from mindspore import nn

class GeMPooling(nn.Cell):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(ops.ones(1) * p)
        self.eps = eps

    def construct(self, x):
        return ops.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p)


class MaxAvgPooling(nn.Cell):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def construct(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return ops.cat((max_f, avg_f), 1)
        