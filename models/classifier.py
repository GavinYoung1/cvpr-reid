import mindspore as ms
from mindspore import ops
from mindspore import nn


__all__ = ['Classifier', 'NormalizedClassifier']


class Classifier(nn.Cell):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def construct(self, x):
        y = self.classifier(x)

        return y
        

class NormalizedClassifier(nn.Cell):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = ms.Parameter(ops.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def construct(self, x):
        w = self.weight  

        x = ops.normalize(x, p=2, dim=1)
        w = ops.normalize(w, p=2, dim=1)

        return ops.linear(x, w)



