import torch.nn as nn
import numpy
from torch.autograd import Variable


class BC():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        self.count = 0
        self.saved_params = []
        self.target_modules = []
        # self.qua_list = qua_list
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.count += 1
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)

    # def init_param(self):
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             m.weight.data.copy_(m.weight.data/2)
    #             #bias 不参与量化，是否要除以2，值得商榷！
    #             m.bias.data.copy_(m.bias.data/2)

    def binarization(self):
        self.save_params()
        for index in range(self.count):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())

    def save_params(self):
        for index in range(self.count):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def restore(self):
        for index in range(self.count):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        clip_scale = []
        m = nn.Hardtanh(-1, 1)
        for index in range(self.count):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.count):
            self.target_modules[index].data.copy_(clip_scale[index].data)





