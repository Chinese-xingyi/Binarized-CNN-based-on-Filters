import torch.nn as nn
import numpy
from torch.autograd import Variable
import mpmath
import torch

# 输入number必须是绝对值
def comp_nearest_power(number):
    count=0
    if number>1:
        while number>1:
            number/=2; count+=1
        #number<2**n
        return 2**(count-1), 2**count

    else:
        while number<=1:
            number*=2; count+=1
        return 2**(-1*count), 2**(1-count)

# 输入是tensor，求出各个clip的，量化核心（2的几次幂）
# 每个clip负责处理输入特征图，卷出一张特征图
def comp_clip_power(tensor):
    len_clip = len(tensor)
    power_list = list(range(len_clip))

    ind = 0
    for i in tensor:
        tmp_mean = torch.mean(i.abs())
        _, tmp_power = comp_nearest_power(tmp_mean)
        power_list[ind] = tmp_power
        ind+=1
    return power_list

# 把tensor，变为power
def bin_tensor(tensor, power_list):
    tensor = tensor.clone()
    tensor_len = tensor.size(0)

    for i in range(tensor_len):
        tensor[i] = tensor[i].sign() * power_list[i]

    return tensor

def hardtanh_tensor(tensor, floor=-1,ceil=1,  max_ind=0, hd_range=2):
    tensor = tensor.clone()
    tensor_len = tensor.size(0)

    hd_func = nn.Hardtanh(floor * (2 ** (max_ind - 1 + 1)), ceil * (2 ** (max_ind - 1 + 1)))
    tensor[ : int(mpmath.ceil(tensor_len*0.9))] = hd_func(tensor[ :int(mpmath.ceil(tensor_len*0.9))])

    hd_func = nn.Hardtanh(floor * (2 ** (max_ind - 2 + 1)), ceil * (2 ** (max_ind - 2 + 1)))
    tensor[int(mpmath.ceil(tensor_len*0.9)) : ] = hd_func(tensor[int(mpmath.ceil(tensor_len*0.9)) : ])
    '''for i in range(tensor_len):
        hd_func = nn.Hardtanh(floor / (2 ** (max_ind - (i%hd_range) + 1)), ceil / (2 ** (max_ind - (i%hd_range) + 1)))
        tensor[i] = hd_func(tensor[i])'''

    return tensor

#def a specific initial funciton for the specific value arrange
def init_tensor(tensor, max_ind=0, init_range=2):
    tensor = tensor.clone()
    tensor_len = tensor.size(0)

    tensor[:int(mpmath.ceil(0.9 * tensor_len))] = tensor[:int(mpmath.ceil(0.9*tensor_len))] * (
            2 ** (max_ind - 1 + 1))

    tensor[int(mpmath.ceil(0.9*tensor_len)): ] = tensor[int(
            mpmath.ceil(0.9*tensor_len)): ] * (2 ** (max_ind -2+ 1))

    return tensor

# def  a function to specific initial a model
def init_model(model, max_ind=1, init_range=2):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.copy_(init_tensor(m.weight.data, max_ind, init_range))


def clip_tensor(tensor, power_list):
    # clip 某层的tensor
    tensor_len = len(tensor)
    abs_tensor = tensor.abs()
    for index in range(tensor_len):
        # 范围为左右临近幂
        # 小于左的，变最小值； 大于右临近值的变最大值
        ind1 = abs_tensor<power_list[index]/2
        ind2 = abs_tensor>power_list[index]*2
        all_1_ind = ind1*0+1

        ind_keep = all_1_ind-ind1-ind2

        # ind1是指明小于左边界值的位置的，ind2指明右边界值位置；ind_keep是介于边界值中间，不需要调整的

    cliped_tensor = tensor*ind_keep.float().cuda() + (power_list[index]/2)*(ind1.float().cuda()*tensor).sign() + (power_list[index]*2)*(ind2.float().cuda()*tensor).sign()

    return cliped_tensor

def clip_tensor2(tensor, power_list):
    # clip 某层的tensor
    tensor_len = len(tensor)
    abs_tensor = tensor.abs()
    for index in range(tensor_len):
        # 范围为左右临近幂
        # 小于左的，变最小值； 大于右临近值的变最大值
        ind = abs_tensor<=power_list[index]*2

        ind_max = ind*(-1)+1

        # ind1是指明小于左边界值的位置的，ind2指明右边界值位置；ind_keep是介于边界值中间，不需要调整的

    cliped_tensor = tensor*ind.float().cuda() + (power_list[index]*2)*(ind_max.float().cuda()*tensor).sign()

    return cliped_tensor

class BC():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        self.model = model
        self.count = 0
        self.saved_params = []
        self.target_modules = []
        self.power_array = []
        # init_model(model, max_ind, index_range)
        for m in model.modules():
            if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
                self.count += 1
                tmp = m.weight.data.clone()
                # saved 存储的是复制的，应该是原始的参数，复制的地址
                self.saved_params.append(tmp)
                # target应该存储的，实际卷积参数的地址
                self.target_modules.append(m.weight)
                tmp_power_list = comp_clip_power(m.weight.data)
                self.power_array.append(tmp_power_list)

        #print(self.count)
                

    '''def binarization(self):
        self.save_params()
        for index in range(self.count):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())'''

    # 更新，power ，可以多个epochs之后，更新一次
    def comp_power_array(self):
        # self.save_params()

        for index in range(self.count):
            # print('index,index',index)
            tmp_power_list = comp_clip_power(self.target_modules[index].data)
            self.power_array[index]=tmp_power_list

    # 每轮都需要binarized，来用量化后的网络计算前向
    def binarization(self):
        self.save_params()
        for index in range(self.count):
            #print('index,index',index)
            tensor = self.target_modules[index].data
            power_list = self.power_array[index]

            binarized_tensor = bin_tensor(tensor, power_list)

            self.target_modules[index].data.copy_(binarized_tensor)


    def save_params(self):
        for index in range(self.count):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def restore(self):
        for index in range(self.count):
            self.target_modules[index].data.copy_(self.saved_params[index])

    # def clip(self, max_ind, hd_range):
    #     clip_scale=[]
    #     m=nn.Hardtanh(-1, 1)
    #     for index in range(self.count):
    #         clip_scale.append(hardtanh_tensor(Variable(self.target_modules[index].data),max_ind=max_ind, hd_range = hd_range))
    #     for index in range(self.count):
    #         self.target_modules[index].data.copy_(clip_scale[index].data)


    def clip(self):
        for index in range(self.count):
            #print('index,index',index)
            tensor = self.target_modules[index].data
            power_list = self.power_array[index]

            cliped_tensor = clip_tensor2(tensor, power_list)

            self.target_modules[index].data.copy_(cliped_tensor)


