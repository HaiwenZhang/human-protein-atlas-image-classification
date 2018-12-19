import torch
import numpy as np


def f1_loss(input, target, eps=1E-8):
    """ Micro averaging f1 loss
    .. math::   
        P_{micro}=\frac{\sum_{i=1}^nTP_i}{\sum_{i=1}^nTP_i + \sum_{i=1}^nFP_i} \\
        R_{micro}=\frac{\sum_{i=1}^nTP_i}{\sum_{i=1}^nTP_i + \sum_{i=1}^nFN_i} \\
        F1_{micro}=\frac{2xP_{micro}xR_{micro}}{P_{micro}+R_{micro}}
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    input = input.float()
    target = target.float()
    
    tp = torch.sum(input * target, dim=1)
    # TP+FP 就是预测值中所有被预测为C的example数量
    precision = tp / (torch.sum(input, dim=1) + eps)
    # TP + FN 就是正式值中被分类为C的exampl数量
    recall = tp / (torch.sum(target, dim=1) + eps)    
    # 调和平均
    f1 = 2 * precision * recall / (precision + recall + eps)
    # 对所有label的F1 loss取均值
    f1 = torch.mean(f1)
    return f1


def test():
    y_pred = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])


    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    py_pred = torch.from_numpy(y_pred)
    py_true = torch.from_numpy(y_true)

    f1 = f1_loss(py_pred, py_true)
    print(f1)

if __name__ == '__main__':
    test()
