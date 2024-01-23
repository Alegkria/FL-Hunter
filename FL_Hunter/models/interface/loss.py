import torch as th
import torch.jit
import torch.nn.functional as func




@torch.jit.script
def focal_loss(preds, labels, gamma: float = 2.):  # 自定义实现聚焦损失函数，目的解决检测中存在的类别不平衡的问题 聚焦因子gamma默认2 用来调整样本权重
                                                   # 使得出现次数较少的类别在训练过程中得到更多的关注，从而缓解类别不平衡问题
    preds = preds.view(-1, preds.size(-1))  # 模型预测的类别概率，形状为[-1, num_classes] -1 表示任意数量的样本，num_classes 表示类别数量
    labels = labels.view(-1, 1)  # 真实的类别标签，形状为[-1, ] 表示每个样本对应的真实类别索引
    preds_logsoft = func.log_softmax(preds, dim=-1)  # log_softmax将概率转化为对数概率
    preds_softmax = th.exp(preds_logsoft)  # softmax应用指数函数消除对数操作得到原始的类别概率
    preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )

    preds_logsoft = preds_logsoft.gather(1, labels)  # 选择真实类别的对数概率
    weights = th.pow((1 - preds_softmax), gamma)  # 计算每个样本的权重 出现次数小的权重大，出现次数大的权重小
    loss = - weights * preds_logsoft  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

    loss = th.sum(loss) / th.sum(weights)
    return loss

def dice_coefficient(labels, preds , smooth=1):
    assert preds.size() == labels.size()
    preds = preds.view(-1, preds.size(-1))  # 模型预测的类别概率，形状为[-1, num_classes] -1 表示任意数量的样本，num_classes 表示类别数量
    labels = labels.view(-1, 1)  # 真实的类别标签，形状为[-1, ] 表示每个样本对应的真实类别索引
    intersection = torch.sum(labels * preds)
    union = torch.sum(labels) + torch.sum(preds)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return torch.mean(dice_score)

def dice_loss(labels, preds):
    return 1-dice_coefficient(labels,preds)


def DSC_loss(labels, preds):
    return (- dice_coefficient(labels, preds))


# @torch.jit.script
def KL_classification_loss(       # 二分类任务的损失函数，用于训练二分类模型 在样本类别不平衡的情况下，通过引入样本权重和聚焦因子来调整损失函数，以便更加关注关键类别，提高模型性能
        prob: th.Tensor, label: th.Tensor, gamma: float = 0.,   # prob是模型的输出，即预测的类别概率，label真实的标签 形状为 [batch_size, num_classes]  gamma: 聚焦因子，默认为 0.0
        normal_fault_weight: float = 1e-1,                      # batch_size 表示批次大小，num_classes 表示类别数量。标签应为 one-hot 编码，每个样本只有一个类别为 1，其他为 0
        target_node_weight: float = 0.5,   # normal（非故障类别）的权重，默认为 0.1  target关键节点的权重，默认为 0.5
) -> th.Tensor:
    assert prob.size() == label.size()
    device = prob.device
    target_id = th.argmax(label, dim=-1)
    prob_softmax = th.sigmoid(prob)  # 利用th.sigmoid()函数，将prob转化为预测的概率值，以便于后续计算
    weights = th.pow(
        th.abs(label - prob_softmax), gamma
    ) * (
                      th.max(label, dim=-1, keepdim=True).values + th.full_like(label, fill_value=normal_fault_weight)
              )     #计算样本权重
    if len(prob.size()) == 1:  #对于单样本情况
        weights[target_id] *= prob.size()[-1] * target_node_weight
    else:   # 多样本情况 两种情况调整权重的方式不同
        weights[th.arange(len(target_id), device=device), target_id] *= prob.size()[-1] * target_node_weight
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    loss = kl_loss(func.log_softmax(prob, dim=1), label.float())
    return th.sum(loss * weights) / th.prod(th.tensor(weights.size()))

