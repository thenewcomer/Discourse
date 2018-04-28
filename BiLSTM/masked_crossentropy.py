import torch
from torch.nn import functional
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

def similarity_mask(predict_list, target, similarity_matrix):
    similarity_vector = []
    index_list = predict_list
    for i in range(len(index_list)):
        similarity_vector.append(1-similarity_matrix[index_list[i]][int(target[i])])
    if USE_CUDA:
        similarity_vector = Variable(torch.FloatTensor(similarity_vector)).cuda()
    else:
        similarity_vector = Variable(torch.FloatTensor(similarity_vector))
    return similarity_vector

def get_index_from_onehot(one_hot):
    index_list = []
    for i in range(len(one_hot)):
        index = 0
        for value in one_hot[i]:
            if int(value) == 1:
                index_list.append(index)
                break
            else:
                index += 1
    return index_list

def masked_cross_entropy(logits, target, similarity_matrix, label_type):

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=-1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    predict_list = []
    for one_data in logits:
        topkv, topki = torch.topk(one_data.data, 1)
        ni = topki[0]
        predict_list.append(ni)
    if label_type.lower() == "conn":
        similarity_vector = similarity_mask(predict_list, target, similarity_matrix)
        losses = losses * similarity_vector
    loss = losses.sum() / float(logits.size()[0])
    return loss
