# Training points selection
# Selects the points that will be used inside the gaussian process
# source: https://github.com/nestordemeure/tabularGP/blob/master/trainset_selection.py

from torch import Tensor, nn
import torch
from fastai.torch_basics import * # for pytorch to tensor conversion

__all__ = ['trainset_of_databunch', 'select_trainset', 'get_worst_element']

#--------------------------------------------------------------------------------------------------
# Distance metric

def _hamming_distances(row:Tensor, data:Tensor):
    "returns a vector with the hamming distance between a row and each row of a dataset"
    if row.dim() == 0: return Tensor([0.0]).to(row.device) # deals with absence of categorial features
    return (row.unsqueeze(dim=0) != data).sum(dim=1)

def _euclidian_distances(row:Tensor, data:Tensor):
    "returns a vector with the euclidian distance between a row and each row of a dataset"
    if row.dim() == 0: return Tensor([0.0]).to(row.device) # deals with absence of continuous features
    return torch.sum((row.unsqueeze(dim=0) - data)**2, dim=1)

#--------------------------------------------------------------------------------------------------
# Selection

def _maximalyDifferentPoints(data_cont:Tensor, data_cat:Tensor, nb_cluster:int):
    """
    returns the given number of indexes such that the associated rows are as far as possible
    according to the hamming distance between categories and, in case of equality, the euclidian distance between continuous columns
    uses a greedy algorithm to quickly get an approximate solution
    """
    # initialize with the first point of the dataset
    indexes = [0]
    row_cat = data_cat[0, ...]
    minimum_distances_cat = _hamming_distances(row_cat, data_cat)
    # we suppose that data_cont is normalized so raw euclidian distance is enough
    row_cont = data_cont[0, ...]
    minimum_distances_cont = _euclidian_distances(row_cont, data_cont)
    for _ in range(nb_cluster - 1):
        # finds the row that maximizes the minimum distances to the existing selections
        # choice is done on cat distance (which has granularity 1) and, in case of equality, cont distance (normalized to be in [0;0.5])
        minimum_distances = minimum_distances_cat + minimum_distances_cont / (2.0 * minimum_distances_cont.max())
        index = torch.argmax(minimum_distances, dim=0)
        indexes.append(index.item())
        # updates distances cont
        row_cont = data_cont[index, ...]
        distances_cont = _euclidian_distances(row_cont, data_cont)
        minimum_distances_cont = torch.min(minimum_distances_cont, distances_cont)
        # update distances cat
        row_cat = data_cat[index, ...]
        distances_cat = _hamming_distances(row_cat, data_cat)
        minimum_distances_cat = torch.min(minimum_distances_cat, distances_cat)
    return torch.LongTensor(indexes)

def trainset_of_databunch(data):
    "takes a databunch and returns a (cat,cont,y) tuple"
    # extracts all the dataset as tensors
    # TODO convert the folowing dataframes into tensors
    data_cat = tensor(data.cats).long()
    data_cont = tensor(data.conts).float()
    # transforms the output into one hot encoding if we are dealing with a classification problem
    is_classification = (data.c > 1) and (len(data.y_names) == 1)
    if is_classification:
        data_y = tensor(data.ys).squeeze().long()
        data_y = nn.functional.one_hot(data_y).float()
    else:
        data_y = tensor(data.ys).float()
    return (data_cat, data_cont, data_y)

def select_trainset(data, nb_points:int, use_random_training_points=False):
    "gets a (cat,cont,y) tuple with the given number of elements"
    # extracts all the dataset as tensors
    (data_cat, data_cont, data_y) = trainset_of_databunch(data)
    # selects training points
    if nb_points >= data_cat.size(0): return (data_cat, data_cont, data_y)
    elif use_random_training_points: indices = torch.arange(0, nb_points)
    else: indices = _maximalyDifferentPoints(data_cont, data_cat, nb_points)
    # assemble the training data
    data_cat = data_cat[indices, ...]
    data_cont = data_cont[indices, ...]
    data_y = data_y[indices, ...]
    return (data_cat, data_cont, data_y)

def get_worst_element(model, dl, loss_func):
    "gets the element, from the given dataloader, with the maximum loss"
    max_loss = None
    max_element = None
    for (xcat,xcont),y in dl:
        out = model(xcat,xcont)
        loss = loss_func(out, y, reduction=None).squeeze()
        id_worst_batch_element = torch.argmin(loss)
        if (max_loss is None) or (loss[id_worst_batch_element] > max_loss):
            max_loss = loss[id_worst_batch_element]
            max_element = xcat[id_worst_batch_element, ...], xcont[id_worst_batch_element, ...], y[id_worst_batch_element, ...]
    return max_element
