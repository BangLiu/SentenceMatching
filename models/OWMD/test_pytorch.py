# -*-coding:utf-8-*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from graphviz import Digraph


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(
                    var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (
                    param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        y = []
        for i in range(1):
            print("i is:")
            print(i)
            print(x)
            mask1 = torch.from_numpy(
                np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).byte()
            mask2 = torch.from_numpy(
                np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8)).byte()
            mask3 = torch.from_numpy(
                np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.uint8)).byte()
            x1 = self.fc1(torch.masked_select(x, mask1))
            print x1
            x2 = self.fc1(torch.masked_select(x, mask2))
            print x2
            x3 = self.fc1(torch.masked_select(x, mask3))
            print x3
            x_out = torch.cat((x1, x2, x3), 0)
            print x_out

        return x_out


net = Net()
print net

params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())

input = Variable(torch.from_numpy(
    np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.float32)))
print input
out = net(input)
print(out)
print(net.fc1.weight)
print(net.fc1.bias)

g = make_dot(out, net.state_dict())
g.view()
