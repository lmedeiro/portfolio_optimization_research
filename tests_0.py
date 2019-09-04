
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
N,D_in,H,D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N,D_in).type(dtype),requires_grad=False)
y = Variable(torch.randn(N ,D_out).type(dtype) ,requires_grad=False)
w1 = Variable(torch.randn(D_in,D_out).type(dtype),requires_grad=True)
learning_rate = 1e-3
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data)
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w1.grad.data.zero_()
