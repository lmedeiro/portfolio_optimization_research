
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
N,D_in,D_out = 64, 1000, 1
x = Variable(torch.randn(N,D_in).type(dtype),requires_grad=False)
y = Variable(torch.randn(N ,D_out).type(dtype) ,requires_grad=False)



class SimpleRegressor(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleRegressor, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out, bias=False)# can also use bilinear, or redefine own transformation


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y_pred = self.linear1(x)
        return y_pred

learning_rate = 1e-2
stopping_criteria = 1e-4
model = SimpleRegressor(D_in, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
for t in range(2000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.data)
    if loss.data <= stopping_criteria:
        print("weight: {}".format(model.weights))
        print('Finished')
        break
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
