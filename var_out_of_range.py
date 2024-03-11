import torch
import numpy
import matplotlib.pyplot as plt
################### Simple NN example by hand: sigma(Wx+b)C #########################
# https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=bmm#torch.bmm
# https://pytorch.org/docs/stable/generated/torch.matmul.html
import torch.nn as nn

class SimpleFullNN(nn.Module):
    """Maps R^m -> R^n, with overparamterized model."""

    def __init__(self, m, n, d, my_seed = 0):
        super(SimpleFullNN, self).__init__()

        self.sigma = nn.Tanh()
        self.seed = my_seed
        torch.cuda.manual_seed_all(self.seed)

        self.g = nn.Sequential(
            nn.Linear(m, d),
            self.sigma,
            nn.Linear(d, d),
            self.sigma,
            nn.Linear(d, d),
            self.sigma,
            nn.Linear(d, n),
        )

    def forward(self, x):
        out = self.g(torch.transpose(x, 0,1))
        return torch.transpose(out,0,1)
    
    
def my_target_fun(x, sigma = 0.0):
  with torch.no_grad():
    y = torch.cos(2*x) + sigma*torch.randn( x.shape )
  return y

# data input
m = 1
num_items = 50
#x = torch.randn( [num_items,m,1], requires_grad = True )
a=-2.0
b=2.0
#x = (b-a)*torch.rand( [num_items,m,1], requires_grad = True ) + a
#x = torch.rand( [num_items,m,1], requires_grad = True )
x = torch.FloatTensor(num_items,m,1).uniform_(a, b)
print(f"x shape {x.shape}")

n = 1
sigma = 0.01
# function: input x, output y=f(x) : R^m -> R^n
y = my_target_fun(x,sigma)

print(f"y={y} shape {y.shape}, input x {x}")

k = 10
#model = SimpleNN(m,n,k)
model = SimpleFullNN(m,n,k)
print(f"simple NN output {model(x)}")

## training
from tqdm import tqdm
num_nn = 10
bootstrap_size = num_items-10
nn_models = [SimpleFullNN(m,n,k,i) for i in range(0,num_nn)]
nn_data = [x[torch.randint(0, num_items, (bootstrap_size,))] for i in range(0,num_nn)]
nn_labels = [my_target_fun(x) for x in nn_data]

for idx, model in tqdm(enumerate(nn_models)):
  x_bootstrap = nn_data[idx]
  y_bootstrap = nn_labels[idx]
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  num_epochs = 3000
  for epoch in range(num_epochs):
      # forward
      y_hat = model(x_bootstrap)
      # loss
      loss = ((y_hat-y_bootstrap)*(y_hat-y_bootstrap)).sum()
      #if epoch%100==0:
        #print(f'Epoch: {epoch} | Loss: {loss.detach()}| Prediction value: {y_hat} at point %.2f | x {x}')
        #print(f"Epoch: {epoch} | Loss: {loss.detach()}")
      optimizer.zero_grad()
      #backward step
      loss.backward()
      ## update model params
      optimizer.step()

import numpy as np
import matplotlib.pyplot as plt
n_step = 1000
x_input = torch.linspace(-5,5, steps = n_step)
var_models = torch.zeros((len(nn_models),x_input.shape[0]))
fig, axs = plt.subplots(1, 2, figsize=(12,5))

plt.rcParams.update({'font.size': 20})
for idx, model in enumerate(nn_models):
  y_out = torch.tensor([model(torch.reshape(x_tmp, (1,1))).detach().numpy() for x_tmp in x_input ]).reshape(n_step,1)
  var_models[idx,:]=y_out.T
  axs[0].plot(x_input.reshape(n_step,1),y_out)
  axs[0].scatter(x.detach().reshape(num_items,1).numpy(), y.reshape(num_items,1).numpy(), marker="x", alpha = 0.8, c = "#000000",zorder =10)
  #axs[0].set_title("Over")
  axs[0].set(xlabel='$x$', ylabel='$\Psi_i(x)$')


axs[1].plot(x_input.reshape(n_step,1),var_models.var(axis=0))
axs[1].set(xlabel='$x$', ylabel='Var[$\Psi_i(x)$]')
axs[1].set_yscale('log')
plt.tight_layout()
fig.savefig("variance_range.pdf")



