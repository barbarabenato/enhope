import torch
import torch.nn as nn


def convert_to_torch_data(X,y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y = torch.reshape(y, (y.size(0),1))

    return X, y

class enHOPE(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, F):
        super(enHOPE, self).__init__()
        self.O = 2 # param fixed by the paper
        self.F = F
        self.m = hiddenSize

        self.w = torch.randn(F, hiddenSize)
        self.b = torch.randn(hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize, bias=False) # V param
        torch.nn.init.normal_(self.fc2.weight.data)
        self.C = torch.randn(inputSize+1, F)

    def forward(self, x):
        # adding extra dimension to x
        aux = torch.ones((x.shape[0],1))
        x = torch.cat((x,aux), dim=1)
        
        sum = torch.zeros((x.shape[0], self.m))
        for i in range(self.F):
            # calculating the polynomial expansion
            pe = torch.matmul(x, self.C[:, i]) ** self.O  # (C.T * X')^O (n, 1)
            prod = torch.ger(pe, self.w[i, :])  # outer product (n, m)
            sum = sum + prod
        x = sum + self.b
        x = torch.sigmoid(x)
        x = self.fc2(x)  # (n, 2)
        return x

    def computeP0(self, y, ind_e):
        n = y.shape[0]
        p = torch.zeros((len(ind_e),n))
        y = y+1

        for j, e in enumerate(ind_e):
            for i in range(n):
                if y[e] == y[i]:
                    p[j, i] = 1.0

        p = p / p.sum(dim=1, keepdim=True)
        return p

    def enHOPE_optim(self, x, y, ind_e, p):
        ye = y[ind_e]
        n = x.shape[0]
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        # for loss computation with BFGRS
        def closure():
            if torch.is_grad_enabled():
                self.optim.zero_grad()
            
            # Forward data
            x_hat = self.forward(x)

            q = torch.zeros(len(ind_e),n)
          
            for i in range(n):
                for j, e in enumerate(ind_e):
                    dij = torch.norm(x_hat[i] - x_hat[e], p=2)
                    q[j, i] = 1 / (1 + dij)
                   
            q = q / q.sum(dim=1, keepdim=True)
            
            loss = criterion(q.log(), p)

            if loss.requires_grad:
                loss.backward()
            
            return loss
        
        self.optim.step(closure)
        
        with torch.no_grad():  
            loss2 = closure()
        return loss2

    def train(self, x, y, epochs, e):
        p = self.computeP0(y, e)
    
        for epoch in range(epochs):
            running_loss = self.enHOPE_optim(x, y, e, p)
            print("[%d] loss: %.4f" % (epoch+1, running_loss.detach().numpy()))

    def predict(self, X):
        X = torch.Tensor(X)
        with torch.no_grad():
            return self(X).detach().numpy().squeeze()


