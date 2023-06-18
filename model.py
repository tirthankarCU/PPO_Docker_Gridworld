import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np 

ip_layer=24
class NNModel(nn.Module):
    def __init__(self):
        global ip_layer
        super(NNModel,self).__init__()
        self.num_actions=4
        self.ip_layer=ip_layer
        self.fc=nn.Sequential(
            nn.Linear(self.ip_layer,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,self.num_actions),
        )
    def forward(self,x):
        return self.fc(x)

epochA=0
def loss_func(yp,y,actions):
    m=len(y)
    sum=0
    for iter in range(m):
        sum+=(yp[iter][actions[iter]]-y[iter])**2
    sum/=m 
    return sum 

def predict(model,STATE,device):
    model.eval()
    with torch.no_grad():
        for id,s in enumerate(STATE):
            sz=len(s)
            sz_left=ip_layer-sz
            addendum=[-1]*sz_left
            STATE[id]=list(s)+addendum
        Xtr=torch.tensor(STATE)
        Xtr=Xtr.to(device)
        return model(Xtr.float())       
     
def train(model,reward_true,STATE,ACTION,device,optim,verbose=False):
    global epochA,ip_layer
    model.train()
    with torch.no_grad():
        for id,s in enumerate(STATE):
            sz=len(s)
            sz_left=ip_layer-sz
            addendum=[-1]*sz_left
            STATE[id]=list(s)+addendum
    Xtr=torch.tensor(STATE)
    Xtr=Xtr.to(device)
    reward_true=torch.tensor(reward_true)
    Q=model(Xtr.float())
    loss_dqn=-1
    '''
    Training >>>
    '''
    optim.zero_grad()
    loss_dqn=loss_func(Q,reward_true,ACTION)
    loss_dqn.backward()
    optim.step()
    '''
    Training <<<
    '''
    if epochA% 50 == 0 or verbose:
        print(f'Train Epoch:{epochA} DQN_Loss:{loss_dqn}')
    epochA+=1
    
if __name__=='__main__':
    dbg1=True
    num_actions=4
    epochs=20     
    if dbg1==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net=NNModel().to(device)
        net_t=NNModel().to(device)
        for param in net_t.parameters():
            param.data.fill_(0)
        optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
        m=256 #number of training examples.
        Xtr=np.random.randn(m,128)
        Xtr=torch.from_numpy(Xtr)
        Xtr=Xtr.to(device)        
        Actions=np.random.randint(0,num_actions,size=(m,1))
        with torch.no_grad():
            Ytr,Ytr_indx=torch.max(net_t(Xtr.float()),dim=1) # np.random.randn(m) # r+Y*(max_a q(s_{t+1},a)) 
            Ytr=Ytr.to(device)
        net.train()
        for iter in range(epochs):
            optimizer.zero_grad()
            yp=net(Xtr.float())
            loss=loss_func(yp,Ytr,Actions)
            loss.backward()
            optimizer.step()
            if iter%5==0:
                print(f'LOSS {loss}')