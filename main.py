import torch, torch.nn as nn, numpy as np
from collections import deque
import random
 
class DQN(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,out_d))
    def forward(self,x): return self.net(x)
 
class ReplayBuffer:
    def __init__(self, cap=10000): self.buf=deque(maxlen=cap)
    def push(self,*t): self.buf.append(t)
    def sample(self, n):
        b=random.sample(self.buf,n); s,a,r,s_,d=zip(*b)
        return (torch.FloatTensor(np.array(s)),torch.LongTensor(a),
                torch.FloatTensor(r),torch.FloatTensor(np.array(s_)),torch.FloatTensor(d))
    def __len__(self): return len(self.buf)
 
class DQNAgent:
    def __init__(self, sdim, adim, lr=1e-3, gamma=0.99, eps=1.0):
        self.adim=adim; self.gamma=gamma; self.eps=eps; self.eps_min=0.01; self.eps_decay=0.995
        self.policy=DQN(sdim,adim); self.target=DQN(sdim,adim)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt=torch.optim.Adam(self.policy.parameters(),lr); self.buf=ReplayBuffer()
    def act(self, s):
        return random.randrange(self.adim) if random.random()<self.eps else self.policy(torch.FloatTensor(s)).argmax().item()
    def step(self, s, a, r, s_, d):
        self.buf.push(s,a,r,s_,float(d))
        if len(self.buf)<64: return None
        s,a,r,s_,d=self.buf.sample(64)
        q=self.policy(s).gather(1,a.unsqueeze(1)).squeeze()
        with torch.no_grad(): qt=r+self.gamma*self.target(s_).max(1)[0]*(1-d)
        loss=nn.MSELoss()(q,qt); self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps=max(self.eps_min,self.eps*self.eps_decay); return loss.item()
    def sync_target(self): self.target.load_state_dict(self.policy.state_dict())
 
ag=DQNAgent(4,2)
for step in range(500):
    s=np.random.rand(4); a=ag.act(s)
    s_=np.random.rand(4); r=np.random.rand(); d=False
    loss=ag.step(s,a,r,s_,d)
    if step%100==0 and loss: print(f"Step {step}: loss={loss:.4f}, eps={ag.eps:.3f}")
