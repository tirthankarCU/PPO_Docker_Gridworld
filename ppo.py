import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing
from simple_gridworld import SimpleGridWorld


class ActorCriticNN(nn.Module):
    def __init__(self,num_inputs,num_actions,hidden_size):
        super().__init__()
        
        self.Critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
        )
        
        self.Actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_actions)
        )
        
        def init_weights(param):
            if isinstance(param,nn.Linear):
                nn.init.normal_(param.weight, mean=0, std=0.1)
                nn.init.constant_(param.bias, 0.1)
        self.apply(init_weights)
        
    def forward(self,x):
        value = self.Critic(x)
        mu_logits = self.Actor(x)
        mu = Categorical(mu_logits)
        return mu, value 
    

def RESETS(envs):
    def reset(env):
        return env.reset()
    pool = multiprocessing.Pool()
    states = pool.map(reset,envs)    
    pool.close()
    pool.join()
    return states 


def STEPS(envs,actions):
    def step(env,action):
        return env.step(action)
    pool = multiprocessing.Pool()
    states, rewards, terminations = pool.map(step,envs,actions)
    pool.close()
    pool.join()
    return states, rewards, terminations


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    N = 10 # number of parallel environments 
    num_steps = 25 # 5x5 grid world
    envs = [SimpleGridWorld() for _ in range(N)]
    num_inputs  = envs[0].input_shape
    num_outputs = envs[0].output_shape
    max_frames = 15000
    frame_idx = 0
    early_stopping = False 
    # Neural Network Hyper params:
    hidden_size      = 256
    lr               = 3e-4
    mini_batch_size  = 5
    ppo_epochs       = 4 
    threshold_reward = -200 # UNK
    # Neural Network
    model = ActorCriticNN(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    while frame_idx < max_frames and not early_stopping:
        log_probsArr = []
        valuesArr    = []
        statesArr    = []
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        states = RESETS(envs)
        for _ in range(num_steps):
            states = torch.FloatTensor(states).to(device)
            dist, value = model(states)

            actions = dist.sample()
            next_states, rewards, done = STEPS(envs,actions.cpu().numpy())

            log_probs = dist.log_prob(actions)
            entropy += dist.entropy().mean()
            
            log_probsArr.append(log_probs)
            valuesArr.append(value)
            rewardsArr.append(torch.FloatTensor(rewards).unsqueeze(1).to(device))
            masksArr.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            statesArr.append(states)
            actionsArr.append(actions)
            
            states = next_states
            frame_idx += 1
            
            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env() for _ in range(10)])
                test_rewards.append(test_reward)
                plot(frame_idx, test_rewards)
                if test_reward > threshold_reward: early_stop = True
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

