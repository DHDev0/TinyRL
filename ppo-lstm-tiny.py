from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
# from tinygrad.nn import Linear
from tinygrad.helpers import dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.ops import _Device as Device
import numpy as np
import gymnasium as gym
import time
# Device.DEFAULT = "GPU"

#Hyperparameters
learning_rate  = 0.0005
gamma          = 0.98
lmbda          = 0.95
eps_clip       = 0.1
K_epoch        = 2
T_horizon      = 20 #number of step before training (one step is an action input into the game)
episode        = 2000 #number of episode is the ammount of completed game
print_interval = 10 #every X epoch
c_type = dtypes.float32


def sgn(x): return x/x.pow(2.).sqrt()#.abs()
def inf_to(x,y): return (1-sgn(x-y))/2
def sup_to(x,y): return (1+sgn(x-y))/2
def smooth_l1_loss(predicted_value, target_value, reduction='mean', beta=1.0):
  elementwise_difference = predicted_value - target_value
  # mask = inf_to(elementwise_difference,beta).detach()
  # mask = (elementwise_difference < beta).detach()
  mask = inf_to(elementwise_difference , beta).detach()
  loss = (mask * (0.5*elementwise_difference**2 / beta)) + ((-1*(mask-1)) * (elementwise_difference.abs() - 0.5*beta))
  # loss = Tensor.where(elementwise_difference < beta, 0.5 * elementwise_difference**2 / beta, elementwise_difference - 0.5 * beta)  
  return loss
  # if reduction == "mean": 
  #   return loss.mean()
  # return loss.sum()

import math
class LSTMCell:
  def __init__(self, input_size, hidden_size, dropout):
    self.dropout = dropout
    bound = math.sqrt(1 / hidden_size)
    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size, low=-bound, high=bound, dtype=c_type)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size, low=-bound, high=bound, dtype=c_type)
    self.bias_ih = Tensor.ones(hidden_size, dtype=c_type).cat(Tensor.zeros(hidden_size * 3,dtype=c_type),dim=0)
    self.bias_hh = Tensor.zeros(hidden_size * 4, dtype=c_type)

  def __call__(self, x, hc):
    # print(x.shape,hc.shape,hc[:x.shape[0]].shape)
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = (f * hc[x.shape[0]:]) + (i * g)
    h = (o * c.tanh()).dropout(self.dropout)
    return Tensor.cat(h, c)


class LSTM:
  def __init__(self, input_size, hidden_size, layers=1, dropout=0):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers
    self.cells = [LSTMCell(input_size, hidden_size, dropout) if i == 0 else LSTMCell(hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

  def __call__(self, x, hc):

    if hc is None: hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)
    output = None
    
    # forward pass through layer
    for t in range(x.shape[0]):
      new_hc = [x[t]]
      for i, cell in enumerate(self.cells):
        new_hc.append( cell(new_hc[i][:x[t].shape[0]], hc[i]) )
      hc = Tensor.stack(new_hc[1:]).realize()
      output = hc[-1:, :x.shape[1]] if output is None else output.cat(hc[-1:, :x.shape[1]], dim=0).realize()
    # for t in range(x.shape[0]):
    #   for i, cell in enumerate(self.cells):
    #     hc = cell(x[t][:x[t].shape[0]], hc[i]).unsqueeze(0) if i == 0 else hc.cat(cell(hc[:x[t].shape[0]], hc[i]).unsqueeze(0),dim=0)
    #   output = hc[-1:, :x.shape[1]] if output is None else output.cat(hc[-1:, :x.shape[1]], dim=0)
    return output, hc


class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5), dtype=c_type)
    bound = 1 / math.sqrt(self.weight.shape[1])
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound, dtype=c_type) if bias else None

  def __call__(self, x):
    return x.linear(self.weight.transpose(), self.bias)

class PPO:
  def __init__(self):
    self.input_layer = Linear(4, 64)
    self.lstm_layer = LSTM(64, 32)
    self.policy_output_layer = Linear(32, 2)
    self.value_output_layer = Linear(32, 1)

  def policy(self, input_state, hidden_state):
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, 1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper)
    lstm_output, lstm_hidden_state = self.lstm_layer(reshaped_input, hidden_state)
    policy_output = self.policy_output_layer(lstm_output)
    action_probability = policy_output.softmax(axis=2)
    return action_probability, lstm_hidden_state
    
  def value(self, input_state, hidden_state):
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper)
    lstm_output, lstm_hidden_state = self.lstm_layer(reshaped_input, hidden_state)
    value_output = self.value_output_layer(lstm_output)
    return value_output


class Buffer:
  def __init__(self):
    self.transition_data = []
      
  def save(self, transition_tuple):
    self.transition_data.append(transition_tuple)
      
  def pull(self):
    state_list, action_list, reward_list, next_state_list, action_prob_list, hidden_input_list, hidden_output_list, done_list = [], [], [], [], [], [], [], []
    for transition in self.transition_data:
      state, action, reward, next_state, action_prob, hidden_input, hidden_output, is_done = transition
      
      state_list.append(state)
      action_list.append([action])
      reward_list.append([reward])
      next_state_list.append(next_state)
      action_prob_list.append([action_prob])
      hidden_input_list.append(hidden_input)
      hidden_output_list.append(hidden_output)
      done_mask = 0 if is_done else 1
      done_list.append([done_mask])
        
    state_tensor, action_tensor, reward_tensor, next_state_tensor, done_mask_tensor, action_prob_tensor = \
    Tensor(state_list, dtype=c_type, requires_grad=False), \
    Tensor(action_list, dtype=c_type, requires_grad=False), \
    Tensor(reward_list, dtype=c_type, requires_grad=False), \
    Tensor(next_state_list, dtype=c_type, requires_grad=False), \
    Tensor(done_list, dtype=c_type, requires_grad=False), \
    Tensor(action_prob_list, dtype=c_type, requires_grad=False)
    
    self.transition_data = []
    return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_mask_tensor, action_prob_tensor, hidden_input_list[0], hidden_output_list[0]

def simulation(ppo_model, buffer, is_done, total_score, hidden_state_output, state):
    hidden_state_input = hidden_state_output
    action_prob, hidden_state_out = ppo_model.policy(state, hidden_state_input)
    action_prob = action_prob.flatten()
    hidden_state_output.assign(hidden_state_out.detach())
    action = np.random.choice(list(range(action_prob.shape[0])), p=action_prob.numpy())
    next_state, reward, is_d, truncated, info = environment.step(action)
    buffer.save((state.detach().numpy().tolist(),
                 action, 
                 reward / 100.0, 
                 next_state.tolist(), 
                 action_prob.detach().numpy()[action].tolist(), 
                 hidden_state_input, 
                 hidden_state_output, 
                 is_done[0]))

    state.assign(Tensor(next_state.tolist(), dtype=c_type, requires_grad=False))
    total_score[0] += reward
    is_done[0] = is_d
  
def training(ppo_model,buffer,optimizer,K_epoch):
  Tensor.training = True
  state_batch, action_batch, reward_batch, next_state_batch, done_mask, action_prob_batch, first_hidden_state, second_hidden_state = buffer.pull()
  
  for iteration in range(K_epoch):
    
    next_state_value = ppo_model.value(next_state_batch, second_hidden_state).squeeze(1)
    td_target = reward_batch + gamma * next_state_value * done_mask
    state_value = ppo_model.value(state_batch, first_hidden_state).squeeze(1)
    td_error = td_target - state_value
    td_error = td_error.detach().numpy()
    
    advantage_list = []
    advantage = 0.0
    for delta in td_error[::-1]:
      advantage = gamma * lmbda * advantage + delta[0]
      advantage_list.append([advantage])
    advantage_list.reverse()
    advantage_tensor = Tensor(advantage_list, dtype=c_type, requires_grad=False)

    pi, _ = ppo_model.policy(state_batch, first_hidden_state)
    pi_action = pi.squeeze(1).gather(idx=action_batch, dim=1)
    ratio = (pi_action.log() - action_prob_batch.log()).exp()
    surrogate1 = ratio * advantage_tensor 
    surrogate2 = ratio.clip(1 - eps_clip, 1 + eps_clip) * advantage_tensor
    diff = (surrogate1 < surrogate2).detach()
    surrogate = (diff*surrogate1) + (surrogate2*(-1*(diff-1)))
    # surrogate = surrogate1.add(surrogate2).sub(surrogate1 - surrogate2).mul(0.5) #equivalent of Torch.minimum
    loss = -surrogate + smooth_l1_loss(state_value, td_target)

    # print("|||||||||||||||||||||||||||||||||||||||||||||||")
    # print("pi_action: ",pi_action.flatten().numpy())
    # print("action_prob: ",action_prob_batch.flatten().numpy())
    # print("RATIO: ",ratio.flatten().numpy())
    # print("surro",-surrogate.numpy())
    # work with 800episode with lossmean with, zergograd,backward,step
    # optimizer.zero_grad()
    # print("|||||||||||||||||||||||||0000000000000000000000000||||||||||||||||||||||")
    # print("CTX: ", loss._ctx.__class__)
    # print("GRAD: ",*(i.grad.numpy() for i in optimizer.params if i.grad is not None ),sep="\n")
    optimizer.zero_grad()
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    # optimizer.zero_grad()
    # print("LOSS: ", loss.detach().numpy())
    # print("|||||||||||||||||||||||||||model grad||||||||||||||||||||")
    # print(*(i.grad.numpy() for i in optimizer.params if i.grad is not None ),sep="\n")

if __name__ == '__main__':
  #check for gymnasium
  import subprocess
  import sys
  import os

  def install_gymnasium():
    subprocess.run(["pip", "install", "gymnasium[all]", "gymnasium[atari]", "gymnasium[accept-rom-license]"])
    # to uninstall :
    #   "pip uninstall gymnasium[all] gymnasium[atari] gymnasium[accept-rom-license]"
    #   subprocess.run(["pip", "uninstall", "gymnasium[all]", "gymnasium[atari]", "gymnasium[accept-rom-license]"])

  try: import gymnasium as gym
  except ImportError:
    install_choice = input("Gymnasium is not installed. Install now? (y/n): ")
    if install_choice.lower() == ['y','ye''yes']:
      install_gymnasium()
      print("Gymnasium installed. Restarting the script.")
      os.execv(sys.executable, ['python'] + sys.argv)
    else: print("Gymnasium is required. Exiting."), sys.exit(1)

  ######## PPO Training ##########
  environment = gym.make('CartPole-v1')
  ppo_model = PPO()
  optimizer = Adam(get_parameters(ppo_model), lr=learning_rate)
  buffer=Buffer()

  state, _ = environment.reset()
  state = Tensor(state.tolist(), dtype=c_type, requires_grad=False)
  hidden_state_output = Tensor.ones(1,2,32, dtype=c_type, requires_grad=False)
  
  is_done = [False]
  total_score = [0.0]
  cyclic_loss=[]
  episode_index=0
  import time
  while episode_index != episode:
    Tensor.training = False
    start = time.time()
    for _ in range(T_horizon):
      simulation(ppo_model,buffer,
                is_done,
                total_score,
                hidden_state_output,
                state)
      if is_done[0]: 
        
        cyclic_loss.append(total_score[0])
        episode_index+=1
        # print(f"| score: {total_score[0]} | ")
        if len(cyclic_loss)>=print_interval and episode_index%print_interval==0 : 
          print(f"| Average score: {sum(cyclic_loss[-print_interval:])/print_interval} of the last {print_interval} episode , over {episode_index} episode| ")
        
        #episode variable to reset
        state, _  = environment.reset()
        state = Tensor(state.tolist(), dtype=c_type, requires_grad=False)
        hidden_state_output = Tensor.ones(1,2,32, dtype=c_type, requires_grad=False)
        
        total_score = [0.]
        is_done = [False]
        #break if you need full episode stacking instead of partial
              
    training(ppo_model,
            buffer,
            optimizer,
            K_epoch)
  print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
      
  environment.close()
  
  
  
  
  
  
  
  #tinyegrad need fix tensor size, so should do the same amount of data
  
#   Tinygrad:
# next_state_value.requires_grad:  True
# td_target.requires_grad:         True
# state_value.requires_grad:       True
# td_error.requires_grad:          True
# advantage_tensor.requires_grad:  False
# pi.requires_grad:                True
# pi_action.requires_grad:         True
# ratio.requires_grad:             True
# surrogate1.requires_grad:        True
# surrogate2.requires_grad:        True
# surrogate.requires_grad:         True
# loss.requires_grad:              True

# Pytorch:

# v_prime.requires_grad:           True
# td_target.requires_grad:         True
# v_s.requires_grad:               True
# delta.requires_grad:             True
# advantage.requires_grad:         False
# pi.requires_grad:                True
# pi_a.requires_grad:              True
# ratio.requires_grad:             True
# surr1.requires_grad:             True
# surr2.requires_grad:             True
# Surrogate.requires_grad:         True
# loss.requires_grad:              True