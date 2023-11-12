from typing import Dict, List, DefaultDict, Tuple, Any, Union
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn import Linear
from tinygrad.helpers import dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.ops import _Device as Device
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.codegen.linearizer import Linearizer
import numpy as np
import os


def sgn(x : Tensor) -> Tensor: return x/x.pow(2.).sqrt()#.abs()
def inf_to(x : Tensor,y : Tensor) -> Tensor: return (1-sgn(x-y))/2
def sup_to(x : Tensor,y : Tensor) -> Tensor: return (1+sgn(x-y))/2
def smooth_l1_loss(predicted_value : Tensor, target_value : Tensor, beta : float = 1.0) -> Tensor:
  elementwise_difference = predicted_value - target_value
  mask = inf_to(elementwise_difference , beta)
  return (mask * (0.5 * elementwise_difference**2 / beta)) + ((-1 *(mask - 1)) * (elementwise_difference.abs() - 0.5 * beta))


import math
class LSTMCell:
  def __init__(self, input_size : int, hidden_size : int, dropout : float, c_type : dtypes):
    self.dropout = dropout
    bound = math.sqrt(1 / hidden_size)
    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size, low=-bound, high=bound, dtype=c_type)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size, low=-bound, high=bound, dtype=c_type)
    self.bias_ih = Tensor.ones(hidden_size, dtype=c_type).cat(Tensor.zeros(hidden_size * 3,dtype=c_type),dim=0)
    self.bias_hh = Tensor.zeros(hidden_size * 4, dtype=c_type)
  def __call__(self, x : Tensor, hc : Tensor) -> Tensor:
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i, f, g, o
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = (f * hc[x.shape[0]:]) + (i * g)
    h = (o * c.tanh()).dropout(self.dropout)
    return Tensor.cat(h, c).realize()


class LSTM:
  def __init__(self, input_size : int, hidden_size : int, layers : int = 1, dropout : float = 0. ,c_type : dtypes = dtypes.float32):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers
    self.cells = [LSTMCell(input_size, hidden_size, dropout,c_type) if i == 0 else LSTMCell(hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]
  def __call__(self, x: Tensor, hc : Tensor) -> Tuple[Tensor,Tensor]:
    if hc is None: hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)
    output = None
    # forward pass through layer
    for t in range(x.shape[0]):
      new_hc = [x[t]]
      new_hc.extend(
          cell(new_hc[i][:x[t].shape[0]], hc[i])
          for i, cell in enumerate(self.cells))
      hc = Tensor.stack(new_hc[1:])
      output = hc[-1:, :x.shape[1]] if output is None else output.cat(hc[-1:, :x.shape[1]], dim=0)
    return output.realize(), hc.realize()


class PPO_tiny():
  def __init__(self, state_size : int = 4, 
                action_size : int = 2, 
                value_size : int = 1, 
                hidden_space_size : int = 64, 
                learning_rate : float = 0.0001,
                gamma : float = 0.98, 
                lmbda : float = 0.95, 
                eps_clip : float = 0.1, 
                K_epoch : int = 2,
                c_type = dtypes.float32):

    self.c_type = dtypes.float32
    self.data = []
    self.gamma = gamma 
    self.lmbda = lmbda 
    self.eps_clip = eps_clip
    self.K_epoch = K_epoch
    self.loss = 0
    self.data = []
    self.hidden_space_size = hidden_space_size
    self.hidden_space_lstm_size = hidden_space_size // 2
    self.input_layer = Linear(state_size, hidden_space_size)
    self.lstm_layer = LSTM(hidden_space_size, hidden_space_size // 2, c_type=c_type)
    self.policy_output_layer = Linear(hidden_space_size // 2, action_size)
    self.value_output_layer = Linear(hidden_space_size // 2, value_size)
    self.optimizer = Adam([value for key, value in get_state_dict(self.input_layer).items() if isinstance(value, Tensor)] +
                          [value for i in self.lstm_layer.cells for key, value in get_state_dict(i).items() if isinstance(value, Tensor)] + 
                          [value for key, value in get_state_dict(self.policy_output_layer).items() if isinstance(value, Tensor)] +
                          [value for key, value in get_state_dict(self.policy_output_layer).items() if isinstance(value, Tensor)],
                          lr=learning_rate)

  def pi(self, input_state : Tensor, hidden_state : Tensor) -> tuple[Tensor, Tensor]:
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, 1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper)
    lstm_output, lstm_hidden_state = self.lstm_layer(reshaped_input, hidden_state)
    policy_output = self.policy_output_layer(lstm_output)
    action_probability = policy_output.softmax(axis=2)
    return action_probability.realize(), lstm_hidden_state.realize()
    
  def v(self, input_state, hidden_state) -> Tensor:
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, 1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper)
    lstm_output, _ = self.lstm_layer(reshaped_input, hidden_state)
    value_output = self.value_output_layer(lstm_output)
    return value_output.realize()        
      
  def put_data(self, transition_tuple : tuple) -> None:
    self.data.append(transition_tuple)
      
  def make_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    state_list, action_list, reward_list, next_state_list, action_prob_list, hidden_input_list, hidden_output_list, done_list = [], [], [], [], [], [], [], []
    for transition in self.data:
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
    Tensor(state_list, dtype=self.c_type, requires_grad=False), \
    Tensor(action_list, dtype=self.c_type, requires_grad=False), \
    Tensor(reward_list, dtype=self.c_type, requires_grad=False), \
    Tensor(next_state_list, dtype=self.c_type, requires_grad=False), \
    Tensor(done_list, dtype=self.c_type, requires_grad=False), \
    Tensor(action_prob_list, dtype=self.c_type, requires_grad=False)
    
    self.data = []
    return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_mask_tensor, action_prob_tensor, Tensor(hidden_input_list[0], dtype=self.c_type, requires_grad=False), Tensor(hidden_output_list[0], dtype=self.c_type, requires_grad=False)
  
  def train_net(self) -> None:

    state_batch, action_batch, reward_batch, next_state_batch, done_mask, action_prob_batch, first_hidden_state, second_hidden_state = self.make_batch()
    Tensor.no_grad, Tensor.training = False, True
    
    for _ in range(self.K_epoch):
      next_state_value = self.v(next_state_batch, second_hidden_state).squeeze(1)
      td_target = reward_batch + self.gamma * next_state_value * done_mask
      state_value = self.v(state_batch, first_hidden_state).squeeze(1)
      td_error = td_target - state_value
      td_error = td_error.detach().numpy()
      
      advantage_list = []
      advantage = 0.0
      for delta in td_error[::-1]:
        advantage = self.gamma * self.lmbda * advantage + delta[0]
        advantage_list.append([advantage])
      advantage_list.reverse()
      advantage_tensor = Tensor(advantage_list, dtype=self.c_type, requires_grad=False)
      
      pi, _ = self.pi(state_batch, first_hidden_state)
      pi_action = pi.squeeze(1).gather(idx=action_batch, dim=1)
      ratio = (pi_action.log() - action_prob_batch.log()).exp()
      surrogate1 = ratio * advantage_tensor 
      surrogate2 = (ratio.clip(1 - self.eps_clip, 1 + self.eps_clip) * advantage_tensor)
      diff = (surrogate1 < surrogate2).detach()
      surrogate = (diff*surrogate1) + (surrogate2*(-1*(diff-1)))
      loss = -surrogate + smooth_l1_loss(state_value, td_target)
      
      self.optimizer.zero_grad()
      loss = loss.mean()
      loss.backward()
      self.optimizer.step()
      loss.realize()
      
    self.loss= loss.detach().numpy()
    if math.isinf(self.loss) or np.isnan(self.loss): raise ("gradient vanish")
   
############## CYCLE TRAIN/INF ##############
def training(env=None, learning_rate : float = 0.0005, gamma : float = 0.98,
             lmbda : float = 0.95, eps_clip : float = 0.1, 
             K_epoch : int = 2, state_size : int = 4, 
             action_size : int = 2, value_size : int = 1, 
             hidden_space_size : int = 64, minimum_batch_size : int = 60, 
             max_steps_limit : int = 60, episode_count : int = 300, 
             print_interval : int = 1, path_save_episode : str = "/home/test1/", 
             model_path : str = "model_all.safetensors") -> None:
  
  model =  PPO_tiny(state_size=state_size, 
                action_size=action_size, 
                value_size=value_size, 
                hidden_space_size=hidden_space_size,
                learning_rate=learning_rate,
                gamma=gamma, 
                lmbda=lmbda, 
                eps_clip=eps_clip, 
                K_epoch=K_epoch)
  full_model_path = os.path.join(path_save_episode, model_path)
  load_state_dict(model, safe_load(full_model_path)) if os.path.exists(full_model_path) else None
  
  current_episode, avg_score = 0, []  
  while current_episode < episode_count:
    done, max_steps, c_score = False, 0 ,0
    Tensor.no_grad, Tensor.training = False, False
    hidden_state = Tensor.ones(1,2,model.hidden_space_lstm_size, dtype=model.c_type, requires_grad=False)
    state, _ = env.reset()

    while not (done or len(model.data) >= minimum_batch_size or max_steps == max_steps_limit):
      action_prob, new_hidden_state = model.pi(Tensor(state.tolist(), dtype=model.c_type, requires_grad=False), hidden_state)
      policy = action_prob.flatten().detach().numpy() 
      action = np.random.choice(list(range(policy.shape[0])), p=policy)

      next_state, reward, done, _, _ = env.step(action)
      model.put_data((state, action, reward/100, next_state, action_prob.flatten().detach().numpy()[action], hidden_state.detach().numpy(), new_hidden_state.detach().numpy(), done))
      state , hidden_state = next_state , new_hidden_state
      max_steps += 1  
      c_score += reward
  
      if len(model.data) == minimum_batch_size:
        model.train_net()
    current_episode += 1 
    avg_score.append(c_score) 
    if current_episode % print_interval == 0:
        print(f"# of episode: {current_episode}, avg score: {np.array(avg_score).mean():.3f}, loss: {model.loss:.5f}")
        safe_save(get_state_dict(model), full_model_path)    
        avg_score.clear()         
  env.close()



def inference(env : List[str], max_number_of_step : int = 20,
              path_save_episode : str = "/home/test1/", model_path : str = "model_all.safetensors",
              learning_rate : float = 0.0003, gamma : float = 0.97,
              lmbda : float = 0.90, eps_clip : float = 0.1, 
              K_epoch : int = 2, state_size : int = 1021, 
              action_size : int = 2, value_size : int = 1, 
              hidden_space_size : int = 512) -> List[Linearizer]:
  
  model =  PPO_tiny(state_size=state_size, 
                action_size=action_size, 
                value_size=value_size, 
                hidden_space_size=hidden_space_size,
                learning_rate=learning_rate,
                gamma=gamma, 
                lmbda=lmbda, 
                eps_clip=eps_clip, 
                K_epoch=K_epoch)
  full_model_path = os.path.join(path_save_episode, model_path)
  load_state_dict(model, safe_load(full_model_path)) if os.path.exists(full_model_path) else None
  
  state, _ = env.reset()
  done=False
  hidden_state = Tensor.ones(1,2,model.hidden_space_lstm_size, dtype=model.c_type, requires_grad=False)
  max_moves = 0

  while max_moves < max_number_of_step or done: 
    action_prob, hidden_state = model.pi(Tensor(state.tolist(), dtype=model.c_type, requires_grad=False), hidden_state)
    sorted_indices = action_prob.flatten().detach().numpy().argsort()[::-1] 
    state, reward, done, _, _ = env.step(sorted_indices[0])
    max_moves += 1 
    print(f"Move: {max_moves}, action: {sorted_indices[0]}, reward: {reward}, Gameover: {done}")
  env.close


  
#ENTRY POINT
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Training or Inference')
  #by default without infer it will perform training
  parser.add_argument('--inference', help='Perform inference', action='store_true')
  #share parameter between inference and training
  parser.add_argument('--env', help='Environment', default='CartPole-v1')
  parser.add_argument('--render', help='Render mode', default='human') # rgb_array will change state to RGB array and anything else will run the env without visual render
  parser.add_argument('--path', help='Path to save episode', default='/home/usr/ubuntu/TinyRL/cartpole/')
  parser.add_argument('--model_name', help='model name', default='model_all.safetensors')
  parser.add_argument('--max_steps_limit', type=int, default=200)
  #training parameter
  parser.add_argument('--learning_rate', type=float, default=0.0005)
  parser.add_argument('--gamma', type=float, default=0.98)
  parser.add_argument('--lmbda', type=float, default=0.95)
  parser.add_argument('--eps_clip', type=float, default=0.1)
  parser.add_argument('--K_epoch', type=int, default=2)
  parser.add_argument('--hidden_space_size', type=int, default=64)
  parser.add_argument('--minimum_batch_size', type=int, default=20)
  parser.add_argument('--episode_count', type=int, default=1000)
  parser.add_argument('--print_interval', type=int, default=1)
  args = parser.parse_args()
  os.makedirs(args.path, exist_ok=True)
  
  import subprocess, sys, os
  def install_gymnasium(): subprocess.run(["pip", "install", "gymnasium[all]", "gymnasium[atari]", "gymnasium[accept-rom-license]"])
  try: import gymnasium as gym
  except ImportError:
    install_choice = input("Gymnasium is not installed. Install now? (y/n): ")
    if 'y' in install_choice.lower():
      install_gymnasium()
      print("Gymnasium installed. Restarting the script.")
      os.execv(sys.executable, ['python'] + sys.argv)
    else: print("Gymnasium is required. Exiting."), sys.exit(1)

  ######## PPO Training ##########
  env = gym.make(args.env, render_mode=args.render)
  action_space = int(env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0])
  observation_space = int(env.observation_space.n if isinstance(env.observation_space, gym.spaces.Discrete) else env.observation_space.shape[0])

  if args.inference:    
    result = inference(env=env,
                        max_number_of_step = args.max_steps_limit,
                        path_save_episode = args.path,
                        model_path = args.model_name,
                        max_trial = args.max_inference_trial,
                        learning_rate = args.learning_rate, 
                        gamma = args.gamma,
                        lmbda = args.lmbda, 
                        eps_clip = args.eps_clip, 
                        K_epoch = args.K_epoch, 
                        state_size = observation_space, 
                        action_size = action_space, 
                        value_size = 1, 
                        hidden_space_size = args.hidden_space_size)
  else:
    training(env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lmbda=args.lmbda,
        eps_clip=args.eps_clip,
        K_epoch=args.K_epoch,
        state_size=observation_space,
        action_size=action_space,
        value_size=1,
        hidden_space_size=args.hidden_space_size,
        minimum_batch_size=args.minimum_batch_size,
        max_steps_limit=args.max_steps_limit,
        episode_count=args.episode_count,
        print_interval=args.print_interval,
        path_save_episode = args.path,
        model_path= args.model_name
    )

        
