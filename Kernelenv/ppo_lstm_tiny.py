from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Any, Callable, Union
from tinygrad.ops import LazyOp
############## TOOLKITS ##############
import re
import cProfile
import pstats
import time
def profile_function(func):
  def wrapper(*args, **kwargs):
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:  # Save only if elapsed time is 1 second or more
      with open("/home/usr/profile_stats.txt", "a") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()
    return result
  return wrapper
# @profile_function

import time
def format_time(elapsed_time : float) -> str:
  hours, remainder = divmod(elapsed_time, 3600)
  minutes, remainder = divmod(remainder, 60)
  seconds, remainder = divmod(remainder, 1)
  remainder = remainder * 1e6
  milliseconds, remainder = divmod(int(remainder), 1000)
  microseconds = int(remainder)
  return f"{int(hours):02} H {int(minutes):02} min {int(seconds):02} sec {int(milliseconds):02} ms {int(microseconds):02} us "

from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
def force_oom() -> None:
  if 1 in [getenv("TRITON") ,getenv("CUDA")]:
    try: large_tensor = Tensor.empty(10000000000000).realize()
    except Exception as e: pass

def split_and_find_max(list_of_lists : list)  -> List[List[Union[float, bool, int]]]:
  stacked_list = []
  temp_list = []
  for inner_list in list_of_lists:
    temp_list.append(inner_list)
    if inner_list[1] is True:  
      stacked_list.append(temp_list.copy())
      temp_list = []
  if temp_list:  
    stacked_list.append(temp_list)
  max_values = [max(stack, key=lambda x: x[0])[0] for stack in stacked_list]
  return max_values

# #extract custom kernel to use in KernelEnv
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from typing import List, Any
def extract_ast_kernels(mdl : Callable, example : Tensor) -> List[str]:
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")
  seen = set()
  mdl(example).lazydata.schedule(seen)
  md = mdl(example)
  sched = md.lazydata.schedule(seen)
  sched = [str(x.ast) for x in sched if x.ast.op not in LoadOps]
  return sched

import numpy as np
def count_pos_neg(arr : np.ndarray) -> Tuple[int,int]:
  pos_count , neg_count = np.sum(arr > 0) , np.sum(arr < 0)
  return pos_count, neg_count

def print_to_file(file_path : str, text : str) -> None:
  with open(file_path, 'a') as file:
    print(text, file=file)

import pickle
def save_val(episode : Any, filename : str = "state.pkl") -> None:
    with open(f"{filename}", "wb") as f:pickle.dump(episode, f)
def load_val(default : Any = 0, filename : str = "state.pkl") -> Any:
  try: 
    with open(f"{filename}", "rb") as f: return pickle.load(f)
  except FileNotFoundError:return default


############## ENV ##############
import os , sqlite3, time , pickle, math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.features.search import actions, bufs_from_lin, time_linearizer
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats
from tinygrad.codegen.optimizer import Opt

class KernelEnv:
  def __init__(self,
                available_kernels : List[str] = None,
                db_path : str = None,
                inference_mode : bool= False,
                index_to_pick : int = 0, 
                max_n_mouve : int = 60,
                max_retries : int = 1000,
                terminal_reward : float = -1.):
    
    self.terminal_reward = terminal_reward#-(math.log(GLOBAL_TIMEOUT)*np.pi)
    self.index = index_to_pick
    self.inference_mode = inference_mode
    self.db_path = db_path
    self.max_n_mouve = max_n_mouve
    self.max_retries = max_retries
    self.init_db()
    if available_kernels is not None: self.add_kernels(available_kernels)
    self.max_global_size= 65536
    
  def query_db(self, query : str, args : tuple =()) -> tuple:
    for retry in range(self.max_retries):
      conn = None
      try:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        c = conn.cursor()
        c.execute(query, args)
        data = c.fetchall()
        conn.close()
        return data
      except Exception as e:
        if not isinstance(e, sqlite3.OperationalError): print(f'An exception occurred: {e}')
        if retry == self.max_retries - 1: raise
        time.sleep(0.1)
      finally: 
        if conn is not None: conn.close()

  def update_db(self, query : str, args: tuple = ()) -> None:
    for retry in range(self.max_retries):
      conn = None
      try:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        c = conn.cursor()
        c.execute(query, args)
        conn.commit()
        conn.close()
        return
      except Exception as e:
        if not isinstance(e, sqlite3.OperationalError): print(f'An exception occurred: {e}')
        if retry == self.max_retries - 1: raise
        time.sleep(0.1)
      finally: 
        if conn is not None: conn.close() 
      
  def init_db(self) -> None:
    if not os.path.exists(self.db_path):
      self.update_db("CREATE TABLE kernels (id INTEGER PRIMARY KEY, kernel_data BLOB, last_num_steps INTEGER, total_visits INTEGER)")

  def add_kernels(self, kernel_list : List[str]) -> None:
    for kernel in kernel_list:
      serialized_kernel = pickle.dumps(kernel)
      if not self.inference_mode:
        res = self.query_db("SELECT * FROM kernels WHERE kernel_data = ?", (serialized_kernel,))
        if len(res) == 0: self.update_db("INSERT INTO kernels (kernel_data, last_num_steps, total_visits) VALUES (?, 0, 0)", (serialized_kernel,))
      else: self.update_db("INSERT INTO kernels (kernel_data, last_num_steps, total_visits) VALUES (?, 0, 0)", (serialized_kernel,))

  def priority_game(self) -> int:
    data = self.query_db("SELECT id, last_num_steps, total_visits FROM kernels")
    active_indices = np.array([row[0] for row in data])
    last_num_steps = np.array([row[1] for row in data])
    total_visits = np.array([row[2] for row in data])
    avg_visit = np.mean(total_visits)
    visit_factors = np.where(total_visits <= avg_visit / 2, 0.5, 1)
    step_factors = (self.max_n_mouve - last_num_steps) / self.max_n_mouve
    weights = visit_factors * step_factors
    normalized_weights = weights / np.sum(weights)
    chosen_index = np.random.choice(active_indices, p=normalized_weights)
    self.update_db(f"UPDATE kernels SET last_num_steps = 0, total_visits = total_visits + 1 WHERE id = {chosen_index}")
    return chosen_index

  def get_kernel(self, kernel_id : int) -> str:
    data = self.query_db(f"SELECT kernel_data FROM kernels WHERE id = {kernel_id}")
    serialized_kernel = data[0][0]
    return pickle.loads(serialized_kernel)

  def update_kernel_step(self, kernel_id : int) -> None:
    self.update_db(f"UPDATE kernels SET last_num_steps = last_num_steps + 1 WHERE id = {kernel_id}")
  
  def delete_kernel(self, kernel_id : int) -> None:
    self.update_db(f"DELETE FROM kernels WHERE id = {kernel_id}")

  def count_kernels(self) -> Tuple[int]:
    return self.query_db("SELECT COUNT(*) FROM kernels")

  def reset(self) -> Tuple[Optional[np.ndarray], None]:
    while True:
      try:
        self.max_regret = [] 
        #Select kernel
        if self.inference_mode:
          self.kernel_pick_index = self.index
          kernel_pick = self.get_kernel(self.kernel_pick_index)
        else:
          self.kernel_pick_index = self.priority_game()
          kernel_pick = self.get_kernel(self.kernel_pick_index)
          
        #Convert ast kernel to linearizer, get initial kernel speed and state           
        linearized_kernel = ast_str_to_lin(kernel_pick)
        buffer_info = bufs_from_lin(linearized_kernel)
        initial_reward = time_linearizer(linearized_kernel, buffer_info, allow_test_size=True, should_copy=True, max_global_size=self.max_global_size)
        if math.isinf(initial_reward) or np.isnan(initial_reward): raise ValueError("Invalid initial time.")
        initial_state = np.array(lin_to_feats(linearized_kernel))
        
        #Save result
        self.i_r , self.i_s, self.l_k = initial_reward, initial_state, linearized_kernel
        self.done = False
        self.max_regret.append(self.i_r)
        return self.i_s, None 
      
      except Exception as e:
        # For broken kernel, show the error and delete them from dataset
        if not self.count_kernels()[0] == 0 : self.delete_kernel(self.kernel_pick_index)
        print("REMOVE KERNEL AFTER ERROR: ",e)
        print(f"Number of active kernels after removal: {self.count_kernels()[0]}")
        if self.count_kernels()[0] == 0 and self.inference_mode: return None, None
  
  def step(self, action_idx : int , no_reward : bool = False) -> Tuple[np.ndarray, float, bool, None, float]:
    if self.done: raise Exception("Episode already done, reset the environment first.")
    #default var for fail action
    reward = self.terminal_reward
    obtained_reward = self.terminal_reward
    
    try:
      #select action from the list of available ops and last available linearized kernel
      action_to_apply = actions[action_idx - 1]
      linearized_kernel_copy = self.l_k
      
      # Check if action axis is within the shape length of the kernel
      if action_to_apply.axis >= linearized_kernel_copy.shape_len:
        raise ValueError("Invalid action: axis out of bounds.")
      # Check if the action amount matches the axis size and a zero-change operation exists
      if (linearized_kernel_copy.full_shape[action_to_apply.axis] == action_to_apply.amt and 
        Opt(action_to_apply.op, action_to_apply.axis, 0) in actions):
        raise ValueError("Invalid action: action amount matches axis size and a zero-change operation exists.")
      # Tentatively apply the action to a copy to test for workgroup size constraints
      linearized_kernel_copy.apply_opt(action_to_apply)
      up, lcl = 1, 1
      for s, c in zip(linearized_kernel_copy.full_shape, linearized_kernel_copy.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256: raise ValueError("Invalid action: exceeds workgroup size constraints.")
      
      if not no_reward:
        # Calculate and save the reward
        buffer_info = bufs_from_lin(linearized_kernel_copy)
        obtained_reward = time_linearizer(linearized_kernel_copy, buffer_info, allow_test_size=True, should_copy=True, max_global_size=self.max_global_size)
        if math.isinf(reward) or np.isnan(reward): raise ValueError("Invalid reward.")
        reward = 1. if obtained_reward < min(self.max_regret) else -1.
        self.l_k  = linearized_kernel_copy
        # print(linearized_kernel_copy.printbufs())
        
    except Exception as e: pass
      # print(f"ILLEGAL MOVE ERROR: {e}") 

    # Determine if the episode is done and compute the new state
    self.done = action_idx == 0 or reward == self.terminal_reward
    state = self.next_state if hasattr(self, 'next_state') else self.i_s
    if not self.done:
      try: state = np.array(lin_to_feats(linearized_kernel_copy))
      except Exception as e:# print(e)
        self.done, state, reward = True, (state*0)+np.pi, self.terminal_reward
    else: state, reward = (state*0)+np.pi, self.terminal_reward
    
    # Updating next state, max regret, and potentially updating the kernel
    self.next_state = state
    self.max_regret.append(obtained_reward  if reward!=self.terminal_reward else self.max_regret[0])
    if not self.inference_mode: self.update_kernel_step(self.kernel_pick_index)
    return self.next_state, reward, self.done, None, obtained_reward
7

############## MODEL ##############
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
# from tinygrad.nn import Linear
from tinygrad.helpers import dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.ops import _Device as Device
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.codegen.linearizer import Linearizer
import numpy as np
import time


def sgn(x : Tensor) -> Tensor: return x/x.pow(2.).sqrt()#.abs()
def inf_to(x : Tensor,y : Tensor) -> Tensor: return (1-sgn(x-y))/2
def sup_to(x : Tensor,y : Tensor) -> Tensor: return (1+sgn(x-y))/2
def smooth_l1_loss(predicted_value : Tensor, target_value : Tensor, reduction : str = 'mean', beta : float = 1.0) -> Tensor:
  elementwise_difference = predicted_value - target_value
  mask = inf_to(elementwise_difference , beta).realize()
  loss = (mask * (0.5*elementwise_difference**2 / beta)) + ((-1*(mask-1)) * (elementwise_difference.abs() - 0.5*beta)).realize()
  return loss


import math
class LSTMCell:
  def __init__(self, input_size : int, hidden_size : int, dropout : float, c_type : dtypes):
    self.dropout = dropout
    bound = math.sqrt(1 / hidden_size)
    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size, low=-bound, high=bound, dtype=c_type)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size, low=-bound, high=bound, dtype=c_type)
    self.bias_ih = Tensor.ones(hidden_size, dtype=c_type).cat(Tensor.zeros(hidden_size * 3,dtype=c_type),dim=0).realize()
    self.bias_hh = Tensor.zeros(hidden_size * 4, dtype=c_type)
  def __call__(self, x : Tensor, hc : Tensor) -> Tensor:
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.realize(), f.realize(), g.realize(), o.realize()
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = (f * hc[x.shape[0]:].realize()) + (i * g)
    h = (o * c.realize().tanh()).dropout(self.dropout).realize()
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
      for i, cell in enumerate(self.cells):
        new_hc.append( cell(new_hc[i][:x[t].shape[0]], hc[i]) )
      hc = Tensor.stack(new_hc[1:]).realize()
      output = hc[-1:, :x.shape[1]].realize() if output is None else output.cat(hc[-1:, :x.shape[1]], dim=0).realize()
    return output, hc


class Linear:
  def __init__(self, in_features : int, out_features : int, bias : bool = True, c_type : dtypes = dtypes.float32):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5), dtype=c_type)
    bound = 1 / math.sqrt(self.weight.shape[1])
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound, dtype=c_type) if bias else None
  def __call__(self, x : Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)


class PPO_tiny():
  def __init__(self, state_size : int = 240, 
                action_size : int = 1+len(actions), 
                value_size : int = 1, 
                hidden_space_size : int = 512, 
                learning_rate : float = 0.0001,
                gamma : float = 0.97, 
                lmbda : float = 0.90, 
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
    self.input_layer = Linear(state_size, hidden_space_size,c_type=c_type)
    self.lstm_layer = LSTM(hidden_space_size, hidden_space_size // 2,c_type=c_type)
    self.policy_output_layer = Linear(hidden_space_size // 2, action_size,c_type=c_type)
    self.value_output_layer = Linear(hidden_space_size // 2, value_size,c_type=c_type)
    self.optimizer = Adam([value for key, value in get_state_dict(self.input_layer).items() if isinstance(value, Tensor)] +
                          [value for i in self.lstm_layer.cells for key, value in get_state_dict(i).items() if isinstance(value, Tensor)] + 
                          [value for key, value in get_state_dict(self.policy_output_layer).items() if isinstance(value, Tensor)] +
                          [value for key, value in get_state_dict(self.policy_output_layer).items() if isinstance(value, Tensor)],
                          lr=learning_rate)

  def pi(self, input_state : Tensor, hidden_state : Tensor) -> tuple[Tensor, Tensor]:
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, 1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper).realize()
    lstm_output, lstm_hidden_state = self.lstm_layer(reshaped_input, hidden_state)
    policy_output = self.policy_output_layer(lstm_output)
    action_probability = policy_output.softmax(axis=2).realize()
    return action_probability, lstm_hidden_state
    
  def v(self, input_state, hidden_state) -> Tensor:
    processed_input = self.input_layer(input_state).relu()
    shaper = ((1, processed_input.shape[0]) if len(processed_input.shape) == 1 else (processed_input.shape[0], 1, processed_input.shape[1]))
    reshaped_input = processed_input.reshape(*shaper).realize()
    lstm_output, lstm_hidden_state = self.lstm_layer(reshaped_input, hidden_state)
    value_output = self.value_output_layer(lstm_output)
    return value_output        
      
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
    return state_tensor.realize(), action_tensor.realize(), reward_tensor.realize(), next_state_tensor.realize(), done_mask_tensor.realize(), action_prob_tensor.realize(), hidden_input_list[0], hidden_output_list[0]
  
  def train_net(self) -> None:

    state_batch, action_batch, reward_batch, next_state_batch, done_mask, action_prob_batch, first_hidden_state, second_hidden_state = self.make_batch()
    Tensor.no_grad, Tensor.training = False, True
    
    for _ in range(self.K_epoch):
      next_state_value = self.v(next_state_batch, second_hidden_state).squeeze(1).realize()
      td_target = reward_batch + self.gamma * next_state_value * done_mask
      state_value = self.v(state_batch, first_hidden_state).squeeze(1).realize()
      td_error = td_target - state_value
      td_error = td_error.detach().numpy()
      
      advantage_list = []
      advantage = 0.0
      for delta in td_error[::-1]:
        advantage = self.gamma * self.lmbda * advantage + delta[0]
        advantage_list.append([advantage])
      advantage_list.reverse()
      advantage_tensor = Tensor(advantage_list, dtype=self.c_type, requires_grad=False).realize()
      
      pi, _ = self.pi(state_batch, first_hidden_state)
      pi_action = pi.squeeze(1).gather(idx=action_batch, dim=1).realize()
      ratio = (pi_action.log() - action_prob_batch.log()).exp()
      surrogate1 = ratio * advantage_tensor 
      surrogate2 = (ratio.clip(1 - self.eps_clip, 1 + self.eps_clip) * advantage_tensor).realize()
      diff = (surrogate1 < surrogate2).detach()
      surrogate = (diff*surrogate1) + (surrogate2*(-1*(diff-1)))
      loss = -surrogate.realize() + smooth_l1_loss(state_value, td_target)
      
      self.optimizer.zero_grad()
      loss = loss.mean()
      loss.backward()
      self.optimizer.step()
      
    self.loss= loss.detach().numpy()
    if math.isinf(self.loss) or np.isnan(self.loss): raise ("gradient vanish")

    
############## CYCLE TRAIN/INF ##############
def training(learning_rate : float = 0.0003, gamma : float = 0.97,
             lmbda : float = 0.90, eps_clip : float = 0.1, 
             K_epoch : int = 2, state_size : int = 240, 
             action_size : int = 1+len(actions), value_size : int = 1, 
             hidden_space_size : int = 512, minimum_batch_size : int = 60, 
             max_steps_limit : int = 60, episode_count : int = 2000, 
             print_interval : int = 1, path_save_episode : str = "/home/test1/", 
             model_path : str = "model_all.safetensors") -> None:
  
  # Initialize model
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
  # Initialize episode variables
  current_episode = load_val(default = 0,filename = path_save_episode+"state.pkl")
  action_count = load_val(default=np.zeros(1 + len(actions)),filename = path_save_episode+"action_count.pkl")
  episode_scores = []
  
  #training
  while current_episode < episode_count:
    # Reset var for new episode
    done, max_steps, c = False, 0 , 2
    Tensor.no_grad, Tensor.training = False, False
    hidden_state = Tensor.ones(1,2,model.hidden_space_lstm_size, dtype=model.c_type, requires_grad=False)
    #init env
    env = KernelEnv(db_path = path_save_episode + 'dataset_training.db',max_n_mouve = max_steps_limit)
    state, _ = env.reset()
    
    # Collect data for one episode
    while not (done or len(model.data) >= minimum_batch_size or max_steps == max_steps_limit):
      # UCB Action Selection
      action_prob, new_hidden_state = model.pi(Tensor(state.tolist(), dtype=model.c_type, requires_grad=False), hidden_state)
      ucb_values = action_prob.flatten().detach().numpy() + c * np.sqrt(np.log(1+max_steps) / (1 + action_count)) 
      action = np.random.choice(list(range(ucb_values.shape[0])), p=ucb_values/ucb_values.sum())
      # Take action and observe next state and reward
      next_state, reward, done, _, actual_runtime = env.step(action)
      model.put_data((state, action, reward, next_state, action_prob.flatten().detach().numpy()[action], hidden_state, new_hidden_state, done))

      del state , hidden_state #ugly
      state , hidden_state = next_state , new_hidden_state #ugly
      modified_reward = env.i_r/actual_runtime if reward != env.terminal_reward else -1 #ugly
      episode_scores.append([modified_reward , done , action]) #ugly
      action_count[action] += 1  # Update action count #ugly
      max_steps += 1  #ugly
    force_oom()#force cuda vram reset
  
    if len(model.data) >= minimum_batch_size:
      model.train_net()
      current_episode += 1 #ugly
      save_val(current_episode ,filename= path_save_episode+"state.pkl") #ugly
      save_val(action_count ,filename = path_save_episode+"action_count.pkl") #ugly
      
      if current_episode % print_interval == 0:
        avg_score = np.array(split_and_find_max(episode_scores))
        pos, neg = count_pos_neg(avg_score) 
        pr_loss= f"# of episode: {current_episode}, avg score: {avg_score.mean():.3f}, repartition positive: {(pos/(neg+pos)) * 100:.3f} % loss: {model.loss:.5f}"
        print(pr_loss)
        print_to_file(path_save_episode + 'state_model_all.txt', f"ep: {current_episode} | <Compute_time : Done : Action>: {[(a[0], a[1], a[2]) for i, a in enumerate(episode_scores)]}") #ugly
        print_to_file(path_save_episode + 'state_model_all.txt', "||||||||||||||||||||||||||") #ugly
        print_to_file(path_save_episode + 'loss_model_all.txt', pr_loss) #ugly
        episode_scores.clear()
        safe_save(get_state_dict(model), full_model_path)
        if current_episode % 100 == 0: safe_save(get_state_dict(model), os.path.join(path_save_episode, f"{current_episode}_episode_"+model_path))
        if current_episode % 1 == 0: force_oom()
              
 # massively ugly/ not tested yet
class Strategy:
  def __init__(self, predicted_values : list, predicted_policy : list, strategy : str, max_trials : int):
    self.predicted_values = np.array(predicted_values)
    self.predicted_policy = np.array(predicted_policy)
    self.strategy = strategy
    self.max_trials = max_trials
    # Dispatch strategy methods based on the strategy name
    self.strategy_dispatcher = {
        "value_filter": self.value_filter,
        "best_max_trials": self.best_max_trials,
        "best_max_value": self.best_max_value,
        "topk_fusmedian": self.topk_fusmedian}

  def execute_strategy(self) -> list:
    # Execute the chosen strategy
    return self.strategy_dispatcher[self.strategy]()

  def value_filter(self) -> list:
    #Selects actions based on value, if none selected, selects top actions up to max_trials
    # Convert inputs to numpy arrays
    predicted_values, actions = np.array(self.predicted_values), np.array(self.predicted_policy)
    # Find indices where predicted values are negative
    negative_indices = np.where(predicted_values < 0)[0]
    # Round the negative predicted values to 2 decimal places
    rounded_values = np.round(predicted_values[negative_indices], 2)
    # Find unique values and their counts
    unique_values, counts = np.unique(rounded_values, return_counts=True)
    # Find unique values that occur only once
    unique_indices = unique_values[counts == 1]
    # Find indices of unique values in rounded values
    no_duplicate_indices = np.isin(rounded_values, unique_indices)
    # Calculate standard deviation of unique values
    std_dev = np.std(rounded_values[no_duplicate_indices])
    # Find values greater than or equal to standard deviation
    stdev_indices = np.abs(rounded_values[no_duplicate_indices]) >= std_dev
    # Sort indices based on values
    sorted_indices = np.argsort(rounded_values[no_duplicate_indices][stdev_indices])
    # Select actions based on sorted indices
    filter_action = actions[negative_indices][no_duplicate_indices][stdev_indices][sorted_indices].tolist()
    # If actions are selected, return them. Else, return top actions up to max_trials
    return negative_indices[no_duplicate_indices][stdev_indices][sorted_indices].tolist() if len(filter_action) != 0 else np.argsort(-1 * self.predicted_values)[:self.max_trials].tolist()

  def best_max_trials(self) -> list:
    #Selects the top actions up to max_trials
    # Return top actions up to max_trials
    return np.argsort(self.predicted_policy)[::-1][:self.max_trials].tolist()

  def best_max_value(self) -> list:
    #Selects the top actions based on value up to max_trials
    # Sort indices based on predicted values and select top actions up to max_trials
    return np.argsort(-1 * self.predicted_values)[::-1][:self.max_trials].tolist()

  def topk_fusmedian(self) -> list:
    #A complex strategy involving sorting, intersection, and median calculation
    #to provide a selection of actions based on a combination of common elements and median values.
    # Sort indices based on predicted values in descending order and select top 20
    value_sort_indices = (np.argsort(-1 * self.predicted_values)[::-1])[:20]
    # Find corresponding actions for sorted indices
    final_indices = self.predicted_policy[value_sort_indices]
    # Find common actions between final_indices and top 20 actions
    common_elements = np.intersect1d(final_indices, self.predicted_policy[:20])
    # Calculate weights for common elements based on their occurrences
    weights = len(final_indices) + len(self.predicted_policy[:20]) - \
              np.searchsorted(final_indices, common_elements) - \
              np.searchsorted(self.predicted_policy[:20], common_elements)
    # Sort common elements based on weights in descending order and select top 10
    sorted_common_elements = common_elements[np.argsort(-weights)].tolist()[:10]
    # Select top actions up to max_trials as an obvious solution
    obvious_solution = self.best_max_trials()
    # Calculate median index of sorted common elements
    median_idx = len(sorted_common_elements) // 2
    # Select median element and its neighbors
    median_and_neighbors = sorted_common_elements[max(0, median_idx-1) : median_idx+2]
    # Find actions in median_and_neighbors that are not part of the obvious solution
    depth_solution = np.setdiff1d(median_and_neighbors, obvious_solution)
    # Concatenate the obvious solution and depth_solution to provide a final set of actions
    return np.concatenate((self.best_max_trials(), depth_solution)).tolist()

def inference(available_kernels : List[str], max_number_of_step : int = 20,
              path_save_episode : str = "/home/test1/", model_path : str = "model_all.safetensors",
              max_trial : int = 3, strategies : int = "topk+fusmedian") -> List[Linearizer]:
  
  #load kernel to optimize
  os.remove(path_save_episode + 'dataset_inference.db') if os.path.exists(path_save_episode + 'dataset_inference.db') else None
  db = KernelEnv(available_kernels=available_kernels, db_path = path_save_episode + 'dataset_inference.db', inference_mode=True)
  
  #load model
  model=PPO_tiny()
  full_model_path = os.path.join(path_save_episode, model_path)
  load_state_dict(model, safe_load(full_model_path)) if os.path.exists(full_model_path) else None
  if model is None: raise(f"Indalid model path at {model_path}")
  
  #init inference var
  result_states = [] #ugly
  len_dataset = len(available_kernels) #ugly
  del available_kernels , db #ugly
  
  # print("Strategy: ",strategies)
  for kernel in range(len_dataset):
    # Reset for new episode
    env = KernelEnv(db_path = path_save_episode + 'dataset_inference.db', inference_mode=True, index_to_pick=kernel, max_n_mouve = 60)
    state, _ = env.reset()
    if state is None : return None
    Tensor.ones(1,2,model.hidden_space_lstm_size, dtype=model.c_type, requires_grad=False)
    max_moves, reward_state_pairs = 0, []
    
    #optimize kernel
    while max_moves < max_number_of_step: 
      cache, value_cache = [], []
      #get kernel without apply move
      initial_kernel = env.linearized_kernel
      #get policy action distribution
      action_prob, new_hidden_state = model.pi(Tensor(state.tolist(), dtype=model.c_type, requires_grad=False), hidden_state)
      sorted_indices = action_prob.flatten().detach().numpy().argsort()[::-1] 
      
      #get value action distribution
      value_cache = []
      for action in sorted_indices:
        env.linearized_kernel = initial_kernel
        next_state, _, _, _, _ = env.step(action, no_reward=True)
        env.done = False
        next_action_values = model.v(Tensor(next_state.tolist(), dtype=model.c_type, requires_grad=False), new_hidden_state)
        value_cache.append(next_action_values.flatten().detach().numpy()[0])
        
      #pick a strategy
      selected_actions = (strategy := Strategy(value_cache, action_prob.flatten().detach().numpy().tolist(), strategy=strategies, max_trials=max_trial)).execute_strategy()
      #trial and error on selected strategy and keep the best apply action
      for action in selected_actions:
        env.linearized_kernel = initial_kernel
        next_state, reward, done, _, orr = env.step(action)
        env.done = False
        cache.append((orr if reward != env.terminal_reward else env.i_r , action, done,next_state,env.linearized_kernel))
        # if not done: break ##if you want to select the first working mouve only
        
      #cache best result
      env.linearized_kernel, done = min(cache)[-1], min(cache)[-3] 
      state, hidden_state = min(cache)[-2], new_hidden_state
      max_moves += 1 #ugly
      reward_state_pairs.append(min(cache)) #ugly
      if done: break #ugly
      
    if kernel % 10 == 0: force_oom()#force cuda vram reset
    #store and print step result
    best_reward,best_kernel =  min(reward_state_pairs)[0], min(reward_state_pairs)[-1] #ugly
    action = [act for speed, act, done, *_ in reward_state_pairs if speed <= env.i_r and not done] #ugly
    result_states.append([kernel, env.i_r/best_reward,best_reward, env.i_r, best_reward, best_kernel, action ]) #ugly
    kernel_shape = re.sub('__', '_', re.sub(' +', '_', result_states[-1][-2].colored_shape()).strip('_') ) #ugly
    print(f"| {'Kernel:':<7} {kernel:>4}| {'Init Spd:':<10} {(env.i_r)*1000:>12.3f} ms | {'Up:':<5} {(env.i_r / best_reward):>10.3f}x | {'New Spd:':<10} {best_reward*1000:>12.3f} ms | {'Act:':<5} {str(action):<60} | {'Kernel shape:':<5} {kernel_shape:<37}")

  #print overall result
  total_time_original = np.array([i[3] for i in result_states]).sum() * 1000
  total_time_optimized = np.array([i[2] for i in result_states]).sum() * 1000
  total_speedup = total_time_original / total_time_optimized
  print(f"TOTAL SpeedUP: {total_speedup:.3f}x, Previous Total time: {total_time_original:.3f} ms, New total time: {total_time_optimized:.3f} ms")
  return [i[-2] for i in result_states]

  
#ENTRY POINT
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Training or Inference')
  #by default without infer it will perform training
  parser.add_argument('--infer', help='Perform inference', action='store_true')
  #share parameter between inference and training
  parser.add_argument('--path', help='Path to save episode', default='/home/hotmil/ubuntu/TinyRL/test5/')
  parser.add_argument('--model_name', help='model name', default='model_all.safetensors')
  parser.add_argument('--max_steps_limit', type=int, default=60)
  #inference parameter
  parser.add_argument('--max_inference_trial', type=int, default=3)
  parser.add_argument('--inference_strategy', type=str, default="topk+fusmedian")
  #training parameter
  parser.add_argument('--learning_rate', type=float, default=0.0003)
  parser.add_argument('--gamma', type=float, default=0.97)
  parser.add_argument('--lmbda', type=float, default=0.90)
  parser.add_argument('--eps_clip', type=float, default=0.1)
  parser.add_argument('--K_epoch', type=int, default=2)
  parser.add_argument('--state_size', type=int, default=240)
  parser.add_argument('--action_size', type=int, default=1 + len(actions))  # Assuming actions is defined elsewhere
  parser.add_argument('--value_size', type=int, default=1)
  parser.add_argument('--hidden_space_size', type=int, default=1024)
  parser.add_argument('--minimum_batch_size', type=int, default=40)
  parser.add_argument('--episode_count', type=int, default=2000)
  parser.add_argument('--print_interval', type=int, default=1)
  args = parser.parse_args()
  os.makedirs(args.path, exist_ok=True)
  
  if args.infer:
    from tinygrad.tensor import Tensor
    from models.resnet import ResNet50
    mdl = ResNet50()
    example = Tensor.empty(64, 3, 224, 224)
    available_kernels = extract_ast_kernels(mdl,example)
    del mdl, example
    
    start_time = time.time()
    result = inference(available_kernels,
                        max_number_of_step=args.max_steps_limit,
                        path_save_episode = args.path,
                        model_path = args.model_name,
                        max_trial = args.max_inference_trial,
                        strategies= args.inference_strategy)
    print(f"Inference elapsed time: {format_time(time.time() - start_time)}")
  else:
    # Initialize database of the environment
    # if load_val(default = 0,filename = "state.pkl") == 0 :
    dataset = load_worlds(True, True, filter_novariable=True)
    load_db = KernelEnv(available_kernels=dataset, db_path = args.path + 'dataset_training.db')
    del load_db, dataset
    # force_oom()

    training(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lmbda=args.lmbda,
        eps_clip=args.eps_clip,
        K_epoch=args.K_epoch,
        state_size=args.state_size,
        action_size=args.action_size,
        value_size=args.value_size,
        hidden_space_size=args.hidden_space_size,
        minimum_batch_size=args.minimum_batch_size,
        max_steps_limit=args.max_steps_limit,
        episode_count=args.episode_count,
        print_interval=args.print_interval,
        path_save_episode = args.path,
        model_path= args.model_name
    )
