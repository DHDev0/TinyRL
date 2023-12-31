############## TOOLKITS ##############
import numpy as np
def selector_np(predicted_values, actions):
    predicted_values_np, actions_np = np.array(predicted_values), np.array(actions)
    negative_indices = np.where(predicted_values_np < 0)[0]
    rounded_values = np.round(predicted_values_np[negative_indices], 2)
    unique_values, counts = np.unique(rounded_values, return_counts=True)
    unique_indices = unique_values[counts == 1]
    no_duplicate_indices = np.isin(rounded_values, unique_indices)
    std_dev = np.std(rounded_values[no_duplicate_indices])
    stdev_indices = np.abs(rounded_values[no_duplicate_indices]) >= std_dev
    sorted_indices = np.argsort(rounded_values[no_duplicate_indices][stdev_indices])
    return actions_np[negative_indices][no_duplicate_indices][stdev_indices][sorted_indices].tolist()

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

import time
def format_time(elapsed_time):
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, remainder = divmod(remainder, 1)
    remainder = remainder * 1e6
    milliseconds, remainder = divmod(int(remainder), 1000)
    microseconds = int(remainder)
    return f"{int(hours):02} H {int(minutes):02} min {int(seconds):02} sec {int(milliseconds):02} ms {int(microseconds):02} us "

from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
def force_oom():
    if 1 in [getenv("TRITON") ,getenv("CUDA")]:
        try: large_tensor = Tensor.empty(10000000000000).realize()
        except Exception as e: pass

def split_and_find_max(list_of_lists):
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
import copy
def extract_ast_kernels(mdl,example) -> List[Any]:
    device: Compiled = Device[Device.DEFAULT]
    print(f"optimizing for {Device.DEFAULT}")
    #seen = set()
    mdl(example).lazydata.schedule()
    md = mdl(copy.deepcopy(example))
    sched = md.lazydata.schedule()
    sched = [str(x.ast) for x in sched if x.ast.op not in LoadOps]
    print(len(sched))
    return sched

import numpy as np
def count_pos_neg(arr):
    pos_count , neg_count = np.sum(arr > 0) , np.sum(arr < 0)
    return pos_count, neg_count

def print_to_file(file_path, text):
    with open(file_path, 'a') as file:
        print(text, file=file)

import pickle
def save_val(episode,filename="state.pkl"):
    with open(f"{filename}", "wb") as f:pickle.dump(episode, f)
def load_val(default=0,filename="state.pkl"):
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
                 available_kernels = None,
                 db_path = None,
                 inference_mode = False,
                 index_to_pick = 0, 
                 max_n_mouve = 60,
                 max_retries = 1000,
                 terminal_reward = -1):
        self.terminal_reward = terminal_reward#-(math.log(GLOBAL_TIMEOUT)*np.pi)
        self.index = index_to_pick
        self.inference_mode = inference_mode
        # state_size = 240 #hardcoded in lin_to_feats()
        # action_size = 1+len(actions) # discret action space
        # value_size = 1 #learn the Xspeedup reward, continuous value
        # # hidden_space_size = whatever you want
        # # print("OBS_size: ",state_size," ACTION_size: ",action_size," VALUE_size: ",value_size, " kernel dataset size: ",len(self.available_kernels))
        # self.action_space = spaces.Discrete(1+len(actions))
        # self.observation_space = spaces.Box(-np.ones(state_size), np.ones(state_size), dtype=np.float32)
        # self.render_mode = render_mode
        
        self.db_path = db_path
        self.max_n_mouve = max_n_mouve
        self.max_retries = max_retries
        self.init_db()
        if available_kernels is not None: self.add_kernels(available_kernels)
        self.max_global_size= 65536

    def init_db(self):
        if not os.path.exists(self.db_path):
            self.update_db("CREATE TABLE kernels (id INTEGER PRIMARY KEY, kernel_data BLOB, last_num_steps INTEGER, total_visits INTEGER)")

    def query_db(self, query, args=()):
        for retry in range(self.max_retries):
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

    def update_db(self, query, args=()):
        for retry in range(self.max_retries):
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

    def add_kernels(self, kernel_list):
        for kernel in kernel_list:
            serialized_kernel = pickle.dumps(kernel)
            if not self.inference_mode:
                res = self.query_db("SELECT * FROM kernels WHERE kernel_data = ?", (serialized_kernel,))
                if len(res) == 0: self.update_db("INSERT INTO kernels (kernel_data, last_num_steps, total_visits) VALUES (?, 0, 0)", (serialized_kernel,))
            else: self.update_db("INSERT INTO kernels (kernel_data, last_num_steps, total_visits) VALUES (?, 0, 0)", (serialized_kernel,))
            
    def priority_game(self):
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

    def update_kernel_step(self, kernel_id):
        self.update_db(f"UPDATE kernels SET last_num_steps = last_num_steps + 1 WHERE id = {kernel_id}")

    def get_kernel(self, kernel_id):
        data = self.query_db(f"SELECT kernel_data FROM kernels WHERE id = {kernel_id}")
        serialized_kernel = data[0][0]
        return pickle.loads(serialized_kernel)
    
    def delete_kernel(self, kernel_id):
        self.update_db(f"DELETE FROM kernels WHERE id = {kernel_id}")

    def count_kernels(self):
        return self.query_db("SELECT COUNT(*) FROM kernels")[0]
    
    def reset(self):
        while True:
            if self.inference_mode:
                self.kernel_pick_index = self.index
                kernel_pick = self.get_kernel(self.kernel_pick_index)
            else:
                self.kernel_pick_index = self.priority_game()
                kernel_pick = self.get_kernel(self.kernel_pick_index) 
            
            self.max_regret = []            
            try:
                linearized_kernel = ast_str_to_lin(kernel_pick)
                buffer_info = bufs_from_lin(linearized_kernel)
                init_reward = time_linearizer(linearized_kernel, buffer_info, allow_test_size=True,should_copy=True, max_global_size=self.max_global_size)
                if math.isinf(init_reward) or np.isnan(init_reward):
                    raise ValueError("Invalid initial time.")
                init_state = np.array(lin_to_feats(linearized_kernel))
                self.init_reward , self.init_state, self.linearized_kernel = init_reward,init_state, linearized_kernel
                self.done = False
                self.max_regret.append(self.init_reward)
                return self.init_state, None #gymnasium style
                # return self.init_state #gym style
            except Exception as e:
                print("REMOVE KERNEL AFTER ERROR: ",e)
                self.delete_kernel(self.kernel_pick_index)
                print(f"Number of active kernels after removal: {self.count_kernels()}")
                
    def step(self, action_idx,no_reward=False):
        reward = self.terminal_reward
        obtained_reward = self.terminal_reward
        if self.done:
            raise Exception("Episode already done, reset the environment first.")
        if action_idx >= 0:
            try:
                action_to_apply = actions[action_idx - 1]
                linearized_kernel_copy = self.linearized_kernel.copy()
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
                if up > 256 or lcl > 256:
                    raise ValueError("Invalid action: exceeds workgroup size constraints.")
                if not no_reward:
                    buffer_info = bufs_from_lin(linearized_kernel_copy)
                    
                    # Calculate and return the reward
                    compute_time = time_linearizer(linearized_kernel_copy, buffer_info, allow_test_size=True, should_copy=True, max_global_size=self.max_global_size)
                    if math.isinf(reward) or np.isnan(reward):
                        raise ValueError("Invalid reward.")
                
                    self.linearized_kernel , obtained_reward = linearized_kernel_copy , compute_time
                    
                    reward = 1 if obtained_reward < min(self.max_regret) else -1
                # print(linearized_kernel_copy.printbufs())
            except Exception as e: 
                # print(f"ILLEGAL MOVE ERROR: {e}") 
                pass

        self.done = action_idx == 0 or reward == self.terminal_reward
        state = self.next_state if hasattr(self, 'next_state') else self.init_state
        if not self.done or no_reward:
            try:
                state = np.array(lin_to_feats(linearized_kernel_copy))
            except Exception as e:
                # print(e)
                self.done = True
                state = (state*0)+np.pi
                reward = self.terminal_reward
                self.linearized_kernel = ast_str_to_lin(self.get_kernel(self.kernel_pick_index))
        else:
            state = (state*0)+np.pi
            reward = self.terminal_reward
            self.linearized_kernel = ast_str_to_lin(self.get_kernel(self.kernel_pick_index))
            
        self.next_state = state
        self.max_regret.append(obtained_reward  if reward!=self.terminal_reward else self.max_regret[0])
        if not self.inference_mode: self.update_kernel_step(self.kernel_pick_index)
        # if reward >= 1 : self.add_kernels([self.linearized_kernel])
        return self.next_state, reward, self.done, None, obtained_reward #gymnasium style
        # return next_state, reward, self.done, info #gym style

    def close(self):
        pass

############## MODEL #############
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.set_default_device('cuda')

class PPO(nn.Module):
    def __init__(self, state_size=4, 
                 action_size=2, 
                 value_size=1, 
                 hidden_space_size=64, 
                 learning_rate=0.0001,
                 gamma=0.97, 
                 lmbda=0.90, 
                 eps_clip=0.1, 
                 K_epoch=2):
        super(PPO, self).__init__()
        
        self.gamma = gamma 
        self.lmbda = lmbda 
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.loss = 0
        self.data = []
        self.hidden_space_size = hidden_space_size
        self.hidden_space_lstm_size = hidden_space_size // 2
        self.fc1 = nn.Linear(state_size, hidden_space_size)
        self.layer_norm1 = nn.LayerNorm(hidden_space_size)
        self.lstm = nn.LSTM(hidden_space_size, hidden_space_size // 2)
        self.layer_norm_lstm = nn.LayerNorm(hidden_space_size // 2)
        self.fc_pi = nn.Linear(hidden_space_size // 2, action_size)
        self.layer_norm_pi = nn.LayerNorm(action_size)
        self.fc_v = nn.Linear(hidden_space_size // 2, value_size)
        self.layer_norm_v = nn.LayerNorm(value_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.activation = nn.LeakyReLU()

    def pi(self, x, hidden):
        x = self.activation(self.fc1(x))
        # x = self.layer_norm1(x)
        x = x.view(-1, 1, self.hidden_space_size)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.layer_norm_lstm(x)
        x = self.fc_pi(x)
        # x = self.layer_norm_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden, show_hidden=False):
        x = self.activation(self.fc1(x))
        # x = self.layer_norm1(x)
        x = x.view(-1, 1, self.hidden_space_size)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.layer_norm_lstm(x)
        v = self.fc_v(x)
        # v = self.layer_norm_v(v)
        if show_hidden: v , lstm_hidden
        return v
    
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), \
                                         torch.tensor(a_lst), \
                                         torch.tensor(r_lst), \
                                         torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), \
                                         torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        self.train()
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for _ in range(self.K_epoch):
            
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.cpu().detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())
            
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=False)
            self.optimizer.step()
        self.loss= loss.mean().detach().cpu().numpy()
        

############## CYCLE TRAIN/INF ##############
def training(learning_rate=0.0003, gamma=0.97,
             lmbda=0.90, eps_clip=0.1, 
             K_epoch=2, state_size=240, 
             action_size=1 + len(actions), 
             value_size=1, hidden_space_size=512,
             minimum_batch_size=60, max_steps_limit=60, 
             episode_count=2000, print_interval=1,
             path_save_episode = "/home/test1/",model_path= "model_all.pth"):
    
    full_model_path = os.path.join(path_save_episode, model_path)
    model = torch.load(full_model_path) if os.path.exists(full_model_path) else PPO(state_size=state_size, 
                                                                          action_size=action_size, 
                                                                          value_size=value_size, 
                                                                          hidden_space_size=hidden_space_size,
                                                                          learning_rate=learning_rate,
                                                                          gamma=gamma, 
                                                                          lmbda=lmbda, 
                                                                          eps_clip=eps_clip, 
                                                                          K_epoch=K_epoch)
    # Initialize episode variables
    current_episode = load_val(default = 0,filename = path_save_episode+"state.pkl")
    episode_scores = []
    action_count = load_val(default=np.zeros(1 + len(actions)),filename = path_save_episode+"action_count.pkl")  # Initialize action count for each action
    
    while current_episode < episode_count:
        # Reset for new episode
        env = KernelEnv(db_path = path_save_episode + 'dataset_training.db',max_n_mouve = 60)
        model.eval()
        state, _ = env.reset()
        done = False
        max_steps = 0
        max_c = 2
        c = 2 #max_c * np.exp(-10 * current_episode / episode_count) #2.0  #1.0 # 0.5 # UCB exploration parameter
        hidden_state = (torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float),
                        torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float))
        
        # Collect data for one episode
        while not (done or len(model.data) >= minimum_batch_size or max_steps == max_steps_limit):
            # UCB Action Selection
            action_prob, new_hidden_state = model.pi(torch.from_numpy(state).float().cuda(), hidden_state)
            # action_values = model.v(torch.from_numpy(state).float().cuda(), new_hidden_state) 
            ucb_values = action_prob.view(-1).detach().cpu().numpy() + c * np.sqrt(np.log(1+max_steps) / (1 + action_count)) 
            action = Categorical(torch.tensor(ucb_values/ucb_values.sum(), dtype=torch.float32, device='cpu')).sample().item() #np.argmax(ucb_values)
            # Take action and observe next state and reward
            next_state, reward, done, _, orr = env.step(action)
            model.put_data((state, action, reward, next_state, action_prob.view(-1)[action].item(), hidden_state, new_hidden_state, done))
            del state , hidden_state
            state , hidden_state = next_state , new_hidden_state
            modified_reward = env.init_reward/orr if reward != env.terminal_reward else -1
            episode_scores.append([modified_reward , done , action])
            action_count[action] += 1  # Update action count
            max_steps += 1

        force_oom()
    
        
        if len(model.data) >= minimum_batch_size:
            model.train_net()
            if math.isinf(model.loss) or np.isnan(model.loss): raise ("gradient vanish")
            current_episode += 1
            save_val(current_episode ,filename= path_save_episode+"state.pkl")
            save_val(action_count ,filename = path_save_episode+"action_count.pkl")
            
            if current_episode % print_interval == 0:
                avg_score = np.array(split_and_find_max(episode_scores))
                pos, neg = count_pos_neg(avg_score)
                pr_loss= f"# of episode: {current_episode}, avg score: {avg_score.mean():.3f}, repartition positive: {(pos/(neg+pos)) * 100:.3f} % loss: {model.loss:.5f}"
                print(pr_loss)
                print_to_file(path_save_episode + 'state_model_all.txt', f"ep: {current_episode} | <Compute_time : Done : Action>: {[(a[0], a[1], a[2]) for i, a in enumerate(episode_scores)]}")
                print_to_file(path_save_episode + 'state_model_all.txt', "||||||||||||||||||||||||||")
                print_to_file(path_save_episode + 'loss_model_all.txt', pr_loss)
                episode_scores.clear()
                torch.save(model, full_model_path)
                if current_episode % 100 == 0: torch.save(model, os.path.join(path_save_episode, f"{current_episode}_episode_"+model_path))
            # if current_episode % 1 == 0: force_oom()
                

def inference(available_kernels,max_number_of_step=20,
              path_save_episode = "/home/test1/",model_path= "model_all.pth",
              max_trial = 3, strategies = "topk+fusmedian"):
    
    model_path = os.path.join(path_save_episode, model_path)
    os.remove(path_save_episode + 'dataset_inference.db') if os.path.exists(path_save_episode + 'dataset_inference.db') else None
    db = KernelEnv(available_kernels=available_kernels, db_path = path_save_episode + 'dataset_inference.db', inference_mode=True)
    len_dataset = len(available_kernels)
    del available_kernels , db
    
    model = torch.load(model_path) if os.path.exists(model_path) else None
    result_states = []
    cache=[]
    value_cache=[]
    remainder=0
    print("Strategy: ",strategies)
    for kernel in range(len_dataset):
        env = KernelEnv(db_path = path_save_episode + 'dataset_inference.db', inference_mode=True, index_to_pick=kernel+1, max_n_mouve = 60)
        max_moves = 0
        reward_state_pairs = []
        state, _ = env.reset()
        hidden_state = (torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float),
                        torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float))
        while max_moves < max_number_of_step: 
            #get policy action distribution
            with torch.no_grad():
                action_prob, new_hidden_state = model.pi(torch.from_numpy(state).float().cuda(), hidden_state)
            # Convert to numpy and sort
            action_prob_np = action_prob.detach().flatten().cpu().numpy()
            sorted_indices = action_prob_np.argsort()[::-1]  # Sort in descending order
            #get value action distribution
            initk = env.linearized_kernel
            value_cache = []
            for action in sorted_indices:
                next_state, _, _, _, _ = env.step(action, no_reward=True)
                env.done = False
                next_action_values = model.v(torch.from_numpy(next_state).float().cuda(), new_hidden_state)
                value_cache.append(next_action_values.detach().flatten().cpu().numpy()[0])
                env.linearized_kernel = initk
            
            #pick strategy
            if strategies == "valuefilter":
                filter_action = selector_np(value_cache, sorted_indices) 
                strategy = filter_action if len(filter_action) != 0 else sorted_indices[:max_trial]
            if strategies == "best_max_trial_policy":
                strategy = sorted_indices[:max_trial]
            if strategies == "best_max_trial_value":
                strategy = sorted_indices[(np.argsort(-1*np.array(value_cache))[::])[:max_trial]]
            if strategies == "topk+fusmedian": # need a general way to define the slice parameter
                value_sort_indices = (np.argsort(-1*np.array(value_cache))[::-1])[:20]
                final_indices = sorted_indices[value_sort_indices] 
                common_elements_np = np.intersect1d(final_indices, sorted_indices[:20])
                weights = len(final_indices) + len(sorted_indices[:20]) - np.searchsorted(final_indices, common_elements_np) - np.searchsorted(sorted_indices[:20], common_elements_np)
                sorted_common_elements = common_elements_np[np.argsort(-weights)].tolist()[:10]
                obvious_solution= sorted_indices[:max_trial]
                median_idx = len(sorted_common_elements) // 2
                median_and_neighbors = sorted_common_elements[max(0, median_idx-1) : median_idx+2]
                depth_solution = np.setdiff1d(median_and_neighbors, obvious_solution)
                strategy = np.concatenate((obvious_solution[:max_trial], depth_solution))

            for action in strategy:
                env.linearized_kernel = initk
                next_state, reward, done, _, orr = env.step(action)
                env.done = False
                cache.append((orr if reward != env.terminal_reward else env.init_reward , action, done,next_state,env.linearized_kernel))
                if not done: break

            env.linearized_kernel = min(cache)[-1]
            done = min(cache)[-3]
            reward_state_pairs.append(min(cache))
            state , hidden_state = min(cache)[-2] , new_hidden_state
            max_moves += 1
            cache=[]
            value_cache=[]
            if done: break
        # torch.cuda.empty_cache()
        if kernel % 10 == 0: force_oom()
        best_reward,best_kernel =  min(reward_state_pairs)[0], min(reward_state_pairs)[-1]
        action = [act for speed, act, done, *_ in reward_state_pairs if speed <= env.init_reward and not done]
        result_states.append([kernel ,env.init_reward / best_reward,best_reward ,env.init_reward, best_reward,best_kernel,action ])
        kernel_shape = re.sub(' +', '_', result_states[-1][-2].colored_shape()).strip('_')
        kernel_shape = re.sub('__', '_', kernel_shape)
        print(f"| {'Kernel:':<7} {kernel + remainder:>4}| {'Init Spd:':<10} {(env.init_reward)*1000:>12.3f} ms | {'Up:':<5} {(env.init_reward / best_reward):>10.3f}x | {'New Spd:':<10} {best_reward*1000:>12.3f} ms | {'Act:':<5} {str(action):<60} | {'Kernel shape:':<5} {kernel_shape:<37}")


    total_time_original = np.array([i[3] for i in result_states]).sum() * 1000
    total_time_optimized = np.array([i[2] for i in result_states]).sum() * 1000
    total_speedup = total_time_original / total_time_optimized
    print(f"TOTAL SpeedUP: {total_speedup:.3f}x, Previous Total time: {total_time_original:.3f} ms, New total time: {total_time_optimized:.3f} ms")
    return result_states




#ENTRY POINT
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training or Inference')
    #by default without infer it will perform training
    parser.add_argument('--infer', help='Perform inference', action='store_true')
    #share parameter between inference and training
    parser.add_argument('--path', help='Path to save episode', default='/home/usr/ubuntu/TinyRL/test3/')
    parser.add_argument('--model_name', help='model name', default='model_all.pth')
    parser.add_argument('--max_steps_limit', type=int, default=20)
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
    parser.add_argument('--minimum_batch_size', type=int, default=60)
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
        force_oom()
        
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
        force_oom()

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

        
