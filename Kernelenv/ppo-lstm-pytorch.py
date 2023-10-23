
############## HYPERPARAMETER##############
llearning_rate = 0.0003
gamma = 0.97
lmbda = 0.90
eps_clip = 0.1
K_epoch = 2
minimum_batch_size = 60
max_steps_limit = 60
episode_count = 2000
print_interval = 1
path_save_episode = '/project/folder/path/'
model_path = path_save_episode+'model_all.pth'
import os; os.makedirs(path_save_episode, exist_ok=True)

############## TOOLKITS ##############

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


def print_to_file(file_path, text):
    with open(file_path, 'a') as file:
        print(text, file=file)

# #extract custom kernel to use in KernelEnv
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from typing import List, Any
def extract_ast_kernels(mdl,example) -> List[Any]:
    device: Compiled = Device[Device.DEFAULT]
    print(f"optimizing for {Device.DEFAULT}")
    seen = set()
    mdl(example).lazydata.schedule(seen)
    md = mdl(example)
    sched = md.lazydata.schedule(seen)
    sched = [str(x.ast) for x in sched if x.ast.op not in LoadOps]
    return sched


import numpy as np
def count_pos_neg(arr):
    pos_count , neg_count = np.sum(arr > 0) , np.sum(arr < 0)
    return pos_count, neg_count

import pickle
def save_val(episode,filename="state.pkl"):
    with open(f"{path_save_episode}{filename}", "wb") as f:pickle.dump(episode, f)
def load_val(default=0,filename="state.pkl"):
    try: 
        with open(f"{path_save_episode}{filename}", "rb") as f: return pickle.load(f)
    except FileNotFoundError:return default
    
# import os
# import signal
# import sys
# def handle_segfault(signum, frame):
#     print("Segmentation fault detected! Restarting script...")
#     os._exit(0)


############## ENV ##############
import os
import sqlite3
import time
import numpy as np
import pickle
import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.features.search import actions, bufs_from_lin, time_linearizer
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats
from tinygrad.codegen.optimizer import Opt
import gc


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
        self.db_path = db_path
        self.max_n_mouve = max_n_mouve
        self.max_retries = max_retries
        self.init_db()
        if available_kernels is not None: self.add_kernels(available_kernels)
        self.max_global_size= 65536

    def __del__(self):
        del self

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
            res = self.query_db("SELECT * FROM kernels WHERE kernel_data = ?", (serialized_kernel,))
            if len(res) == 0: self.update_db("INSERT INTO kernels (kernel_data, last_num_steps, total_visits) VALUES (?, 0, 0)", (serialized_kernel,))

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
                linearized_kernel = ast_str_to_lin(kernel_pick).copy()
                buffer_info = bufs_from_lin(linearized_kernel).copy()
                init_reward = time_linearizer(linearized_kernel, buffer_info, allow_test_size=True,should_copy=True, max_global_size=self.max_global_size)
                if math.isinf(init_reward) or np.isnan(init_reward):
                    del linearized_kernel , buffer_info , init_reward
                    raise ValueError("Invalid initial time.")
                init_state = np.array(lin_to_feats(linearized_kernel).copy())
                self.init_reward , self.init_state, self.linearized_kernel = init_reward,init_state, str(linearized_kernel.ast)
                del linearized_kernel , buffer_info , init_reward
                self.done = False
                self.max_regret.append(self.init_reward)
                return self.init_state, None #gymnasium style
                # return self.init_state #gym style
            except Exception as e:
                print("REMOVE KERNEL AFTER ERROR: ",e)
                self.delete_kernel(self.kernel_pick_index)
                print(f"Number of active kernels after removal: {self.count_kernels()}")
                
    
    def step(self, action_idx):
        reward = self.terminal_reward
        obtained_reward = self.terminal_reward
        if self.done:
            raise Exception("Episode already done, reset the environment first.")
        if action_idx >= 0:
            try:
                action_to_apply = actions[action_idx - 1]
                linearized_kernel_copy = ast_str_to_lin(self.linearized_kernel).copy()
                # Check if action axis is within the shape length of the kernel
                if action_to_apply.axis >= linearized_kernel_copy.shape_len:
                    del linearized_kernel_copy
                    raise ValueError("Invalid action: axis out of bounds.")
                
                # Check if the action amount matches the axis size and a zero-change operation exists
                if (linearized_kernel_copy.full_shape[action_to_apply.axis] == action_to_apply.amt and 
                    Opt(action_to_apply.op, action_to_apply.axis, 0) in actions):
                    del linearized_kernel_copy
                    raise ValueError("Invalid action: action amount matches axis size and a zero-change operation exists.")
                
                # Tentatively apply the action to a copy to test for workgroup size constraints
                linearized_kernel_copy.apply_opt(action_to_apply)
                up, lcl = 1, 1
                for s, c in zip(linearized_kernel_copy.full_shape, linearized_kernel_copy.colors()):
                    if c in {"magenta", "yellow"}: up *= s
                    if c in {"cyan", "green", "white"}: lcl *= s
                if up > 256 or lcl > 256:
                    del linearized_kernel_copy
                    raise ValueError("Invalid action: exceeds workgroup size constraints.")
                buffer_info = bufs_from_lin(linearized_kernel_copy).copy()
                
                # Calculate and return the reward
                compute_time = time_linearizer(linearized_kernel_copy, buffer_info, allow_test_size=True, should_copy=True, max_global_size=self.max_global_size)
                if math.isinf(reward) or np.isnan(reward):
                    del linearized_kernel_copy , buffer_info , compute_time
                    raise ValueError("Invalid reward.")
                self.linearized_kernel , obtained_reward = str(linearized_kernel_copy.ast) , compute_time
                
                reward = 1 if obtained_reward < min(self.max_regret) else -1
                # print(linearized_kernel_copy.printbufs())
                del buffer_info,compute_time
            except Exception as e: 
                # print(f"ILLEGAL MOVE ERROR: {e}") 
                pass

        self.done = action_idx == 0 or reward == self.terminal_reward
        state = self.next_state if hasattr(self, 'next_state') else self.init_state
        if not self.done:
            try:
                state = np.array(lin_to_feats(linearized_kernel_copy).copy())
                del linearized_kernel_copy
            except Exception as e:
                # print(e)
                self.done = True
                state = (state*0)+np.pi
                reward = self.terminal_reward
        else:
            state = (state*0)+np.pi
            reward = self.terminal_reward
            
        self.next_state = state
        self.max_regret.append(obtained_reward  if reward!=self.terminal_reward else self.max_regret[0])
        if not self.inference_mode: self.update_kernel_step(self.kernel_pick_index)
        if reward >= 1 : self.add_kernels([self.linearized_kernel])
        if 'linearized_kernel_copy' in locals(): del linearized_kernel_copy
        return self.next_state, reward, self.done, None, obtained_reward #gymnasium style
        # return next_state, reward, self.done, info #gym style

    def close(self):
        pass


############## MODEL ##############

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np
import os

torch.set_default_device('cuda')

class PPO(nn.Module):
    def __init__(self, state_size=4, action_size=2, value_size=1, hidden_space_size=64, learning_rate=0.0001):
        super(PPO, self).__init__()
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
        del self.data
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        self.train()
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.cpu().detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())
            
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=False)
            self.optimizer.step()
        self.loss= loss.mean().detach().cpu().numpy()
        del s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out)
        
        
############## CYCLE TRAIN/INF ##############
def training():
    model = torch.load(model_path) if os.path.exists(model_path) else PPO(state_size=240, action_size=1 + len(actions), value_size=1, hidden_space_size=512)
    
    # Initialize episode variables
    current_episode = load_val(default = 0,filename = "state.pkl")
    episode_scores = []
    action_count = load_val(default=np.zeros(1 + len(actions)),filename = "action_count.pkl")  # Initialize action count for each action
    
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
            state = next_state
            hidden_state = new_hidden_state
            modified_reward = env.init_reward/orr if reward != env.terminal_reward else -1
            episode_scores.append([modified_reward , done , action])
            action_count[action] += 1  # Update action count
            max_steps += 1
        env.close()
        
        del env
        env = None
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        torch.cuda.empty_cache()
        
        if len(model.data) >= minimum_batch_size:
            with open("/home/hotmil/ubuntu/TinyRL/saving_model.txt", "w") as f: f.write("")
            model.train_net()
            current_episode += 1
            save_val(current_episode ,filename= "state.pkl")
            save_val(action_count ,filename = "action_count.pkl")
            
            if current_episode % print_interval == 0:
                avg_score = np.array(split_and_find_max(episode_scores))
                pos, neg = count_pos_neg(avg_score)
                pr_loss= f"# of episode: {current_episode}, avg score: {avg_score.mean():.3f}, repartition positive: {(pos/(neg+pos)) * 100:.3f} % loss: {model.loss:.5f}"
                print(pr_loss)
                print_to_file(path_save_episode + 'state_model_all.txt', f"ep: {current_episode} | <Compute_time : Done : Action>: {[(a[0], a[1], a[2]) for i, a in enumerate(episode_scores)]}")
                print_to_file(path_save_episode + 'state_model_all.txt', "||||||||||||||||||||||||||")
                print_to_file(path_save_episode + 'loss_model_all.txt',pr_loss)
                episode_scores.clear()
                torch.save(model, model_path)
                os.remove("/home/hotmil/ubuntu/TinyRL/saving_model.txt")

def inference(available_kernels,max_number_of_step=20):
    os.remove(path_save_episode + 'dataset_inference.db') if os.path.exists(path_save_episode + 'dataset_inference.db') else None
    db = KernelEnv(available_kernels=available_kernels, db_path = path_save_episode + 'dataset_inference.db')
    len_dataset = len(available_kernels)
    del available_kernels , db
    model = torch.load(model_path) if os.path.exists(model_path) else PPO(state_size=240, action_size=1 + len(actions), value_size=1, hidden_space_size=512)
    result_states = []
    action_register= []
    kernels_saver = []
    for kernel in range(len_dataset):
        env = KernelEnv(db_path = path_save_episode + 'dataset_inference.db', inference_mode=True, index_to_pick=kernel+1, max_n_mouve = 60)
        max_moves = 0
        reward_state_pairs = []
        state, _ = env.reset()
        hidden_state = (torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float),
                        torch.zeros([1, 1, model.hidden_space_lstm_size], dtype=torch.float))
        while max_moves < max_number_of_step: 
            with torch.no_grad():
                action_prob, new_hidden_state = model.pi(torch.from_numpy(state).float().cuda(), hidden_state)
            hidden_state = new_hidden_state
            action = Categorical(action_prob.view(-1)).sample().item()
            next_state, reward, done, _, orr = env.step(action)
            reward_state_pairs.append((orr if reward != env.terminal_reward else float("inf") , action, done))
            max_moves += 1
            state = next_state
            if done: break
            
        best_reward = min(reward_state_pairs)[0] if min(reward_state_pairs)[0] != float("inf") else env.init_reward
        action = [act for speed, act, done in reward_state_pairs if speed <= max(reward_state_pairs)[0] and [speed , done] != [float("inf"), True]]
        print(f"| Kernel: {kernel} | Initial compute speed: {(env.init_reward)*1000:.3f} ms | Speedup: {env.init_reward / best_reward:.3f}x | New compute speed: {best_reward*1000:.3f} ms | with action: {action} |" )
        
        result_states.append([kernel ,env.init_reward / best_reward,best_reward ,env.init_reward ]) # replace 0 by env..linearized_kernel
        action_register.append(action)
        kernels_saver.append(env.linearized_kernel)
        
        env.close()
        del env
        env = None
        gc.collect()
    # for i in result_states: print(f"| Kernel number: {i[0]} | speedup: {i[3]:.3f} | initial speed: {i[5]*1000:.3f} ms| new speed: {i[4]*1000:.3f} ms| | action: {i[2]} |")
    print(f"|TOTAL SpeedUP: {(np.array(result_states)[:,3].sum()*1000)/(np.array(result_states)[:,2].sum()*1000):.3f}x | Previous Total time: {np.array(result_states)[:,3].sum()*1000:.3f} ms --> New total time: {np.array(result_states)[:,2].sum()*1000:.3f} ms")
        
    #return result_states

############## RUNNER ##############
from tinygrad.tensor import Tensor
from models.resnet import ResNet50
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training or Inference')
    parser.add_argument('--infer', help='Perform inference', action='store_true')
    args = parser.parse_args()

    if args.infer:
        mdl = ResNet50()
        example = Tensor.ones(64, 3, 224, 224)
        available_kernels = extract_ast_kernels(mdl,example)
        del mdl
        del example
        inference(available_kernels)
    else:
        # Initialize database of the environment
        dataset = load_worlds(True, True, filter_novariable=True)
        load_db = KernelEnv(available_kernels=dataset, db_path = path_save_episode + 'dataset_training.db')
        del load_db, dataset
        training()



