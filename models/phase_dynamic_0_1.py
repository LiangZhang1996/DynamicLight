"""
Default network for DynamicLight-MQL
"""
from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Subtract, Add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import copy


class DynamicAgent01(NetworkAgent):
    def build_network(self):
        ins0 = Input(shape=(self.max_lane, self.num_feat2))
        ins1 = Input(shape=(1, self.num_phases))
        uni_dim, dims2 = 4, 16
        feat_list = tf.split(ins0, self.num_feat2, axis=2)
        feat_embs = [Dense(uni_dim, activation='sigmoid')(feat_list[i]) for i in range(self.num_feat2)]
        feats = tf.concat(feat_embs, axis=2) 
        feats = Dense(dims2, activation="relu")(feats)
        lane_feats_s = tf.split(feats, self.max_lane, axis=1)
        MHA1 = MultiHeadAttention(4, 16, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        phase_feats_map_2 = []
        for i in range(self.num_phases):
            tmp_feat_1 = tf.concat([lane_feats_s[idx] for idx in self.phase_map[i]], axis=1)
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_2)
            phase_feats_map_2.append(tmp_feat_3)
        phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
        selected_phase_feat = Lambda(lambda x: tf.matmul(x[0], x[1]))([ins1, phase_feat_all])
        selected_phase_feat = Reshape((dims2, ))(selected_phase_feat)

        hidden = Dense(20, activation="relu")(selected_phase_feat)
        hidden = Dense(20, activation="relu")(hidden)
        q_values = self.dueling_block(hidden)
        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()
        return network

    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        a = Dense(self.num_action_dur, activation="linear", name="dueling_advantages")(tmp_a)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantages = Subtract()([a, mean])
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values

    def phase_index2matrix(self, phase_index):
        # [batch, 1] -> [batch, 1, num_phase]
        lab = to_categorical(phase_index, num_classes=self.num_phases)
        return lab
    
    def choose_action(self, states, list_need):
        phase = []
        phase2 = []  
        dic_state_feature_arrays = {}
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1:])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            feat0 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][0]]
            tmp_idx = self.phase_control_policy(feat0)
            phase.append([[tmp_idx]])
            phase2.append(tmp_idx)
            
            for feat_name in used_feature:
                dic_state_feature_arrays[feat_name].append(s[feat_name])    

        phase_idx = np.array(phase)
        phase_matrix = self.phase_index2matrix(phase_idx)
        phase_feat2 = np.concatenate([np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), self.max_lane, -1) for feature_name in used_feature], axis=-1)
        
        q_values = self.q_network.predict([phase_feat2, phase_matrix])
        action = self.epsilon_choice(q_values)
        return phase2, action

    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_action_dur, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act

    def phase_control_policy(self, feat0):
        if self.num_lane == 8:
            feat10 = feat0[1] + feat0[3]
            feat20 = feat0[5] + feat0[7]
            feat30 = feat0[0] + feat0[2]
            feat40 = feat0[4] + feat0[6]
        elif self.num_lane == 10:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[9]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[8]
        elif self.num_lane == 12:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[10]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[9]
        elif self.num_lane == 16:
            feat10 = feat0[1] + feat0[2] + feat0[5] + feat0[6]
            feat20 = feat0[9] + feat0[10] + feat0[13] + feat0[14]
            feat30 = feat0[0] + feat0[4]
            feat40 = feat0[8] + feat0[12]
        idx = np.argmax([feat10, feat20, feat30, feat40])

        return idx

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1:]
        _state = [[] for _ in used_feature]
        _next_state = [[] for _ in used_feature]
        _action1 = []  # phase index
        _action2 = []
        _reward = []
        #  use average reward
        for i in range(len(sample_slice)):
            state, action1, action2, next_state, reward, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _action1.append([[action1]])
            _action2.append(action2)
            _reward.append(reward)

        _state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _state], axis=-1)
        _next_state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _next_state], axis=-1)

        phase_matrix = self.phase_index2matrix(np.array(_action1))

        cur_qvalues = self.q_network.predict([_state2, phase_matrix])
        next_qvalues = self.q_network_bar.predict([_next_state2, phase_matrix])
        # [batch, 4]
        target = np.copy(cur_qvalues)
        for i in range(len(sample_slice)):
            target[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = [_state2, phase_matrix]
        self.Y = target
