import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.models import Model, clone_model
from itertools import product

def make_states(n):
    states = np.zeros((n,8))
    for i in np.arange(n):
        idx = np.random.choice(8, size=3, replace=False)
        states[i, idx] = np.random.uniform(0.0, 1.0, size=3)
    return states
    
def run(state, action):
    if state[action] != 0:
        return state[action]
    else:
        before, after = state[(action - 1) % 8], state[(action + 1) % 8]
        if before != 0 and after != 0:
            return before + after
        else:
            return 0

def get_optimal_action_and_value(state):
    values = np.array([run(state, action) for action in np.arange(8)])
    optimal_action = values.argmax()
    return optimal_action, values[optimal_action]

def get_optimal_actions_and_values(state_list):
    res = [get_optimal_action_and_value(state) for state in state_list]
    actions = np.array([pairs[0] for pairs in res])
    values = np.array([pairs[1] for pairs in res])
    return actions, values
    
def get_optimal_statics(n_rounds):
    state_list = make_states(n_rounds)
    _, values = get_optimal_actions_and_values(state_list)
    return np.mean(values), np.std(values)

def get_random_policy_statics(n_rounds):
    values = [run(state, np.random.choice(8)) for state in make_states(n_rounds)]
    return np.mean(values), np.std(values)

def get_model_actions(model, state_list):
    return model(np.array(state_list)).numpy().argmax(axis = 1)

def test_model(model, n_test_rounds):
    state_list = make_states(n_test_rounds)
    actions = get_model_actions(model, state_list)
    values = [run(state, action) for state, action in zip(state_list, actions)]
    optimal_actions, _ = get_optimal_actions_and_values(state_list)
    accuracy = np.mean(actions == optimal_actions)
    return np.mean(values), np.std(values), accuracy

def one_batch(model, optimizer, batch_size=128):
    states = make_states(batch_size)
    _, optimal_values = get_optimal_actions_and_values(states)
    with tf.GradientTape() as tape:
        probs = model(states)

        actions = np.array([np.random.choice(8, p=prob) for prob in probs.numpy()])
        rewards = np.array([run(state, action) for state, action in zip(states, actions)])
        rewards = rewards - optimal_values
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        action_probs = tf.gather_nd(probs, [[i, a] for i, a in enumerate(actions)])
        log_probs = tf.math.log(action_probs)
        loss = -tf.reduce_mean(log_probs * rewards)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# def one_batch_supervised(model, batch_size, n_test_rounds=0):
#     state_list = make_states(batch_size)
#     y_target_list = np.zeros((batch_size, 8))
#     optimal_actions, _ = get_optimal_actions_and_values(state_list)
#     for i in np.arange(batch_size):
#         y_target_list[i, optimal_actions[i]] = 1
#     model.fit(np.array(state_list), np.array(y_target_list), verbose=0)
#     if n_test_rounds > 0:
#         return test_model(model, n_test_rounds)

def train(one_batch, optimizer, model, max_batch=200, batch_size=128, test_lapse = 10, n_test_rounds=10000, verbose = 0):
    best_weights = []
    best_idx = 0
    best_score = 0
    for i in np.arange(max_batch):
        one_batch(model, optimizer, batch_size)
        if (i+1) % test_lapse == 0 or i + 1 == max_batch:
            score, _, accuracy = test_model(model, n_test_rounds)
            if best_score < score:
                best_score = score
                best_idx = i
                best_weights = model.get_weights()
            if verbose == 1:
                print(i, score, accuracy)
    return best_idx, best_score, best_weights

# def train_supervised(one_batch_supervised, model, max_batch=200, batch_size=128, n_test_rounds=10000, verbose = 0):
#     best_weights = []
#     best_idx = 0
#     best_score = 0
#     for i in np.arange(max_batch):
#         score, std, accuracy = one_batch_supervised(model, batch_size, n_test_rounds)
#         if best_score < score:
#             best_score = score
#             best_idx = i
#             best_weights = model.get_weights()
#         if verbose == 1:
#             print(i, score, accuracy)
#     return best_idx, best_score, best_weights


def create_model(n_hidden_layers, n_dense_units, ratio_dropout):
    input_shape = (8,) 
    inputs = Input(shape=input_shape)

    x = inputs
    for i in np.arange(n_hidden_layers):
        x = Dense(n_dense_units)(x) 
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(ratio_dropout)(x)
    
    outputs = Dense(8, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
