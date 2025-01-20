import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.models import Model, clone_model
from itertools import product

def make_state():
    state = np.zeros(8)
    state[np.random.choice(8, size=3, replace=False)] = np.random.uniform(size=3)
    return state

def run(state, action):
    state_cyclic = np.concatenate((state, state[0:2]))
    if state_cyclic[action] != 0:
        return state_cyclic[action]
    elif state_cyclic[action-1] != 0 and state_cyclic[action+1] != 0:
        return state_cyclic[action-1] + state_cyclic[action+1]
    else:
        return 0

def get_optimal_value(state):
    value = state.max()

    state_cyclic = np.concatenate((state, state[0:2]))
    for i in np.arange(1, 9):
        if state_cyclic[i] == 0 and state_cyclic[i-1] != 0 and state_cyclic[i+1] != 0:
            value2 = state_cyclic[i-1] + state_cyclic[i+1]
            if value2 > value:
                value = value2
    return value

def get_optimal_statics(n_rounds):
    values = np.array([get_optimal_value(make_state()) for i in np.arange(n_rounds)])
    return np.mean(values), np.std(values)

def get_baseline_value(n_rounds):
    values = [run(make_state(), np.random.choice(8)) for i in np.arange(n_rounds)]
    return np.mean(values)
    

def get_optimal_action(state):
    action = state.argmax()
    value = state[action]
    state_cyclic = np.concatenate((state, state[0:2]))
    for i in np.arange(1, 9):
        if state_cyclic[i] == 0 and state_cyclic[i-1] != 0 and state_cyclic[i+1] != 0:
            value2 = state_cyclic[i-1] + state_cyclic[i+1]
            if value2 > value:
                value = value2
                action = i % 8
    return action

def get_optimal_actions(state_list):
    return np.array([get_optimal_action(state) for state in state_list])
    

def get_model_actions(model, state_list):
    return model(np.array(state_list)).numpy().argmax(axis = 1)

def test_model(model, n_rounds):
    state_list = np.array([make_state() for i in np.arange(n_rounds)])
    actions = get_model_actions(model, state_list)
    values = [run(state, action) for state, action in zip(state_list, actions)]
    return np.mean(values)

def one_batch(model, batch_size, n_test_rounds=0):
    y_target_list = []
    state_list = np.array([make_state() for i in np.arange(batch_size)])
    prob_list = model(np.array(state_list)).numpy()
    actions = get_model_actions(model, state_list)
    for i, prob in enumerate(prob_list):
        y_target = np.zeros(8)
        state = state_list[i]
        action = np.random.choice(8, p=prob)
        y_target[action] = run(state, action)
        y_target_list.append(y_target)
    model.fit(state_list, np.array(y_target_list), verbose = 0)
    if n_test_rounds > 0:
        return (test_model(model, n_test_rounds))

def test_model_accuracy(model, n_tests):
    count = 0
    state_list = [make_state() for i in np.arange(n_tests)]
    optimal_values = np.array([get_optimal_value(state) for state in state_list])
    model_actions = get_model_actions(model, state_list)
    model_values = np.array([run(state, action) for state, action in zip(state_list, model_actions)])
    return np.sum(np.abs(optimal_values - model_values) < 1e-6)/n_tests

def one_batch_supervised(model, batch_size, n_test_rounds=0):
    state_list = []
    y_target_list = []
    # rewards = 0
    for i in np.arange(batch_size):
        state = make_state()
        action = get_optimal_action(state)
        y_target = np.zeros(8)
        y_target[action] = 1
        state_list.append(state)
        y_target_list.append(y_target)
    model.fit(np.array(state_list), np.array(y_target_list), verbose=0)
    if n_test_rounds > 0:
        return (test_model(model, n_test_rounds))

def train(model, max_batch=200, batch_size=128, n_test_rounds=10000, verbose = 0):
    best_weights = []
    best_idx = 0
    best_score = 0
    for i in np.arange(max_batch):
        score = one_batch(model, batch_size, n_test_rounds)
        if best_score < score:
            best_score = score
            best_idx = i
            best_weights = model.get_weights()
        if verbose == 1:
            print(i, score)
    return best_idx, best_score, best_weights

def train_supervised(model, max_batch=200, batch_size=128, n_test_rounds=10000, verbose = 0):
    best_weights = []
    best_idx = 0
    best_score = 0
    for i in np.arange(max_batch):
        score = one_batch_supervised(model, batch_size, n_test_rounds)
        if best_score < score:
            best_score = score
            best_idx = i
            best_weights = model.get_weights()
        if verbose == 1:
            print(i, score)
    return best_idx, best_score, best_weights


def create_model(n_hidden_layers, n_dense_units, ratio_dropout, optimizer):
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
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy')
    return model
