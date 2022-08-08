import sys
import time
import random
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf

# reset any existing graph
tf.reset_default_graph()

# create new graph
graph = tf.Graph()
with graph.as_default():
    items_size = len(int_to_items) + 1

    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, [batch_size, timesteps], name='inputs')
        targets = tf.placeholder(tf.int32, [batch_size, timesteps], name='targets')

    with tf.name_scope("other_placeholders"):
        lr = tf.placeholder(tf.float32, name='learning_rate')
        x_drop_prob = tf.placeholder(tf.float32, name='x_drop_prob')
        rnn_keep_prob = tf.placeholder(tf.float32, name='rnn_keep_prob')
    
    with tf.name_scope("x_dropout"):
        inputs_dropped = tf.layers.dropout(inputs, rate=x_drop_prob)
    
    with tf.name_scope("embedding"):
        embedding = tf.get_variable('embedding_matrix', [items_size, embed_dim])
        rnn_inputs = tf.nn.embedding_lookup(embedding, inputs_dropped)

    with tf.name_scope("cell"):
        cell = tf.contrib.rnn.GRUCell(rnn_size)

    with tf.name_scope("rnn_dropout"):
        cell_dropped = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                     input_keep_prob=1,
                                                     state_keep_prob=rnn_keep_prob,
                                                     output_keep_prob=rnn_keep_prob,
#                                                      variational_recurrent=True,
                                                     input_size=rnn_size,
                                                     dtype=tf.float32)

    with tf.name_scope("rnn_dropout"):
        rnn_layer = tf.contrib.rnn.MultiRNNCell([cell_dropped] * num_layers)

    with tf.name_scope("initial_state"):
        initial_state = rnn_layer.zero_state(batch_size, tf.int32)
        initial_state = tf.identity(initial_state, name='initial_state')

    with tf.name_scope("rnn_output"):
        rnn_output, final_state = tf.nn.dynamic_rnn(rnn_layer, rnn_inputs, dtype=tf.float32)
        final_state =  tf.identity(final_state, name='final_state')

    with tf.name_scope("fully_connected"):
        logits = tf.contrib.layers.fully_connected(rnn_output,
                                                   items_size,
                                                   activation_fn=None,
                                                   biases_initializer=tf.constant_initializer(0.1))
    
    with tf.name_scope("softmax"):
        # y is our prediction
        probs = tf.nn.softmax(logits, name='probs')
        probs = tf.slice(probs, [0, 0, 1], [-1, -1, -1])
        zeros = tf.zeros([batch_size, timesteps, 1], tf.float32)
        probs = tf.concat([zeros, probs], 2)
    
    with tf.name_scope("masking"):
        # top k predictions: Shape = (batch_size, timesteps, k)
        top_preds_values, top_preds = tf.nn.top_k(probs, k=top_k)

        # making targets a 3D matrix and finding the mask values
        targets_ = tf.tile(tf.expand_dims(targets, 2), [1, 1, top_k])
        mask_3d = tf.sign(tf.to_float(targets_))
        mask_2d = tf.sign(tf.to_float(targets))

        equal_pad = tf.equal(tf.sign(tf.to_float(targets)), 0)
        pad_ints = tf.cast(equal_pad, tf.int32)
        pad_count = tf.reduce_sum(pad_ints)

        multiplier = tf.to_float(tf.divide((tf.multiply(batch_size, timesteps)), (tf.multiply(batch_size, timesteps) - pad_count)))

    with tf.name_scope("accuracy_calc"):
        # calculating accuracy with mask
        correct_pred = tf.equal(top_preds, targets_)
        cor_pred = tf.sign(tf.to_float(correct_pred))
        mask_acc = tf.multiply(mask_3d, cor_pred)
    
    with tf.name_scope('accuracy'):
        accuracy = tf.multiply(tf.reduce_mean(tf.cast(mask_acc, tf.float32)), top_k)
        accuracy_ = tf.multiply(accuracy, multiplier)
    
    with tf.name_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=targets)
        masked_losses = tf.multiply(mask_2d, loss)
        cost = tf.reduce_mean(masked_losses)

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    with tf.name_scope("saver"):
        saver = tf.train.Saver()
    
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", cost)
        tf.summary.scalar("accuracy", accuracy_)
        merged = tf.summary.merge_all()