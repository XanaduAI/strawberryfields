#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

eng, q = sf.Engine(1)

alpha = tf.Variable(0.1)
with eng:
    Dgate(alpha) | q[0]
state = eng.run('tf', cutoff_dim=7, eval=False)

# loss is probability for the Fock state n=1
prob = state.fock_prob([1])
loss = -prob  # negative sign to maximize prob

# Set up optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

# Create Tensorflow Session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Carry out optimization
for step in range(50):
    prob_val, _ = sess.run([prob, minimize_op])
    print("Value at step {}: {}".format(step, prob_val))
