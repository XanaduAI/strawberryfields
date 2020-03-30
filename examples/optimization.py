#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

raise NotImplementedError('This example is unavailable until SF supports TensorFlow 2.0.')


# initialize engine and program objects
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
circuit = sf.Program(1)

tf_alpha = tf.Variable(0.1)
tf_phi = tf.Variable(0.1)

alpha, phi = circuit.params('alpha', 'phi')
with circuit.context as q:
    Dgate(alpha, phi) | q[0]
results = eng.run(circuit, args={'alpha': tf_alpha, 'phi': tf_phi}, eval=False})

# loss is probability for the coherent state |alpha=1>
prob = results.state.fidelity_coherent([1.0])
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
