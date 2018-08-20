import numpy as np
import tensorflow as tf
from ptbloader import ptb, Vocabulary

E_SIZE = 100
BATCH_SIZE = 100
NUM_SAMPLED = 100
EPOCHS = 20
BATCHES = 200

train_text, test_text, valid_text = ptb(
        '/home/ahab/dataset/simple-examples.tgz'
)
v = Vocabulary(train_text)
s = v.skipgram_trainset(train_text, 5, BATCH_SIZE)

embeddings = tf.Variable(tf.random_uniform([len(v.v), E_SIZE], -1., 1.))

nce_weights = tf.Variable(
        tf.truncated_normal([len(v.v), E_SIZE], stddev=1. / np.sqrt(E_SIZE))
)
nce_biases = tf.Variable(tf.zeros([len(v.v)]))

train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss = tf.reduce_mean(
        tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=NUM_SAMPLED,
                num_classes=len(v.v)
        )
)

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(EPOCHS):
    h = []
    for b in range(BATCHES):
        batch_inputs, batch_labels = next(s)
        _, cur_loss = sess.run([optimizer, loss], {
                train_inputs: batch_inputs,
                train_labels: batch_labels
        })
        h.append(cur_loss)
    print('epoch #{0} loss: {1}'.format(e+1, np.mean(h)))

