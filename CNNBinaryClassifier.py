import tensorflow as tf
from functools import reduce
from numpy import unique, array, vectorize
from sklearn.metrics import accuracy_score, f1_score

class CNNBinaryClassifier:

    def __init__(self, train_data=None):
        data, labels = train_data

        labels = self._transform_labels(labels)
        data = self._reshape_input(data)
        
        self.train_data = (data, labels)

        self.assemble_graph()

        self._open_session()

        if train_data:
            self.train()     

    def assemble_graph(self, learning_rate = 0.02):
        data, labels = self.train_data

        input_data = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="Input")
        output_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Output")

        conv1 = tf.layers.conv2d(
            inputs=input_data,
            filters=3,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=3,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 3])
        dense = tf.layers.dense(inputs=pool2_flat, units=4, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=1)

        output = tf.nn.sigmoid(logits, name="softmax_tensor")

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_label, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        self.graph_terminals = {
                'out': output,
                'loss':loss,
                'train':train,
                'input_data':input_data,
                'output_label': output_label}   

    def train(self, epochs=10, minibatch_size=256):
        
        minibatches = self._create_minibatches(minibatch_size)

        train = self.graph_terminals['train']
        input_data = self.graph_terminals['input_data']
        output_label = self.graph_terminals['output_label']
        loss = self.graph_terminals['loss']
        out = self.graph_terminals['out']
        sess = self.sess

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for data, labels in minibatches:
                sess.run(train, {input_data: data, output_label: labels})
                # print(labels[:10])
                # print(self.predict(data)[:10])
            
            loss_val = sess.run(loss, {input_data: minibatches[0][0], output_label: minibatches[0][1]})
            print("Epoch %d, loss: %.2f" % (e, loss_val))

    def predict(self, data):

        data = self._reshape_input(data)
        output = self.graph_terminals['out']
        input_data = self.graph_terminals['input_data']
        
        predict = self.sess.run(output, {input_data: data})

        # print(predict[:10])

        reverse_map = vectorize(lambda x: self.inverse_label_mapping[1 if x > .5 else 0])

        return reverse_map(predict)

    def _create_minibatches(self, minibatch_size):
        pos = 0

        data, labels = self.train_data
        n_samples = len(labels)

        batches = []
        while pos + minibatch_size < n_samples:
            batches.append((data[pos:pos+minibatch_size,:], labels[pos:pos+minibatch_size]))
            pos += minibatch_size

        if pos < n_samples:
            batches.append((data[pos:n_samples,:], labels[pos:n_samples,:]))

        return batches

    def _transform_labels(self, labels):
        labels = labels.reshape((-1, 1)) if len(labels.shape) == 1 else labels
        u_labels = unique(labels)
        
        if len(u_labels) != 2:
            raise ValueError("This CNN implementation can handle only binary classification, but more than 2 unique labels provided")
        
        label_mapping = dict(zip(u_labels[:2],[0,1]))
        self.inverse_label_mapping = dict(zip(label_mapping.values(), label_mapping.keys()))

        map_labels = vectorize(lambda x:label_mapping[x])
        return map_labels(labels)

    def _reshape_input(self, data):
        if len(data.shape) == 3:
            data = data.reshape((-1, 28, 28, 1))
        return data

    def _open_session(self):
        self.sess = tf.Session()





if __name__ == "__main__":



    def mnist_to_binary(train_data, train_label, test_data, test_label):

        binarized_labels = []
        for labels in [train_label, test_label]:
            remainder_2 = vectorize(lambda x: x%2)
            # remainder_2 = vectorize(lambda x: 1 if x==5 else 0)
            binarized_labels.append(remainder_2(labels))

        train_label, test_label = binarized_labels

        return train_data, train_label, test_data, test_label




    ((train_data, train_labels),
        (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data, train_labels, test_data, test_labels = mnist_to_binary(train_data, train_labels, eval_data, eval_labels)

    cnn = CNNBinaryClassifier((train_data, train_labels))
    print("Testing score f1: {}".format(f1_score(test_labels, cnn.predict(test_data))))
    # svm.train()

