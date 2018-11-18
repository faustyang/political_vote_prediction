from layers import *
from metrics import *
import math
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.votelayers=[]
        self.activations = []
        self.voteactivation=[]
        self.inputs = None
        self.outputs = None
        self.final_layers=[]
        self.final_activation=[]
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.mlse=0
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        #Add predict model


        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.opt_ops = self.optimizer.minimize(self.mlse)
    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim,word_number ,**kwargs):
        super(GCN, self).__init__(**kwargs)

        #self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.word_number=word_number
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.member_dim = placeholders['labels'].get_shape().as_list()[0]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.vote_rate=placeholders['vote_rate']
        self.vote_dim=placeholders["vote_rate"].get_shape().as_list()[1]
        self.description=placeholders["description"]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.context_management()
        self.feature()
        self.inputs=self.features
        placeholders["feature"]=self.inputs
        self.placeholders = placeholders
        self.build()
    def context_management(self):
        with tf.device("/cpu:0"):
            words_embeddings=tf.Variable(
                tf.random_uniform([self.word_number+1,self.input_dim//2],-1.0,1.0),
                name="words_embeddings"
            )
            embed=tf.nn.embedding_lookup(words_embeddings,self.description)
            vote_vector=RNNmodel(embed)
            self.vote_vector=vote_vector
    def feature(self):
        with tf.device("/cpu:0"):
            member_embeddings=tf.Variable(
                tf.random_uniform([self.member_dim,self.input_dim//2],-1.0,1.0),
                name="member_embeddings",dtype="float32"
            )

            vote_list=tf.ones([self.member_dim,1])


            vote_vector=tf.matmul(vote_list,self.vote_vector)

            feature=tf.concat([vote_vector,member_embeddings],1)
            self.features=feature
    def _predict_vote(self):

        self.votelayers.append(Dense(input_dim=FLAGS.hidden_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.votelayers.append(Dense(input_dim=FLAGS.hidden1,
                                            output_dim=self.vote_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
        self.final_layers.append(Dense(input_dim=6,
                                       output_dim=3,
                                       placeholders=self.placeholders,
                                       act=tf.nn.relu,
                                       dropout=False,
                                       sparse_inputs=False,
                                       logging=self.logging


        ))

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        #self.rate=tf.Variable()
        self.loss +=0.1* mse(self.voteoutputs,self.vote_rate)
        self.mlse=mse(self.voteoutputs,self.vote_rate)
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
            self._predict_vote()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.output = self.activations[-1]
        # Add predict model
        self.voteactivation.append(self.vote_vector)
        for layer in self.votelayers:
            hidden = layer(self.voteactivation[-1])
            self.voteactivation.append(hidden)
        self.voteoutputs = tf.nn.softmax(self.voteactivation[-1])
        vote_list = tf.ones([self.member_dim, 1])


        vote_predict = tf.matmul(vote_list, self.voteoutputs)
        self.final=tf.concat([vote_predict,self.output ],1)
        self.final_activation.append(self.final)
        for layer in self.final_layers:
            hidden=layer(self.final_activation[-1])
            self.final_activation.append(hidden)
        self.outputs=self.final_activation[-1]
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.opt_ops = self.optimizer.minimize(self.mlse)
    def predict(self):
        return tf.nn.softmax(self.outputs)
