# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        # raise NotImplementedError
        return tf.pow(vector, 3)
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        elif activation_name == "relu":
            self._activation = tf.keras.activations.relu
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.trainable_embeddings = trainable_embeddings


        self.embeddings = tf.Variable(tf.random.truncated_normal(shape = [self.vocab_size, self.embedding_dim], stddev = 0.1))
        self.w1_weights = tf.Variable(tf.random.truncated_normal(shape = [self.num_tokens*self.embedding_dim, self.hidden_dim], stddev = 0.1))
        self.biases = tf.Variable(tf.zeros(shape = [1, self.hidden_dim]))
        self.w2_weights = tf.Variable(tf.random.truncated_normal(shape = [self.hidden_dim, self.num_transitions], stddev = 0.0001))

        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        embeddings = tf.reshape(embeddings, shape = [embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]])
        
        wx = tf.matmul(embeddings, self.w1_weights)
        wx_b = tf.add(wx, self.biases)
        w_activation = self._activation(wx_b)
        logits = tf.matmul(w_activation, self.w2_weights)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        mask = tf.cast(tf.ones(shape = labels.shape), dtype = "float32")
        lbls = (labels >= -1)
        masked_labels = lbls*mask
        t1 = tf.nn.softmax(logits*masked_labels)

        mask_greedy = tf.cast((labels == 1), dtype = "float32")
        masked_greedy_labels = labels*mask_greedy
        loss = -1*tf.reduce_mean(tf.math.log(tf.reduce_sum(t1*masked_greedy_labels,axis=1)))

        if self.trainable_embeddings == True:
            print("**********************************")
            print("Tunable embeddings is turned on")

            l1 = tf.nn.l2_loss(self.w1_weights)
            l2 = tf.nn.l2_loss(self.w2_weights)
            l3 = tf.nn.l2_loss(self.biases)
            l4 = tf.nn.l2_loss(self.embeddings)


            regularization = self._regularization_lambda*(l1 + l2 + l3 + l4)
        else:
            print("**********************************")
            print("Tunable embeddings is turned off")
            l1 = tf.nn.l2_loss(self.w1_weights)
            l2 = tf.nn.l2_loss(self.w2_weights)
            l3 = tf.nn.l2_loss(self.biases)

            regularization = self._regularization_lambda*(l1 + l2 + l3)



        # TODO(Students) End
        return loss + regularization
