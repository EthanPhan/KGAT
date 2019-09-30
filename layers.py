import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
import tensorflow as tf
from tensorflow.keras import layers


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpGraphAttentionLayer(layers.Layer):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, nrela_dim, drop_rate):
        super(SpGraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nrela_dim = nrela_dim

        self.dropout = layers.Dropout(drop_rate)

        self.w1 = self.add_weight(shape=(out_features, 2 * in_features + nrela_dim),
                                 initializer='glorot_normal',
                                 trainable=True)

        self.w2 = self.add_weight(shape=(1, out_features),
                                 initializer='glorot_normal',
                                 trainable=True)


    def call(self, inputs):
        # unpack inputs
        x, edges, edge_embed, nhop_edges, nhop_edge_embed = inputs

        N = x.shape[0]

        # Self-attention on the nodes - Shared attention mechanism
        # all edges including nhop edges
        all_edges = tf.concat([edges[:, :], nhop_edges[:, :]], 1)

        # embeddings of all edges
        all_edge_embeds = tf.concat(
            [edge_embed[:, :], nhop_edge_embed[:, :]], 0)

        # Eq. 5 in the paper
        h_i = tf.nn.embedding_lookup(x,all_edges[0, :])
        h_j = tf.nn.embedding_lookup(x,all_edges[1, :])
        g_k = all_edge_embeds[:, :]

        triple_embed = tf.transpose(tf.concat([h_i, h_j, g_k], 1))

        c_ijk = tf.matmul(self.w1, triple_embed)

        # Eq. 6
        b_ijk = tf.nn.leaky_relu(tf.matmul(self.w2, c_ijk))

        b_exp = tf.exp(b_ijk)

        indices=tf.transpose(tf.cast(all_edges, tf.int64))
        values=tf.squeeze(b_exp)
        sparse_b = tf.sparse.SparseTensor(indices=indices, values=values,
                                          dense_shape=(N, N))

        # The denominator of the Eq. 7
        b_rowsum = tf.sparse.reduce_sum(sparse_b, axis=1)
        assert not tf.reduce_any(tf.math.is_nan(b_rowsum))
        

        # half of the Eq. 8; c_ijk * the numerator of the Eq. 7
        b_exp = self.dropout(b_exp)
        weighted_c_ijk = tf.transpose(b_exp * c_ijk)

        l = weighted_c_ijk.shape[1]
        weighted_rowsum = []
        for i in range(l):
            value = weighted_c_ijk[:, i]
            temp = tf.sparse.SparseTensor(indices=indices, values=value, dense_shape=(N, N))
            row_sum = tf.sparse.reduce_sum(temp, axis=1)
            row_sum = tf.reshape(row_sum, shape= (N, 1))
            
            weighted_rowsum.append(row_sum)
            
        weighted_rowsum = tf.concat(weighted_rowsum, axis=-1)
        
        assert not tf.reduce_any(tf.math.is_nan(weighted_rowsum))

        # another half of the Eq. 8;
        b_rowsum = tf.expand_dims(b_rowsum, -1)
        h_prime = weighted_rowsum / (b_rowsum + 1e-12)
        assert not tf.reduce_any(tf.math.is_nan(h_prime))

        return h_prime


class SpGAT(layers.Layer):
    def __init__(self, nfeat, nhid, relation_dim, nheads, drop_rate, concat_output=True, out_relation_dim=None):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.concat_output = concat_output
        self.out_relation_dim = out_relation_dim

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 drop_rate)
                               for _ in range(nheads)]
        
        # W matrix to alter relation embedding's dimension
        if self.out_relation_dim:
            self.w = self.add_weight(shape=(relation_dim, out_relation_dim),
                                 initializer='glorot_normal',
                                 trainable=True)

    def call(self, inputs):
        # unpack all inputs
        x, relation_embeddings, edges, edge_types, nhop_edges, nhop_edge_types = inputs

        # list of embedding vectors for the edges
        edge_embed = tf.nn.embedding_lookup(relation_embeddings, edge_types)

        # list of embedding vectors for the nhop edges
        nhop_edge_embed = tf.nn.embedding_lookup(relation_embeddings,
            nhop_edge_types[:, 0]) + tf.nn.embedding_lookup(relation_embeddings,nhop_edge_types[:, 1])

        if self.concat_output:
            x = tf.concat([tf.nn.elu(att([x, edges, edge_embed, nhop_edges, nhop_edge_embed]))
                           for att in self.attentions], 1)

        else:
            x = tf.keras.layers.average([att([x, edges, edge_embed, nhop_edges, nhop_edge_embed])
                           for att in self.attentions])
            x = tf.nn.elu(x)

        if self.out_relation_dim:
            relation_embeddings = tf.matmul(relation_embeddings, self.w)

        return x, relation_embeddings
