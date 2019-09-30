import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB, SpGAT
from tensorflow.keras.layers import Dense, Dropout

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpKBGATModified(tf.keras.Model):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[1]

        self.drop_GAT = drop_GAT

        # Pretrained enity embedding and relation embeddings
        self.entity_embeddings = tf.Variable(initial_value=initial_entity_emb,
                                             trainable=True)

        self.relation_embeddings = tf.Variable(initial_value=initial_relation_emb,
                                               trainable=True)

        self.sparse_gat_1 = SpGAT(self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.nheads_GAT_1, self.drop_GAT, True, self.relation_out_dim_1)

        self.sparse_gat_2 = SpGAT(self.entity_out_dim_1 * 2, self.entity_out_dim_2, self.relation_out_dim_1,
                                  self.nheads_GAT_1, self.drop_GAT, concat_output=False)

        self.skip_connection = Dense(self.entity_out_dim_1 * self.nheads_GAT_1, kernel_initializer='glorot_normal')

        self.dropout = Dropout(drop_GAT)


    def call(self, inputs):
        # Unpack inputs
        adj, batch_inputs, train_indices_nhop = inputs
        
        # getting edge list
        edges = adj[0]
        edge_types = adj[1]

        nhop_edges = tf.concat([tf.expand_dims(train_indices_nhop[:, 3], -1),
                               tf.expand_dims(train_indices_nhop[:, 0], -1)], 1)
        nhop_edges = tf.transpose(nhop_edges)

        nhop_edge_types = tf.concat([tf.expand_dims(train_indices_nhop[:, 1], -1),
                               tf.expand_dims(train_indices_nhop[:, 2], -1)], 1)

        edge_embed = tf.nn.embedding_lookup(self.relation_embeddings, edge_types)

        # normalize the enity embeddings
        self.entity_embeddings = tf.math.l2_normalize(self.entity_embeddings, axis=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1([self.entity_embeddings,
                                                          self.relation_embeddings,
                                                          edges, edge_types,
                                                          nhop_edges, nhop_edge_types])

        out_entity_1 = self.dropout(out_entity_1)

        out_entity_2, out_relation_2 = self.sparse_gat_2([out_entity_1,
                                                          out_relation_1,
                                                          edges, edge_types,
                                                          nhop_edges, nhop_edge_types])

        skip_entity = self.skip_connection(self.entity_embeddings)

        mask_indices, _ = tf.unique(batch_inputs[:, 2])
        mask_indices = tf.expand_dims(mask_indices, -1)
        update = tf.squeeze(tf.ones_like(mask_indices, dtype = tf.float32))
        shape = tf.constant([self.entity_embeddings.shape[0]])
        mask = tf.scatter_nd(mask_indices, update, shape)

        mask = tf.expand_dims(mask, -1)

        out_entity= skip_entity + mask * out_entity_2

        out_entity = tf.math.l2_normalize(out_entity, axis=1)

        self.final_entity_embeddings = out_entity
        self.final_relation_embeddings = out_relation_2

        return out_entity, out_relation_2


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
