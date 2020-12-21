import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel


DEFAULT_OBJECT_PAIR_LAYER_SIZES = [32, 32, 32]
DEFAULT_COMBINED_OBJECT_LAYER_SIZES = [32, 32]


class PrediNetModel(BaseObjectModel):
    def __init__(self, dataset,
                 key_size, num_heads, num_relations, output_hidden_size,
                 output_hidden_activation_class=nn.ReLU, output_activation_class=None,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_log_prefix=None, validation_log_prefix=None, test_log_prefix=None):
        super(PrediNetModel, self).__init__(dataset, loss=loss,
                                               optimizer_class=optimizer_class,
                                               lr=lr, batch_size=batch_size, train_log_prefix=train_log_prefix,
                                               validation_log_prefix=validation_log_prefix,
                                               test_log_prefix=test_log_prefix)

        self.position_slice = self.dataset.object_generator.get_position_slice()
        self.key_size = key_size
        self.key_layer = nn.Linear(self.object_size, self.key_size, bias=False)

        self.num_heads = num_heads
        self.query_1_layer = nn.Linear(self.num_objects * self.object_size, self.num_heads * self.key_size, bias=False)
        self.query_2_layer = nn.Linear(self.num_objects * self.object_size, self.num_heads * self.key_size, bias=False)

        self.num_relations = num_relations
        self.relation_embedding_layer = nn.Linear(self.object_size, self.num_relations, bias=False)

        self.output_hidden_size = output_hidden_size
        self.output_hidden_activation_class = output_hidden_activation_class
        self.output_hidden_activation = self.output_hidden_activation_class()
        self.output_hidden_layer = nn.Linear(self.num_heads * (self.num_relations + 4), self.output_hidden_size)
        self.output_layer = nn.Linear(self.output_hidden_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        return x

    def predict(self, x):
        B, N, F = x.shape
        K = self.key_size
        H = self.num_heads

        x_flat = x.view(B, -1)

        keys = self.key_layer(x)  # B x N x K
        keys_tiled = keys.reshape(B, 1, N, K).repeat(1, H, 1, 1)  # B x H x N x K

        queries_1 = self.query_1_layer(x_flat).reshape(B, H, 1, K)  # B x (H * K) => B x H x 1 x K
        queries_2 = self.query_2_layer(x_flat).reshape(B, H, 1, K)  # B x (H * K) => B x H x 1 x K

        keys_transposed = keys_tiled.transpose(2, 3)  # B x H x K x N
        attention_1 = (queries_1 @ keys_transposed).softmax(-1)  # B x H x 1 x N
        attention_2 = (queries_2 @ keys_transposed).softmax(-1)  # B x H x 1 x N

        x_tiled = x.reshape(B, 1, N, F).repeat(1, H, 1, 1)  # B x H x N x F
        attention_weighted_features_1 = (attention_1 @ x_tiled).squeeze(2)  # B x H x F
        attention_weighted_features_2 = (attention_2 @ x_tiled).squeeze(2)  # B x H x F

        relation_embeddings_1 = self.relation_embedding_layer(attention_weighted_features_1)  # B x H x R
        relation_embeddings_2 = self.relation_embedding_layer(attention_weighted_features_2)  # B x H x R

        relations = relation_embeddings_1 - relation_embeddings_2  # B x H x R
        attention_weighted_positions_1 = attention_weighted_features_1[:, :, self.position_slice]  # B x H x 2
        attention_weighted_positions_2 = attention_weighted_features_2[:, :, self.position_slice]  # B x H x 2
        relations_with_positions = torch.cat((relations, attention_weighted_positions_1,
                                              attention_weighted_positions_2), -1)  # B x H x (R + 4)

        output_hidden = self.output_hidden_activation(self.output_hidden_layer(relations_with_positions.view(B, -1)))
        return self.output_activation(self.output_layer(output_hidden))



