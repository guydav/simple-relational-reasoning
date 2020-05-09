import torch
from torch import nn
import torch.nn.functional as F
from simple_relational_reasoning.models.base import BaseObjectModel, ObjectCombinationMethod


TRANSFORMER_MLP_SIZES = [16, 8]
DEFAULT_MLP_SIZES = [32, 32]


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.q_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.k_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, q, k, v):
        B, N, E = q.shape

        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        # reshape to split between heads -- if I understand correctly, -1 = N
        q = q.view(B, -1, self.num_heads, self.head_size)
        k = k.view(B, -1, self.num_heads, self.head_size)
        v = v.view(B, -1, self.num_heads, self.head_size)

        # transpose to B, H, N, E/H -- that is, switch the head and N_objects dimensions
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot-product attention
        attention_scaled_v = self._scaled_dot_product_attention(q, k, v)

        # transpose back to B, N, H, E/H
        attention_scaled_v = attention_scaled_v.transpose(1, 2)

        # reshape back to B, N, E
        joined_heads_v = attention_scaled_v.contiguous().view(B, -1, E)

        # pass through the final linear layer
        return self.out_layer(joined_heads_v)

    def _scaled_dot_product_attention(self, q, k, v):
        B, H, N, E_per_H = q.shape

        # transpose to multiply across the E_per_H dimension -- output shape is B, H, N, N
        scores = torch.matmul(q, k.transpose(-2, -1)) / (E_per_H ** 0.5)
        # TODO: mask?
        # same shape, but now the last dimension adds up to 1
        scores = F.softmax(scores, dim=-1)
        # TODO: dropout?
        # same shape, as we effectively multiply (N x N) x (N x E_per_H), projecting across B, H
        return torch.matmul(scores, v)


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads=1, per_object_mlp_sizes=None,
                 mlp_activation_class=nn.ReLU, activation_mlp_output=False):
        super(TransformerEncoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_size, num_heads)

        if per_object_mlp_sizes is not None:
            mlp_input_size = embedding_size
            mlp_layers = []

            for size in per_object_mlp_sizes:
                mlp_layers.append(nn.Linear(mlp_input_size, size))
                # TODO: dropout?
                mlp_layers.append(mlp_activation_class())
                mlp_input_size = size

            if not activation_mlp_output:
                mlp_layers = mlp_layers[:-1]

            self.mlp_module = nn.Sequential(*mlp_layers)

        else:
            self.mlp_module = nn.Identity()

    def forward(self, x):
        # TODO: normalization?
        # TODO: do we want the residual connections?
        x = x + self.multi_head_attention(x, x, x)
        x = x + self.mlp_module(x)
        return x


class TransformerModel(BaseObjectModel):
    def __init__(self, object_generator,
                 embedding_size=None, embedding_activation_class=nn.ReLU,
                 num_transformer_layers=1, num_heads=1,
                 transformer_mlp_sizes=TRANSFORMER_MLP_SIZES, transformer_mlp_activation_class=nn.ReLU,
                 transformer_mlp_activation_output=False, object_combiner=ObjectCombinationMethod.MEAN,
                 mlp_sizes=DEFAULT_MLP_SIZES, mlp_activation_class=nn.ReLU,
                 output_size=2, output_activation_class=None,
                 loss=F.cross_entropy, optimizer_class=torch.optim.Adam, lr=1e-4,
                 batch_size=32, train_epoch_size=1024, validation_epoch_size=128, regenerate_every_epoch=False):

        super(TransformerModel, self).__init__(object_generator, loss=loss, optimizer_class=optimizer_class,
                                               lr=lr, batch_size=batch_size, train_epoch_size=train_epoch_size,
                                               validation_epoch_size=validation_epoch_size,
                                               regenerate_every_epoch=regenerate_every_epoch)

        self.embedding_module = nn.Identity()
        self.embedding_size = self.object_size

        if embedding_size is not None:
            self.embedding_size = embedding_size
            self.embedding_module = nn.Sequential(
                nn.Linear(self.object_size, self.embedding_size),
                embedding_activation_class()
            )

        if num_transformer_layers > 1:
            if transformer_mlp_sizes is not None and transformer_mlp_sizes[-1] != embedding_size:
                raise ValueError(f'With more than one transformer block, expecting the last MLP output {transformer_mlp_sizes} to be the same as the embedding size {embedding_size}...')

        # Transformer setup
        self.transformer_module = nn.Sequential(
            *[TransformerEncoder(self.embedding_size, num_heads, transformer_mlp_sizes,
                                 transformer_mlp_activation_class, transformer_mlp_activation_output)
                for _ in range(num_transformer_layers)]
        )

        self.object_combiner = object_combiner

        mlp_input_size = (transformer_mlp_sizes is not None) and transformer_mlp_sizes[-1] or self.embedding_size
        if self.object_combiner == ObjectCombinationMethod.CONCAT:
            mlp_input_size *= self.num_objects

        self.mlp_module = nn.Identity()
        if mlp_sizes is not None and len(mlp_sizes) > 0:
            mlp_layers = []
            for size in mlp_sizes:
                mlp_layers.append(nn.Linear(mlp_input_size, size))
                mlp_layers.append(mlp_activation_class())
                mlp_input_size = size

            self.mlp_module = nn.Sequential(*mlp_layers)

        # Output/prediction layer from the transformer output
        self.output_size = output_size
        self.output_layer = nn.Linear(mlp_input_size, self.output_size)

        if output_activation_class is None:
            output_activation_class = nn.Identity
        self.output_activation = output_activation_class()

    def embed(self, x):
        x = self.pre_embedding_module(x)
        return self.transformer_module(x)

    def predict(self, x):
        x = self.object_combiner.combine(x)
        x = self.mlp_module(x)
        return self.output_activation(self.output_layer(x))

