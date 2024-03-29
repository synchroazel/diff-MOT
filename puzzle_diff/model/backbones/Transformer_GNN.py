from torch import nn
from torch_geometric.nn import TransformerConv


class Transformer_GNN(nn.Module):
    def __init__(self, input_size, hidden_dim, heads, output_size) -> None:
        super().__init__()

        self.output_size = output_size

        self.module_list = nn.ModuleList(
            [
                TransformerConv(
                    input_size, out_channels=hidden_dim // heads, heads=heads
                ),
                TransformerConv(
                    hidden_dim, out_channels=hidden_dim // heads, heads=heads
                ),
                TransformerConv(
                    hidden_dim, out_channels=hidden_dim // heads, heads=heads
                ),
                TransformerConv(
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    out_channels=output_size // heads,
                ),
            ]
        )

    def forward(self, x, edge_index, *args):
        x = self.module_list[0](x=x, edge_index=edge_index)
        x = nn.functional.gelu(x)
        x = self.module_list[1](x=x, edge_index=edge_index)

        x = nn.functional.gelu(x)
        x = self.module_list[2](x=x, edge_index=edge_index)

        x = nn.functional.gelu(x)
        x = self.module_list[3](x=x, edge_index=edge_index)

        x = nn.Linear(x.shape[1], self.output_size)(x)

        return x
