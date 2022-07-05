import typing

import torch, torch.nn.functional
import torch_geometric as pyg

from stochpool.layers.stoch_pool import StochPool


class GraphPooledConvolutionalNetwork(torch.nn.Module):
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super(GraphPooledConvolutionalNetwork.ResidualBlock, self).__init__()
            self.conv = pyg.nn.GCNConv(in_channels, out_channels)
            self.bn = pyg.nn.BatchNorm(out_channels)
            self.activation = torch.nn.SiLU()

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv(x, edge_index)
            x = self.bn(x)
            x = self.activation(x)
            return x

    def __init__(self, in_channels: int, conv_channels: typing.Tuple[int], pool_after: int = 2):
        """

        :param conv_channels: List of (input_channels, hidden_channels, output_channels)
        """
        super().__init__()

        self.input_block = GraphPooledConvolutionalNetwork.ResidualBlock(
            in_channels=in_channels, out_channels=conv_channels[0]
        )
        self.res_blocks = torch.nn.ModuleList(
            [
                GraphPooledConvolutionalNetwork.ResidualBlock(
                    in_channels=in_channels, out_channels=out_channels
                )
                for _sub_layer in range(pool_after)
                for in_channels, out_channels in zip(
                    conv_channels[:-1], conv_channels[1:]
                )
            ]
        )
        self.pooling_layers = torch.nn.ModuleList([
            StochPool(torch.nn.Linear(input_channels))
            for input_channels in conv_channels
        ])
        self.pool_after = pool_after
        self.lin1 = torch.nn.Linear(conv_channels[-1], conv_channels[-1])
        self.lin2 = torch.nn.Linear(conv_channels[-1], conv_channels[-1])

    def forward(self, x, edge_index, _batch, batch_ptr):
        self.input_block(x, edge_index)
        for i in range(len(self.pooling_layers)):
            for j in range(self.pool_after):
                x = self.res_blocks[i * self.pool_after](x)
            x = self.pooling_layers[i](x)
        x = self.conv3(x, edge_index).relu()
        x = pyg.nn.global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return (
            torch.nn.functional.log_softmax(x, dim=-1),
            torch.tensor(0.),
        )
