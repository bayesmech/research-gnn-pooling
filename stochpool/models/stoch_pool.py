import typing

import torch, torch.nn.functional
import torch_geometric as pyg

from stochpool.layers.stoch_pool import StochPool
from stochpool.utils.param_scheduler import HyperParameter, StaticHyperParameter


class StochPooledConvolutionalNetwork(torch.nn.Module):
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super(StochPooledConvolutionalNetwork.ResidualBlock, self).__init__()
            self.conv = pyg.nn.GCNConv(in_channels, out_channels)
            self.bn = pyg.nn.BatchNorm(out_channels)
            self.activation = torch.nn.SiLU()

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv(x, edge_index)
            x = self.bn(x)
            x = self.activation(x)
            return x

    class ResidualBlockPoolStack(torch.nn.Module):
        def __init__(
            self,
            input_channel: int,
            channels: int,
            pool_after: int,
            n_clusters: int,
        ):
            super().__init__()

            self.input_block = StochPooledConvolutionalNetwork.ResidualBlock(
                in_channels=input_channel, out_channels=channels
            )

            self.res_blocks = torch.nn.ModuleList(
                StochPooledConvolutionalNetwork.ResidualBlock(
                    in_channels=channels, out_channels=channels
                )
                for _ in range(pool_after - 1)
            )
            self.pool = StochPool(
                pyg.nn.GCNConv(in_channels=channels, out_channels=n_clusters)
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            batch_ptr: torch.Tensor,
            edge_weight: typing.Optional[torch.Tensor],
            tau: HyperParameter,
        ):
            x = self.input_block(x, edge_index)

            for i in range(len(self.res_blocks)):
                x = x + self.res_blocks[i](x, edge_index)

            return self.pool(x, edge_index, batch, batch_ptr=batch_ptr, edge_weight=edge_weight, tau=tau)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_channels: typing.Tuple[int, ...],
        n_clusters: typing.Tuple[int, ...],
        pool_after: int = 2,
        tau: HyperParameter = StaticHyperParameter("tau", 1.0),
    ):
        super().__init__()
        self.tau = tau

        self.res_blocks = torch.nn.ModuleList()

        for it in range(len(conv_channels)):
            self.res_blocks.append(
                StochPooledConvolutionalNetwork.ResidualBlockPoolStack(
                    input_channel=in_channels if it == 0 else conv_channels[it - 1],
                    channels=conv_channels[it],
                    pool_after=pool_after,
                    n_clusters=n_clusters[it],
                )
            )

        self.lin = torch.nn.Linear(conv_channels[-1], out_channels)

    def forward(self, x, edge_index, batch, batch_ptr):
        total_entropy_loss = 0
        total_link_loss = 0
        edge_weight = None

        for i in range(len(self.res_blocks)):
            (
                x,
                edge_index,
                edge_weight,
                link_loss,
                entropy_loss,
                batch,
                batch_ptr,
            ) = self.res_blocks[i](x, edge_index, batch, batch_ptr, edge_weight, tau=self.tau)

            total_entropy_loss = total_entropy_loss + entropy_loss
            total_link_loss = total_link_loss + link_loss

        x = pyg.nn.global_mean_pool(x, batch)

        x = self.lin(x)

        return (
            torch.nn.functional.log_softmax(x, dim=1),
            0 * total_entropy_loss + 0 * total_link_loss,
        )
