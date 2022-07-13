import typing

import torch_geometric as pyg
import torch, torch.nn.functional


class DiffPooledConvolutionalNetwork(torch.nn.Module):
    class DiffPool(torch.nn.Module):
        def __init__(self, channels: int, n_clusters: int):
            super(DiffPooledConvolutionalNetwork.DiffPool, self).__init__()
            self.pooling_nn = pyg.nn.DenseGCNConv(
                in_channels=channels, out_channels=n_clusters
            )

        def forward(self, x: torch.Tensor, adj: torch.Tensor):
            s = self.pooling_nn(x, adj)
            out_x, out_adj, link_loss, entropy_loss = pyg.nn.dense_diff_pool(x, adj, s)
            return out_x, out_adj, link_loss, entropy_loss

    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super(DiffPooledConvolutionalNetwork.ResidualBlock, self).__init__()
            self.conv = pyg.nn.DenseGCNConv(in_channels, out_channels)
            self.bn = torch.nn.BatchNorm1d(out_channels)
            self.activation = torch.nn.SiLU()

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            x = self.conv(x, adj)
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
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

            self.input_block = DiffPooledConvolutionalNetwork.ResidualBlock(
                in_channels=input_channel, out_channels=channels
            )

            self.res_blocks = torch.nn.ModuleList(
                DiffPooledConvolutionalNetwork.ResidualBlock(
                    in_channels=channels, out_channels=channels
                )
                for _ in range(pool_after - 1)
            )
            self.pool = DiffPooledConvolutionalNetwork.DiffPool(
                channels=channels,
                n_clusters=n_clusters,
            )

        def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor,
        ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self.input_block(x, adj)

            for i in range(len(self.res_blocks)):
                x = x + self.res_blocks[i](x, adj)

            out_x, out_adj, link_loss, entropy_loss = self.pool(x, adj)
            return out_x, out_adj, link_loss, entropy_loss

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_channels: typing.Tuple[int, ...],
        n_clusters: typing.Tuple[int, ...],
        pool_after: int = 2,
    ):
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()

        for it in range(len(conv_channels)):
            self.res_blocks.append(
                DiffPooledConvolutionalNetwork.ResidualBlockPoolStack(
                    input_channel=in_channels if it == 0 else conv_channels[it - 1],
                    channels=conv_channels[it],
                    pool_after=pool_after,
                    n_clusters=n_clusters[it],
                )
            )

        self.lin = torch.nn.Linear(conv_channels[-1], out_channels)

    def forward(self, x, edge_index, batch, _batch_ptr):
        x, mask = pyg.utils.to_dense_batch(x, batch)
        adj = pyg.utils.to_dense_adj(edge_index, batch)

        # Mask out the invalid nodes created for making dense batches
        batch_size, num_nodes, _ = x.size()
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * mask

        total_link_loss = torch.tensor(0.0, device=x.device)
        total_entropy_loss = torch.tensor(0.0, device=x.device)
        for i in range(len(self.res_blocks)):
            (x, adj, link_loss, entropy_loss,) = self.res_blocks[
                i
            ](x, adj)
            total_link_loss = total_link_loss + link_loss
            total_entropy_loss = total_entropy_loss + entropy_loss

        x = x.mean(dim=1)

        x = self.lin(x)

        return (
            torch.nn.functional.log_softmax(x, dim=1),
            total_link_loss + total_entropy_loss,
        )
