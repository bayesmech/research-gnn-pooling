import typing

import torch, torch.nn.functional


class StochPool(torch.nn.Module):
    def __init__(self, pooling_net):
        super().__init__()
        self.pooling_net = pooling_net

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_ptr: torch.Tensor,
        edge_weight: typing.Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> typing.Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Runs the stoch pool operation, similar to diff pool
        :param x: tensor of shape (num_nodes_in_batched_graph, node_feature_dim)
        :param edge_index: tensor of shape (2, num_edges_in_batched_graph)
        :param edge_weight: tensor of shape (num_edges_in_batched_graph,)
        :param batch_ptr: tensor of shape (num_batches,)
        :param normalize: boolean for normalizing the link loss
        :return:
        """
        num_batches = batch_ptr.size(0) - 1
        s = self.pooling_net(
            x
        )  # tensor of shape (num_nodes_in_batched_graph, num_output_pools)
        num_pools_per_graph = s.size(1)
        s = torch.nn.functional.gumbel_softmax(s, hard=True, dim=-1)

        (
            out,
            out_edge_index,
            out_edge_weight,
            link_loss_total,
            entropy_loss_total,
        ) = self.pool_all_batches(
            s, x, edge_index, edge_weight, batch_ptr, num_pools_per_graph, normalize
        )

        batch = (
            torch.arange(num_batches)
            .repeat_interleave(num_pools_per_graph)
            .to(x.device)
        )
        batch_ptr = torch.arange(
            0, (num_batches + 1) * num_pools_per_graph, num_pools_per_graph
        ).to(x.device)

        return (
            out,
            out_edge_index,
            out_edge_weight,
            link_loss_total,
            entropy_loss_total,
            batch,
            batch_ptr,
        )

    @staticmethod
    @torch.jit.script
    def pool_all_batches(
        s: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: typing.Optional[torch.Tensor],
        batch_ptr: torch.Tensor,
        num_pools_per_graph: int,
        normalize: bool,
    ):
        num_batches = batch_ptr.size(0) - 1

        out, out_edge_index, out_edge_weight = [], [], []
        link_loss_total, entropy_loss_total = torch.tensor(
            0.0, device=x.device
        ), torch.tensor(0.0, device=x.device)
        for batch_idx in range(num_batches):
            selected_edges = (batch_ptr[batch_idx] <= edge_index[0]) & (
                edge_index[0] < batch_ptr[batch_idx + 1]
            )
            num_nodes_in_graph = int(
                (batch_ptr[batch_idx + 1] - batch_ptr[batch_idx]).item()
            )
            x_graph = x[batch_ptr[batch_idx] : batch_ptr[batch_idx + 1]]
            edge_index_graph = edge_index[:, selected_edges] - batch_ptr[batch_idx]
            s_graph = s[batch_ptr[batch_idx] : batch_ptr[batch_idx + 1]]
            edge_weight_graph = (
                edge_weight[selected_edges]
                if edge_weight is not None
                else torch.ones(
                    edge_index_graph.size(1), device=edge_index_graph.device
                )
            )

            s_graph_sparse = s_graph.to_sparse()
            adj_graph_sparse = torch.sparse_coo_tensor(
                indices=edge_index_graph,
                values=edge_weight_graph,
                size=[num_nodes_in_graph, num_nodes_in_graph],
            )

            # tensor of shape (num_output_pools, node_feature_dim)
            out_graph = torch.mm(s_graph.transpose(0, 1), x_graph)
            out_adj_graph = torch.sparse.mm(
                torch.sparse.mm(s_graph_sparse.transpose(0, 1), adj_graph_sparse),
                s_graph_sparse,
            )

            link_loss_graph = adj_graph_sparse - torch.sparse.mm(
                s_graph_sparse, s_graph_sparse.transpose(0, 1)
            )
            link_loss_graph = torch.norm(link_loss_graph.coalesce().values(), p=2)
            if normalize is True:
                link_loss_graph = link_loss_graph / edge_index_graph.size(1)

            entropy_loss_graph = (
                (-s_graph * torch.log(s_graph + 1e-15)).sum(dim=-1).mean()
            )

            out.append(out_graph)
            out_edge_index.append(
                out_adj_graph.coalesce().indices() + batch_idx * num_pools_per_graph
            )
            out_edge_weight.append(out_adj_graph.coalesce().values())
            link_loss_total += link_loss_graph
            entropy_loss_total += entropy_loss_graph

        out = torch.cat(out, dim=0)
        out_edge_index = torch.cat(out_edge_index, dim=1)
        out_edge_weight = torch.cat(out_edge_weight, dim=0)
        link_loss_total /= num_batches
        entropy_loss_total /= num_batches

        return out, out_edge_index, out_edge_weight, link_loss_total, entropy_loss_total
