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
        batch: torch.Tensor,
        batch_ptr: torch.Tensor,
        edge_weight: typing.Optional[torch.Tensor] = None,
        normalize: typing.Optional[bool] = True,
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
        :param batch: tensor of shape (num_nodes,)
        :param batch_ptr: tensor of shape (num_batches,)
        :param normalize: boolean for normalizing the link loss
        :return:
        """
        num_batches = batch_ptr.size(0) - 1
        s = self.pooling_net(
            x, edge_index
        )  # tensor of shape (num_nodes_in_batched_graph, num_output_pools)
        num_pools_per_graph = s.size(1)

        # s = torch.nn.functional.gumbel_softmax(s, hard=True, dim=-1)
        s = torch.nn.functional.softmax(s, dim=-1)
        main_idx = torch.argmax(s, -1, keepdim=True)
        one_hot = torch.zeros_like(s)
        one_hot.scatter_(-1, main_idx, 1)
        s = one_hot - s.detach() + s

        edge_weight = (
            edge_weight
            if edge_weight is not None
            else torch.ones(edge_index.size(1), device=edge_index.device)
        )

        (
            out,
            out_edge_index,
            out_edge_weight,
            link_loss_total,
        ) = self.pool_all_batches(
            s,
            x,
            edge_index,
            edge_weight,
            batch,
            batch_ptr,
            num_pools_per_graph,
            normalize,
        )

        batch = torch.arange(num_batches, device=x.device).repeat_interleave(
            num_pools_per_graph
        )
        batch_ptr = torch.arange(
            0,
            (num_batches + 1) * num_pools_per_graph,
            num_pools_per_graph,
            device=x.device,
        )

        return (
            out,
            out_edge_index,
            out_edge_weight,
            link_loss_total,
            torch.tensor(0.0, device=x.device),  # We don't have any entropy loss
            batch,
            batch_ptr,
        )

    @staticmethod
    def pool_all_batches(
        s: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        batch_ptr: torch.Tensor,
        num_pools_per_graph: int,
        normalize: bool,
    ):
        num_nodes = x.size(0)
        num_batches = batch.max() + 1

        s_sparse = s.to_sparse().coalesce()
        new_indices = s_sparse.indices() + torch.stack(
            [
                torch.zeros(num_nodes, device=x.device, dtype=batch_ptr.dtype),
                batch * num_pools_per_graph,
            ],
            dim=0,
        )
        s_sparse = torch.sparse_coo_tensor(
            indices=new_indices,
            values=s_sparse.values(),
            size=[num_nodes, num_pools_per_graph * num_batches],
        )
        adj_sparse = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_weight,
            size=[x.size(0), x.size(0)],
        )

        # tensor of shape (num_output_pools, node_feature_dim)
        out = torch.sparse.mm(s_sparse.transpose(0, 1), x)
        out_adj = torch.sparse.mm(
            torch.sparse.mm(s_sparse.transpose(0, 1), adj_sparse), s_sparse
        )

        link_loss = adj_sparse - torch.sparse.mm(s_sparse, s_sparse.transpose(0, 1))
        link_loss = torch.norm(link_loss.coalesce().values(), p=2)
        if normalize is True:
            link_loss = link_loss / edge_index.size(1)

        out_edge_index = out_adj.coalesce().indices()
        out_edge_weight = out_adj.coalesce().values()

        return out, out_edge_index, out_edge_weight, link_loss
