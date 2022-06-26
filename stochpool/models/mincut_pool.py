import torch_geometric as pyg
import torch, torch.nn.functional


class MinCutPooledConvolutionalNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(in_channels, hidden_channels)
        self.pool1 = torch.nn.Linear(hidden_channels, 100)
        self.conv2 = pyg.nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.pool2 = torch.nn.Linear(hidden_channels, 10)
        self.conv3 = pyg.nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Initial convolutions
        x = self.conv1(x, edge_index).relu()
        x, mask = pyg.utils.to_dense_batch(x, batch)
        adj = pyg.utils.to_dense_adj(edge_index, batch)
        # First round of pool-conv
        s = self.pool1(x)
        x, adj, mc1, o1 = pyg.nn.dense_mincut_pool(x, adj, s, mask)
        x = self.conv2(x, adj).relu()
        # Second round of pool-conv
        s = self.pool2(x)
        x, adj, mc2, o2 = pyg.nn.dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        # Final global pool and output
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1), mc1 + mc2 + o1 + o2
