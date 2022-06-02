import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfExpressiveLayer(nn.Module):
    def __init__(self, num_samples, init_affinity=None, device='cpu'):
        super(SelfExpressiveLayer, self).__init__()
        self.init_affinity = init_affinity
        self.device = device
        self.affinity_mat = nn.Parameter(torch.ones(num_samples, num_samples), requires_grad=True).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.affinity_mat, a=math.sqrt(5))
        if self.init_affinity is not None:
            if not isinstance(self.init_affinity, torch.Tensor):
                self.init_affinity = torch.from_numpy(self.init_affinity).float()
            self.affinity_mat.data = self.init_affinity
        else:
            nn.init.constant_(self.affinity_mat, 1.)

    def forward(self, x):
        latent_recon = torch.matmul(self.affinity_mat, x)
        return latent_recon


class Net(nn.Module):

    def __init__(self, in_channel, patch_size, association_mat, init_affinity=None, device='gpu'):
        super(Net, self).__init__()
        if not isinstance(association_mat, torch.Tensor):
            association_mat = torch.from_numpy(association_mat).float()
        self.association_mat = association_mat.to(device)
        # # normalize column of association_mat for averaging pixels
        self.norm_association_mat = self.association_mat / self.association_mat.sum(dim=0)
        self.patch_size = patch_size
        self.init_affinity = init_affinity
        num_spixel = association_mat.shape[1]

        if patch_size[0] < 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channel, 64, (1, 1), stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, (1, 1), stride=1, padding=0),  # # N*(W-2)*(W-2)*64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(64, 64, (1, 1), stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, in_channel, (1, 1), stride=1, padding=0),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channel, 64, (1, 1), stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, (3, 3), stride=1, padding=0),  # # N*(W-2)*(W-2)*64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, in_channel, (1, 1), stride=1, padding=0),
            )

        self.self_expr_layer = SelfExpressiveLayer(num_spixel, init_affinity=self.init_affinity, device='cpu')  # self.init_affinity, patch_size[0]*patch_size[1]*32,

        self.sp_lin1 = nn.Sequential(nn.Linear(256, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True),
                                     # nn.Dropout(0.5)
                                     )
        self.sp_lin2 = nn.Sequential(nn.Linear(512, 64),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     )

    def forward(self, x, sp_adj=None):
        x = self.encoder(x)
        x = torch.matmul(self.norm_association_mat.t(), x)  # n_spixel * n_dim
        # pixel_to_sp = self.graph_att(pixel_to_sp, sp_adj)
        pixel_to_sp = self.sp_lin1(x)
        latent_recon = self.self_expr_layer(pixel_to_sp)
        sp_to_pixel = self.sp_lin2(latent_recon)
        sp_to_pixel = torch.matmul(self.association_mat, sp_to_pixel)  # n_pixel * n_dim
        sp_to_pixel = sp_to_pixel.view((-1, 64, 1, 1))
        sp_to_pixel = F.interpolate(sp_to_pixel, size=(self.patch_size[0], self.patch_size[1]),
                                    mode='bilinear',
                                    align_corners=True)
        x_recon = self.decoder(sp_to_pixel)
        return pixel_to_sp, latent_recon, x_recon


class NCL_Loss_Graph(nn.Module):
    """
    neighborhood contrastive learning loss based on using a N*N affinity/similarity matrix/adjacent matrix
    """
    def __init__(self, n_neighbor=5, temperature=1.):
        """
        :param n_neighbor:
        :param temperature:
        """
        super(NCL_Loss_Graph, self).__init__()
        self.n_neighbor = n_neighbor
        self.temperature = temperature
        self.BCE = nn.BCELoss()

    def forward(self, affinity_pred, affinity_init):
        N = affinity_init.shape[0]
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0).bool().to(affinity_pred.device)
        affinity_pred, affinity_init = affinity_pred[mask].view((N, -1)), affinity_init[mask].view((N, -1)).to(affinity_pred.device)

        _, pos_indx = torch.topk(affinity_init, k=self.n_neighbor, dim=1, largest=True, sorted=True)
        sim_matrix = torch.exp(affinity_pred / self.temperature)
        pos_sim = torch.gather(sim_matrix, dim=1, index=pos_indx)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1).view(N, -1) + 1e-8)).sum(dim=1).mean()
        return loss


class GraphRegularizer(nn.Module):

    def __init__(self, aff_init, n_neighbor=10):
        super(GraphRegularizer, self).__init__()

        res, pos_indx = torch.topk(aff_init, k=n_neighbor, dim=1, largest=True, sorted=True)
        A = torch.zeros((aff_init.shape[0], aff_init.shape[0]))
        A = A.scatter(src=torch.ones_like(res), dim=1, index=pos_indx)
        A = (A + A.t())/2
        # A = torch.where(A>=2, A, torch.zeros_like(A))/2
        # A.fill_diagonal_(0)
        D = torch.pow(torch.sum(A, dim=1), -0.5)
        D = torch.diag(D)
        L = torch.eye(A.shape[0]) - torch.matmul(torch.matmul(D, A), D)
        self.L = L

    def forward(self, pred_aff):
        L = self.L.to(pred_aff.device)
        loss = torch.trace(pred_aff.t().mm(L).mm(pred_aff))
        return loss


class Loss(nn.Module):
    def __init__(self, trade_off=0.005, trade_off_sparse=0.005, n_neighbor=10, temperature=0.1, regularizer='NC', aff_init=None):
        """

        :param trade_off:
        :param trade_off_sparse:
        :param n_neighbor:
        :param temperature:
        :param regularizer:  'NC': neighbor contrastive; 'L1'/'L2', None
        :param aff_init:
        """
        super(Loss, self).__init__()
        self.regularizer = regularizer
        self.trade_off = trade_off
        self.trade_off_sparse = trade_off_sparse
        self.criterion_mse_recon = nn.MSELoss()
        self.criterion_mse_latent = nn.MSELoss()
        self.aff_init = aff_init
        self.eta = nn.Parameter(torch.Tensor([0, -1., -10.]), requires_grad=True)

        # self.eta = nn.Parameter(torch.Tensor([0.1, -1., -10.]), requires_grad=True)
        if aff_init is not None and not isinstance(aff_init, torch.Tensor):
            self.aff_init = torch.from_numpy(aff_init).float()
            self.graph_reg = GraphRegularizer(self.aff_init, n_neighbor=n_neighbor)
        self.ncl_criterion = NCL_Loss_Graph(n_neighbor=n_neighbor, temperature=temperature)

    def forward(self, model, x_true, x_predict, z, z_pred):
        loss_regularization = torch.tensor(0., requires_grad=False).to(x_true.device)
        for name, param in model.named_parameters():
            if 'affinity_mat' in name:
                # loss_regularization += torch.sum(torch.norm(param, p=2))
                # loss_regularization += torch.abs(torch.diagonal(param)).sum()
                # loss_regularization += self.ncl_criterion(param, self.aff_init)
                if self.regularizer == 'NC':
                    loss_regularization += self.ncl_criterion(param, self.aff_init)
                elif self.regularizer == 'L1':
                    loss_regularization += torch.sum(torch.abs(param))
                elif self.regularizer == 'L2':
                    loss_regularization += torch.norm(param, p='fro')
                elif self.regularizer == 'None':
                    loss_regularization = loss_regularization
                elif self.regularizer == 'Graph':
                    loss_regularization += self.graph_reg(param)
                else:
                    raise Exception('invalid regularizer!')
        loss_model_recon = self.criterion_mse_recon(x_predict, x_true)
        loss_se_recon = self.criterion_mse_latent(z_pred, z)
        if self.regularizer == 'NC':
            loss_ = torch.stack([loss_model_recon, loss_se_recon, loss_regularization]).to(x_true.device)
            loss = (loss_ * torch.exp(-self.eta) + self.eta).sum()
        else:
            loss = loss_model_recon + self.trade_off * loss_se_recon + self.trade_off_sparse * loss_regularization
        return loss, loss_model_recon, loss_se_recon, loss_regularization, torch.exp(-self.eta)
