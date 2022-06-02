import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/caiyaom/python_codes/')
sys.path.append('/root/python_codes/')
import os
import numpy as np
import torch
import argparse
from module import HSI_Loader
from module import net
from utils import yaml_config_hook, save_model
from utils.postprocess import spixel_to_pixel_labels, affinity_to_pixellabels
from utils.evaluation import cluster_accuracy, get_parameter_number
import time


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


def train():
    model.train()  # # training flag
    for step, (x_batch, y_batch) in enumerate(data_loader):
        optimizer.zero_grad()
        x_batch = x_batch.to(DEVICE)
        latent, latent_recon, x_recon = model(x_batch, dataset.sp_graph)
        loss_, loss_model_recon, loss_se_recon, loss_regularization, loss_w = loss_criterion(model, x_batch, x_recon, latent, latent_recon)
        loss_.backward()
        optimizer.step()

    return loss_.item(), loss_model_recon.item(), loss_se_recon.item(), \
           loss_regularization.item(), loss_w.detach().cpu().numpy().tolist()


def evaluate():
    model.eval()  # # training flag
    for name, param in model.named_parameters():
        if 'affinity_mat' in name:
            affinity_mat_ = param.detach().cpu().numpy()
    y_pre_sp = affinity_to_pixellabels(affinity_mat_, dataset.n_classes)
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, dataset.association_mat)
    class_map = y_pre_pixel.reshape(dataset.gt.shape)
    np.savez(f'{args.dataset}-class-map-affinity.npz', y_pred=class_map, affinity_mat_=affinity_mat_)
    # adj = 0.5 * (np.abs(affinity_mat_) + np.abs(affinity_mat_.T))
    # adj[np.where(dataset.sp_graph == 0)] = 0

    # show_graph(adj, dataset.sp_centroid)

    # pro = Processor()
    # color_map = pro.colorize_map(class_map, colors=None)
    # plt.imshow(color_map)
    # plt.show()

    indx = np.where(dataset.gt != 0)
    y_target = dataset.gt[indx]
    y_predict = class_map[indx]
    acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=False)
    # acc, kappa, nmi, ari, pur, ca = cluster_eva(y_target, y_predict, is_HSI_score=True)
    print(
        'Epoch = {:}\n OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} '
        'Purity = {:.4f}'.format(epoch, acc, kappa, nmi, ari, pur))
    for j in np.round(np.array(ca), 4):
        print(j)
    acc_history.append([acc, kappa, nmi, ari, pur])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = args.dataset_root
    # prepare data
    if args.dataset == "HSI-InP-2010":
        im_, gt_ = 'Indian_Pines_2010', 'Indian_Pines_2010_gt'
    elif args.dataset == "HSI-SaN":
        im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    elif args.dataset == "HSI-TrT":
        im_, gt_ = 'Trento-HSI', 'Trento-GT'
    else:
        raise NotImplementedError
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'

    dataset = HSI_Loader.HSI_Data(img_path, gt_path,
                                  path_to_sp=args.sp_path,
                                  patch_size=(args.image_size, args.image_size),
                                  num_superpixel=args.num_superpixel,
                                  pca=True, is_superpixel=True, pca_dim=args.in_channel,
                                  is_labeled=False, transform=None)
    class_num = dataset.n_classes
    # plt.imshow(dataset.association_mat, cmap='jet')
    # plt.show()

    print('Processing %s ' % img_path)
    print(dataset.data_size, dataset.n_classes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.data_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    # initialize model
    model = net.Net(args.in_channel, (args.image_size, args.image_size), dataset.association_mat,
                    init_affinity=None,  # dataset.sp_graph,
                    device=DEVICE)
    model = model.to(DEVICE)

    # print(model)
    # summary(model, input_size=(args.in_channel, args.image_size, args.image_size), batch_size=dataset.data_size)
    get_parameter_number(model)

    # from thop import profile
    # inputs = torch.randn(dataset.data_size, 10, 7, 7).to(DEVICE)
    # flops, params = profile(model, [inputs, dataset.sp_graph])
    # print('flops: ', flops, 'params: ', params)

    # optimizer / loss
    loss_criterion = net.Loss(args.loss_tradeoff, args.sparse_tradeoff,
                              n_neighbor=args.n_neighbor,
                              temperature=args.temperature,
                              regularizer=args.regularizer,
                              aff_init=dataset.sp_graph).to(DEVICE)
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if 'affinity_mat' in n],
         'lr': args.learning_rate * 20},  # # set SE layer
        {"params": [p for n, p in model.named_parameters() if 'affinity_mat' not in n],
         'lr': args.learning_rate,
         'weight_decay':args.weight_decay},
        {"params":  loss_criterion.parameters(), 'lr': args.learning_rate}
    ]
    # grouped_parameters = list(model.parameters()) + list(loss_criterion.parameters())
    optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_history = []
    acc_history = []
    loss_wei_history = []
    # train
    # epoch = -1
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        # print('%s, LR %s ==============================================' % (epoch, lr))
        if epoch % 5 == 0:
            evaluate()
            save_model(args, model, optimizer, epoch)
        loss_epoch, loss_recon, loss_se, loss_reg, loss_weights = train()
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}"
              f"\t Loss model recon: {loss_recon}\t Loss SE: {loss_se}\t Loss Reg: {loss_reg}\t Loss Wei: {loss_weights}")
        loss_history.append([loss_epoch, loss_recon, loss_se, loss_reg])
        loss_wei_history.append(loss_weights)
        lr_scheduler.step()
    save_model(args, model, optimizer, args.epochs)
    print(loss_history)
    print(acc_history)
    print(loss_wei_history)
    print(f'running time {time.time() - start_time}')

