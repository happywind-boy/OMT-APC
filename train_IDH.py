# -- coding : uft-8 --
# Author : Wang Han
# Southeast University
import argparse
from monai.data import CSVSaver, DataLoader
from src.dataset import *
from src.trainer import *
from src.logger import *
from src.config import *
from src.utils import *
from src.metric import *
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# from pyecharts import options as opt
parser = argparse.ArgumentParser()

# set dataset
parser.add_argument('--data', type=str, default='omt', help='omt or raw')
# parser.add_argument(
#    '--data_file', type=str,
#    default=["/njcam_share/gj/IDH_classfication/UCSF/"],
#    nargs='+', help='training file')
parser.add_argument(
    '--label_file', type=str,
    default=["/njcam_share/wh/Classification/feature/data_table.csv", ],
    nargs='+', help='label file')

parser.add_argument(
    '--feature_file', type=str,
    default=[
        "/njcam_share/wh/Classification/feature/Class_Data_norm.csv",
    ],
    nargs='+', help='feature file')

parser.add_argument(
    '--tensor_feature_file', type=str,
    default=[
        "/njcam_share/wh/Classification/feature/feature-IDH_rank20_weival-weitest-WPXtikonov-diag_1e-4.csv",
    ],
    nargs='+', help='tensor feature file')

# set model
parser.add_argument('--model', type=str, default='preresnet20', help='model')
parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
parser.add_argument('--out_channels', type=int, default=2, help='number of output channels')
parser.add_argument('--hidden_channels', type=int, default=16, help='number of hidden channels in ResI3D')
parser.add_argument('--linear_channels', type=int, default=20, help='number of Last linear channels in Pre-SEResNet')
parser.add_argument('--reduction', type=int, default=16, help='number of reduction')
parser.add_argument('--feature_channels', type=int, default=64, help='number of radiomic features')
parser.add_argument('--new_feature_channels', type=int, default=12, help='number of cat-vector length')
parser.add_argument('--init_filters', type=int, default=32, help='number of output initial filters')
parser.add_argument('--norm', type=str, default='INSTANCE', help='normalization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--act', type=str, default='softmax', help='activation function')
parser.add_argument('--pretrain_weight', type=str, default='', help='path of pretrain weight')

# set training
parser.add_argument('-ep', '--epoches', type=int, default=400, help='number of epoches')
parser.add_argument('-bs', '--batch_size', type=int, default=2, help='number of batch size')
parser.add_argument('--valid_period', type=int, default=5, help='test period')
parser.add_argument('--save_topk', type=int, default=10, help='number of save checkpoint')

# set optimization
parser.add_argument('--loss', type=str, default='CEL', help='loss function')
parser.add_argument('--optim', type=str, default='adam', help='optimizer')
parser.add_argument('--scheduler', type=str, default='step', help='learning rate schedule')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--lr_decay_period', type=int, default=10, help='learning rate decay period')
parser.add_argument('--lr_decay_factor', type=float, default=0.98, help='learning rate decay factor')

# set augmentation
parser.add_argument('--prob_rot90', type=float, default=0.25, help='probability of rot 90 degree')
parser.add_argument('--prob_fliplr', type=float, default=0.25, help='probability of flip left and right')
parser.add_argument('--prob_flipud', type=float, default=0.25, help='probability of flip up and down')
parser.add_argument('--prob_noise', type=float, default=0.1, help='probability of random gaussian noise')

# set device
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
parser.add_argument('--device', type=int, default=[2], nargs='+', help='index of gpu device')

# set root
parser.add_argument('--log_save_path', type=str,
                    default="/njcam_share/wh/Classification/code/result",
                    help='log_save_path')
parser.add_argument('--train_id', type=str,
                    default=["/njcam_share/wh/Classification/code/index/IDH-training.txt", ],
                    nargs='+', help='train_id')
parser.add_argument('--val_id', type=str,
                    default="/njcam_share/wh/Classification/code/index/IDH-TCGA.txt",
                    help='val_id')
parser.add_argument('--test_id', type=str,
                    default="/njcam_share/wh/Classification/code/index/IDH-testing.txt",
                    help='test_id')
# set information
parser.add_argument('--information', type=str, default='IDH-Img-Fea', help='record')

opt = parser.parse_args()


def train(opt):

    data_file = {'omt': ['/njcam_share/wh/Classification/Data/','/njcam_share/wh/Classification/Data1.0/',],
                 'raw': ['/njcam_share/wh/Classification/Data/','/njcam_share/wh/Classification/Data1.0/',]}
    data_loc = data_file[opt.data]
    
    train_id = []
    file_ls = []
    for k in range(len(opt.train_id)):
        train_idk = []
        file_lsk = []
        f = open(opt.train_id[k], encoding='gbk')
        for line in f:
            train_idk.append(line.strip())
            file_lsk.append(data_loc[k])
        train_id += train_idk
        file_ls += file_lsk
    
    #print(len(train_id))
    #print(len(file_ls))
        
    #f = open(opt.train_id, encoding='gbk')
    #for line in f:
    #    train_id.append(line.strip())

    val_id = []
    f = open(opt.val_id, encoding='gbk')
    for line in f:
        val_id.append(line.strip())

    test_id = []
    f = open(opt.test_id, encoding='gbk')
    for line in f:
        test_id.append(line.strip())

    save_name = f'{opt.model}_{opt.information}_{opt.loss}_{datetime.now().year}-{datetime.now().month}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}'
    print('Path:', save_name)
    print('Fea:', opt.tensor_feature_file[0])
    save_root = os.path.join(opt.log_save_path, save_name)
    if not os.path.exists(save_root):
        os.makedirs(os.path.join(save_root, 'weight'))

    log = Logger(os.path.join(save_root, f'result.csv'))
    save_json(opt, os.path.join(save_root, 'config.json'))

    get_dataset = {'omt': get_train_val_test_omt_dataset, 'raw': get_train_val_test_dataset}

    device = torch.device(f"cuda:{opt.device[0]}" if torch.cuda.is_available() else "cpu")
    print('GPU:', opt.device)

    model = get_model(opt)
    print('Model:', opt.model)
    model = model.to(device)

    criterion = get_criterion(opt)
    print('Loss:', opt.loss)
    optimizer = get_optimizer(opt, model)
    scheduler = get_lr_scheduler(opt, optimizer)
    print('Optim:', opt.optim)

    epoches = opt.epoches

    train_set, val_set, test_set = get_dataset[opt.data](
        train_id, val_id, test_id, file_ls, data_loc, opt)

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False
    )

    train_record = init_record()
    best_auc = 0

    for epoch in range(epoches):
        model.train()

        train_bar = tqdm(train_loader)

        pred_all = []
        label_all = []
        acc_correct = []
        total_count = []

        loss_train = 0
        for train_data in train_bar:
            optimizer.zero_grad()
            ID = train_data['ID']
            train_images, train_labels = train_data["image"].to(device), train_data["label"].to(device)
            #train_features1 = train_data["feature1"].to(device)
            train_features2 = train_data["feature2"].to(device)

            #train_outputs = model(train_images)
            train_outputs = model(train_images, train_features2)
            pred = train_outputs.float()

            label = train_labels.squeeze(dim=1)
            loss = criterion(pred, label)
            loss.requires_grad_(True)

            pred1 = pred.argmax(dim=1)

            value = torch.eq(pred1, label)
            total_count.append(len(value))
            acc_correct.append(value.sum().item())
            pred_all.extend(pred1.cpu())
            label_all.extend(label.cpu())

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            train_bar.desc = 'Epoch{}/{},loss:{:4f}'.format(epoch, epoches, loss.item())

        sen_train, spe_train = metrics(pred_all, label_all)
        acc_train = sum(acc_correct) / sum(total_count)
        auc_train = roc_auc_score(label_all, pred_all)
        print('Training ACC:{:.4f}'.format(acc_train))
        print('Training AUC:{:.4f}'.format(auc_train))
        print('Training Sensitivity:{:.4f}'.format(sen_train))
        print('Training Specificity:{:.4f}'.format(spe_train))
        train_record = {
            'Loss': loss_train / len(train_bar),
            'ACC': acc_train,
            'AUC': auc_train,
            'Sensitivity': sen_train.item(),
            'Specificity': spe_train.item()
        }

        log.add(epoch=epoch, type='train', **train_record)
        scheduler.step()
        train_bar.close()

        if epoch >= 30:
            model.eval()
            val_record = init_record()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0

                pred_val = []
                label_val = []

                loss_val = 0
                val_bar = tqdm(val_loader, desc='Validation')
                for val_data in val_bar:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_features1 = val_data["feature1"].to(device)
                    val_features2 = val_data["feature2"].to(device)

                    #val_outputs = model(val_images)
                    val_outputs = model(val_images, val_features2)
                    pred = val_outputs.float()
                    label = val_labels.squeeze(dim=1)

                    loss = criterion(pred, label)
                    pred1 = pred.argmax(dim=1)

                    value = torch.eq(pred1, label)
                    metric_count += len(value)
                    num_correct += value.sum().item()

                    pred_val.extend(pred1.cpu())
                    label_val.extend(label.cpu())

                    loss_val += loss.item()

                sen_val, spe_val = metrics(pred_val, label_val)
                acc_val = num_correct / metric_count
                auc_val = roc_auc_score(label_val, pred_val)
                print('Validation ACC:{:.4f}'.format(acc_val))
                print('Validation AUC:{:.4f}'.format(auc_val))
                print('Validation Sensitivity:{:.4f}'.format(sen_val))
                print('Validation Specificity:{:.4f}'.format(spe_val))
                val_record = {
                    'Loss': loss_val / len(val_bar),
                    'ACC': acc_val,
                    'AUC': auc_val,
                    'Sensitivity': sen_val.item(),
                    'Specificity': spe_val.item()
                }
                log.add(epoch=epoch, type='val', **val_record)
                log.save()
                
                torch.save(model.state_dict(),
                           os.path.join(save_root, 'weight',
                                        f'weight_{epoch}_{acc_val:.4f}_{auc_val:.4f}.pth'))

                #if auc_val > best_auc:
                #    best_auc = acc_val
                #    model_path = os.path.join(save_root, 'weight',
                #                              f'weight_{epoch}_{acc_val:.4f}_{auc_val:.4f}.pth')
                #    torch.save(model.state_dict(), model_path)
            
            with torch.no_grad():
                num_correct_test = 0.0
                metric_count_test = 0

                loss_test = 0
                test_bar = tqdm(test_loader, desc='Testing')
                test_pred_all = []
                test_label_all = []
                test_acc_correct = []
                test_total_count = []

                test_record = init_record()

                for test_data in test_bar:
                    test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
                    #test_features1 = test_data["feature1"].to(device)
                    test_features2 = test_data["feature2"].to(device)

                    #test_outputs = model(test_images)
                    test_outputs = model(test_images, test_features2)

                    test_pred = test_outputs.float()
                    test_label = test_labels.squeeze(dim=1)

                    test_loss = criterion(test_pred, test_label)
                    test_pred1 = test_pred.argmax(dim=1)
                    loss_test += test_loss.item()

                    test_value = torch.eq(test_pred1, test_label)
                    metric_count_test += len(test_value)
                    num_correct_test += test_value.sum().item()

                    test_total_count.append(len(test_value))
                    test_acc_correct.append(test_value.sum().item())

                    test_pred_all.extend(test_pred1.cpu())
                    test_label_all.extend(test_label.cpu())

                sen_test, spe_test = metrics(test_pred_all, test_label_all)
                acc_test = num_correct_test / metric_count_test
                auc_test = roc_auc_score(test_label_all, test_pred_all)
                print('Test ACC:{:.4f}'.format(acc_test))
                print('Test AUC:{:.4f}'.format(auc_test))
                print('Test Sensitivity:{:.4f}'.format(sen_test))
                print('Test Specificity:{:.4f}'.format(spe_test))
                test_record = {
                    'Loss': loss_test / len(test_bar),
                    'ACC': acc_test,
                    'AUC': auc_test,
                    'Sensitivity': sen_test.item(),
                    'Specificity': spe_test.item()
                }
                log.add(epoch=epoch, type='test', **test_record)
                log.save()
                
                #torch.save(model.state_dict(),
                #           os.path.join(save_root, 'weight',
                #                        f'weight_{acc_test:.4f}_{auc_test:.4f}.pth'))

        # predict_model = get_model(opt)
        # predict_model.load_state_dict(torch.load(model_path))
        # predict_model.to(device)
        # predict_model.eval()
        # test_bar = tqdm(test_loader)
        #
        # test_pred_all = []
        # test_label_all = []
        # test_acc_correct = []
        # test_total_count = []
        #
        # for test_data in test_bar:
        #     test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
        #     test_features1 = test_data["feature1"].to(device)
        #     test_features2 = test_data["feature2"].to(device)
        #
        #     test_outputs = model(test_images, test_features1, test_features2)
        #
        #     test_pred = test_outputs.float()
        #     test_label = test_labels.squeeze(dim=1)
        #
        #     test_value = torch.eq(test_pred, test_label)
        #     test_total_count.append(len(test_value))
        #     test_acc_correct.append(value.sum().item())
        #     test_pred_all.extend(pred1.cpu())
        #     test_label_all.extend(label.cpu())
        #
        # sen_test, spe_test = metrics(test_pred_all, test_label_all)
        # acc_test = sum(acc_correct) / sum(total_count)
        # auc_test = roc_auc_score(label_all, pred_all)
        # print('Test ACC:{:.4f}'.format(acc_test))
        # print('Test AUC:{:.4f}'.format(auc_test))
        # print('Test Sensitivity:{:.4f}'.format(sen_test))
        # print('Test Specificity:{:.4f}'.format(spe_test))
        # test_record = {
        #     'ACC': acc_test,
        #     'AUC': auc_test,
        #     'Sensitivity': sen_test.item(),
        #     'Specificity': spe_test.item()
        # }
        # log.add(epoch=epoch, type='test', **test_record)
        # log.save()
        # idx += 1


if __name__ == "__main__":
    train(opt)
