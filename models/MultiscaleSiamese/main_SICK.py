# -*-coding:utf-8-*-
# from config_SICK_hcti import opt
from config_SICK_lstm import opt

import os
import os.path
import torch
import models
from data.dataset_vector_feature_align import STSDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from inspect import getsource
from utils.visualize import Visualizer
import scipy.stats as mea

def train(**kwargs):
    # torch.manual_seed(100) # 10, 100, 666,
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: load data
    if os.path.isfile(opt.train_features_path) and\
       os.path.isfile(opt.train_targets_path):
        print "load train dataset from file"
        features = torch.load(opt.train_features_path)
        # features[features == float('Inf')] = 0  # for errors
        targets = torch.load(opt.train_targets_path)
        train_data = torch.utils.data.TensorDataset(features, targets)
        train_dataloader = DataLoader(train_data, opt.batch_size,
                                      shuffle=True,
                                      num_workers=opt.num_workers)
    else:
        train_data = STSDataset(opt.train_data_path, opt)
        train_dataloader = DataLoader(train_data, opt.batch_size,
                                      shuffle=True,
                                      num_workers=opt.num_workers)
        torch.save(train_data.X, opt.train_features_path)
        torch.save(train_data.y, opt.train_targets_path)

    if os.path.isfile(opt.dev_features_path) and\
       os.path.isfile(opt.dev_targets_path):
        print "load dev dataset from file"
        features = torch.load(opt.dev_features_path)
        # features[features == float('Inf')] = 0  # for errors
        targets = torch.load(opt.dev_targets_path)
        dev_data = torch.utils.data.TensorDataset(features, targets)
        dev_dataloader = DataLoader(dev_data, opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers)
    else:
        dev_data = STSDataset(opt.dev_data_path, opt)
        dev_dataloader = DataLoader(dev_data, opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers)
        torch.save(dev_data.X, opt.dev_features_path)
        torch.save(dev_data.y, opt.dev_targets_path)

    if os.path.isfile(opt.test_features_path) and\
       os.path.isfile(opt.test_targets_path):
        print "load test dataset from file"
        features = torch.load(opt.test_features_path)
        # features[features == float('Inf')] = 0  # for errors
        targets = torch.load(opt.test_targets_path)
        test_data = torch.utils.data.TensorDataset(features, targets)
        test_dataloader = DataLoader(test_data, opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.num_workers)
    else:
        test_data = STSDataset(opt.test_data_path, opt)
        test_dataloader = DataLoader(test_data, opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.num_workers)
        torch.save(test_data.X, opt.test_features_path)
        torch.save(test_data.y, opt.test_targets_path)

    # step3: set criterion and optimizer
    criterion = torch.nn.MSELoss()
    lr = opt.lr
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
                                    # weight_decay=opt.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # step4: set meters
    loss_meter = meter.MSEMeter()
    previous_loss = 1e100

    # train
    test_p = []
    dev_p_s_m = []
    test_p_s_m = []
    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii, (data, label) in enumerate(train_dataloader):
            # train model on a batch data
            input = Variable(data)
            target = Variable(torch.FloatTensor(label.numpy()))
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            # print score

            loss.backward()
            optimizer.step()

            # update meters and visualize
            loss_meter.add(score.data, target.data)

            # if ii % opt.print_freq == opt.print_freq - 1:
            #     vis.plot('loss', loss_meter.value())

        # save model for each epoch
        # model.save()

        # validate and visualize
        train_mse, train_pearsonr, train_spear = val(model, train_dataloader)
        val_mse, val_pearsonr, val_spear = val(model, dev_dataloader)
        test_mse, test_pearsonr, test_spear = val(model, test_dataloader)
        dev_p_s_m.append([val_pearsonr, val_spear, val_mse])
        test_p_s_m.append([test_pearsonr, test_spear, test_mse])
        test_p.append(test_spear)
        print('epoch %d' %epoch)
        print('test spear & pearson: ', test_spear, test_pearsonr)



        # vis.plot_many({"train_mse": train_mse,
        #                "val_mse": val_mse,
        #                "test_mse": test_mse})
        # vis.plot_many({"train_pearsonr": train_pearsonr,
        #                "val_pearsonr": val_pearsonr,
        #                "test_pearsonr": test_pearsonr})
        # vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, \
        #          train_mse:{train_mse}, train_pearson:{train_pearson}, \
        #          val_mse:{val_mse}, val_pearson:{val_pearson}, \
        #          test_mse:{test_mse}, test_pearson:{test_pearson}".format(
        #     epoch=epoch, lr=lr, loss=loss_meter.value(),
        #     train_mse=str(train_mse), train_pearson=str(train_pearsonr),
        #     val_mse=str(val_mse), val_pearson=str(val_pearsonr),
        #     test_mse=str(test_mse), test_pearson=str(test_pearsonr)))

        # update learning rate
        if train_mse > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = train_mse

    x1 = max(test_p)
    ind = test_p.index(x1)
    return dev_p_s_m[ind], test_p_s_m[ind], ind

def val(model, dataloader):
    """ Test model accuracy on validation dataset.
    """
    model.eval()
    mse = meter.MSEMeter()
    all_score = []
    all_label = []
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(torch.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        mse.add(score.data.squeeze(), label.type(torch.FloatTensor))
        all_score.extend(score.data.numpy().reshape(-1).tolist())
        all_label.extend(label.numpy().reshape(-1).tolist())
    model.train()
    mse_value = mse.value()
    pearsonr_value, b = mea.pearsonr(all_score, all_label)
    spear = mea.spearmanr(all_score, all_label)[0]
    return mse_value, pearsonr_value, spear


if __name__ == '__main__':
    # os.system("python -m visdom.server")  # http://localhost:8097/
    # import fire
    # fire.Fire()
    dev1, test1, ind = train()
    print('best test pearson: ' + str(test1)+'in epoch %d' %ind)
