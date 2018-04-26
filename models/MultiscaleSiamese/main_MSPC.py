# -*-coding:utf-8-*-
# from config_MSPC_hcti import opt
from config_MSPC_lstm import opt

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
from sklearn.metrics import f1_score


def train(**kwargs):
    # torch.manual_seed(100) # 10, 100, 666,
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

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
        targets = torch.load(opt.train_targets_path) * 5  # !!!!!!!!!!!!!!!!!
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

    if os.path.isfile(opt.test_features_path) and\
       os.path.isfile(opt.test_targets_path):
        print "load test dataset from file"
        features = torch.load(opt.test_features_path)
        # features[features == float('Inf')] = 0  # for errors
        targets = torch.load(opt.test_targets_path) * 5  # !!!!!!!!!!!!!!!!!
        test_data = torch.utils.data.TensorDataset(features, targets)
        test_dataloader = DataLoader(test_data, opt.batch_size,
                                     shuffle=False,
                                     num_workers=opt.num_workers)
    else:
        test_data = STSDataset(opt.test_data_path, opt)
        test_dataloader = DataLoader(test_data, opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.num_workers)
        torch.save(test_data.X, opt.test_features_path)
        torch.save(test_data.y, opt.test_targets_path)

    # step3: set criterion and optimizer
    # criterion = torch.nn.MarginRankingLoss()
    criterion = torch.nn.BCELoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                                 # weight_decay=opt.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    #                                 weight_decay=opt.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # step4: set meters
    loss_meter = meter.AUCMeter()
    # loss_meter = meter.ClassErrorMeter()
    previous_loss = 1e100

    # train
    testf1 = []
    test_f_a_l = []
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
            #     vis.plot('loss', loss_meter.value()[0])

        # save model for each epoch
        # model.save()

        # validate and visualize
        train_ce, train_acc, train_f1 = val(model, train_dataloader)
        test_ce, test_acc, test_f1 = val(model, test_dataloader)
        testf1.append(test_acc)
        test_f_a_l.append([test_f1, test_acc, test_ce])
        print('epoch: %d' %epoch)
        print('test acc, f1: '+str(test_acc)+' , '+str(test_f1))

        # vis.plot_many({"train_ce": train_ce,
        #                "test_ce": test_ce})  # !!!!!!!!!!!!!!!!!
        # vis.plot_many({"train_acc": train_acc,
        #                "test_acc": test_acc})
        # vis.plot_many({"train_f1": train_f1,
        #                "test_f1": test_f1})
        # vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, \
        #          train_ce:{train_ce}, train_acc:{train_acc}, \
        #          test_ce:{test_ce}, test_acc:{test_acc}".format(
        #     epoch=epoch, lr=lr, loss=loss_meter.value(),
        #     train_ce=str(train_ce), train_acc=str(train_acc),
        #     test_ce=str(test_ce), test_acc=str(test_acc)))

        # update learning rate
        if loss_meter.value() > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()
    x1 = max(testf1)
    ind = testf1.index(x1)
    return test_f_a_l[ind], ind

def val(model, dataloader):
    """ Test model accuracy on validation dataset.
    """
    model.eval()
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
        all_score.extend(score.data.numpy().reshape(-1).tolist())
        all_label.extend(label.numpy().reshape(-1).tolist())
    model.train()
    loss = torch.nn.BCELoss()
    ce = loss(Variable(torch.FloatTensor(all_score)), Variable(torch.FloatTensor(all_label)))
    prediction = torch.Tensor(all_score) > 0.5
    # prediction = torch.Tensor(all_score)
    acc = sum(prediction.numpy() == all_label) / (len(all_label) + 0.0)
    F1 = f1_score(all_label, prediction.numpy(), average='binary')
    return ce, acc, F1


if __name__ == '__main__':
    # os.system("python -m visdom.server")  # http://localhost:8097/
    # import fire
    # fire.Fire()
    test1, ind = train()
    print('best test f1: ' + str(test1[0])+'in epoch %d' %ind)
