from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import warnings
warnings.filterwarnings('ignore')

#############################################################################################
######################## Cross-entropy losses and train functions ###########################

def CE_loss(preds, labels, device, args, criterion):
    prob = F.softmax(preds, dim=1)

    loss_all = criterion(preds, labels)
    loss = torch.mean(loss_all)
    return prob, loss, loss_all

def mixup_criterion(pred, y_a, y_b, lam, criterion):
    prob = F.softmax(pred, dim=1)
    return prob, lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def ricap_data_original(args, input, targets, train_loader, index, epoch, device):
    '''RICAP DA'''
    I_x, I_y = input.size()[2:]

    w = int(np.round(I_x * np.random.beta(args.alpha, args.alpha)))
    h = int(np.round(I_y * np.random.beta(args.alpha, args.alpha)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    cropped_images = {}
    c_ = {}
    W_ = {}
    i_ = {}

    for k in range(4):
        idx = torch.randperm(input.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[idx]
        W_[k] = w_[k] * h_[k] / (I_x * I_y)
        i_[k] = index[idx]

    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
        torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
    patched_images = patched_images.to(device)

    return patched_images, c_, W_, i_


def ricap_criterion(criterion, pred, c_, W_):
    prob = F.softmax(pred, dim=1)
    l_1 = criterion(pred, c_[0])
    l_2 = criterion(pred, c_[1])
    l_3 = criterion(pred, c_[2])
    l_4 = criterion(pred, c_[3])
    loss_all = W_[0]*l_1 + W_[1]*l_2 + W_[2]*l_3 + W_[3]*l_4
    loss = torch.mean(loss_all)
    return prob, loss, loss_all, l_1, l_2, l_3, l_4


#############################################################################################
########################            Training function             ###########################

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, lemniscate = 0, criterion = 0):
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss(reduction = 'none')

    # switch to train mode
    model.train()

    # ricap variables
    l_1 = 0
    i_ = 0
    c_ = 0

    counter = 1
    for texts, labels, index in train_loader:

        torch.cuda.empty_cache() 
        texts, labels, index = texts.to(device), labels.to(device), index.to(device)
        outputs = model(texts)

        prob, loss, loss_all = CE_loss(outputs, labels, device, args, criterion)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 2])
        train_loss.update(loss.item(), texts.size(0))
        top1.update(prec1.item(), texts.size(0))
        top5.update(prec5.item(), texts.size(0))

        if not args.method == "SGD":
            update_sampling_metrics(args, train_loader, loss_all, prob, index, labels, l_1)

        num_samples =  len(train_loader.sampler)

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(texts), num_samples, 100. * counter / len(train_loader), loss.item(),
                prec1, optimizer.param_groups[0]['lr']))

        counter = counter + 1

    return train_loss.avg, top5.avg, top1.avg

def update_sampling_metrics(args, train_loader, loss_all, prob, index, labels, l_1):

    index = index.cpu()


    loss_all = loss_all.cpu().detach().numpy()
    labels = labels.cpu()
    prob = prob.cpu().detach().numpy()

    count = train_loader.dataset.times_seen[index].copy()
    if count.max() < 1:
        count = np.zeros(len(count))

    # Updating probs:
    avg_probs = train_loader.dataset.avg_probs[index].copy()
    avg_probs[avg_probs == -1] = 0.0
    accumulated_probs = count*avg_probs
    avg_probs = (accumulated_probs + (1-prob[range(len(labels)), labels]))/(count+1)
    train_loader.dataset.avg_probs[index] = avg_probs

    times_seen = train_loader.dataset.times_seen
    times_seen[index] += 1
    times_seen[times_seen == 1+1e-6] = 1


###############################################################################
################################ Testing #####################################

def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target, *_) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


##############################################################################
##################### Selecting data and preparing loaders ###################

def select_samples(args, train_loader, epoch):
    '''Gives back the indexes that correspond to the samples to be used for training'''
    if args.method == "unif-SGD":
        curr_prob = np.ones(len(train_loader.dataset.labels))
    elif args.method == "p-SGD":
        # probs_dummy = list(train_loader.dataset.avg_probs)
        # print(probs_dummy)
        curr_prob = train_loader.dataset.avg_probs.copy()
        print(curr_prob)
        # This is for the initial epochs, to force the moded to see all the samples:
        if curr_prob.max() == -1:
            curr_prob *= -1
        max_prob = curr_prob.max()
        curr_prob[curr_prob==-1] = max_prob
    elif args.method == "c-SGD":
        curr_prob = train_loader.dataset.avg_probs.copy()
        # This is for the initial epochs, to force the moded to see all the samples:
        if curr_prob.max() == -1:
            curr_prob *= -1
        max_prob = curr_prob.max()
        curr_prob[curr_prob==-1] = max_prob
        # Use the confusion instaed of the probability:
        curr_prob = curr_prob * (1 - curr_prob)
    
    # Random sampling warmup for baselines without budget restrictions
    if epoch < args.c_sgd_warmup:
        len_curr = len(curr_prob)
        curr_prob = np.ones(len_curr)

    # Smoothness constant
    c = curr_prob.mean()
    curr_prob = curr_prob + c

    # Probability normalization
    y = curr_prob
    if y.sum() == 0:
        y = y+1e-10
    curr_prob = (y)/(y).sum()

    # Select the samples to be used:
    samples_to_keep = int(len(curr_prob)*args.budget)
    try:
        curr_samples_idx = np.random.choice(len(curr_prob), (samples_to_keep), p = curr_prob, replace = False)
    except:
        curr_prob[curr_prob == 0] = 1e-10
        curr_samples_idx = np.random.choice(len(curr_prob), (samples_to_keep), p = curr_prob/curr_prob.sum(), replace = False)

    return curr_samples_idx


def prepare_loader(args, train_loader, epoch):
    '''Prepares the dataset with the samples to be used in the following epochs'''
    curr_samples_idx = select_samples(args, train_loader, epoch)

    dataset_sampler = torch.utils.data.SubsetRandomSampler(curr_samples_idx)
    train_loader.dataset.train_samples_idx = curr_samples_idx
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, sampler=dataset_sampler, \
                                            batch_size=args.batch_size, \
                                            pin_memory=True, \
                                            drop_last = True)

    return train_loader


##############################################################################
################################ Other functions ##############################

def linearLR_per_it(args, optimizer, iteration,  max_count):
    """Sets the learning rate"""
    lr = np.linspace(args.lr, 1e-6, args.epoch*max_count)
    try:
        lr = lr[iteration]
    except:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def test_sb(loader, epoch, sb, cnn):
    # Testing function when using selective backpropagation
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
            loss = nn.CrossEntropyLoss()(pred, labels)
            test_loss += loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    # test_loss /= total
    test_loss /= len(loader)
    val_acc = correct / total

    print('============ EPOCH {} ============'.format(epoch))
    print('FPs: {} / {}\nBPs: {} / {}\nTest loss: {:.6f}\nTest acc: {:.3f}'.format(
                sb.logger.global_num_forwards,
                sb.logger.global_num_skipped_fp + sb.logger.global_num_forwards,
                sb.logger.global_num_backpropped,
                sb.logger.global_num_skipped + sb.logger.global_num_backpropped,
                test_loss,
                100.*val_acc))
    cnn.train()
    return [100. * val_acc], [test_loss]
