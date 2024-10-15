import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import compute_dice, coords_normalize
from DiffusionNet.utils import read_vtk, write_vtk

lossfunc = nn.CrossEntropyLoss()

def train(net, dataset, batch_size, optimizer, epoch, device):
    net.train()
    Loss = 0
    Acc = 0

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

    for _, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
        coords = coords.to(torch.device(device))
        labels = labels.to(torch.device(device))
        mass = mass.to(torch.device(device))
        evals = evals.to(torch.device(device))
        evecs = evecs.to(torch.device(device))
        gradX = gradX.to(torch.device(device))
        gradY = gradY.to(torch.device(device))

        optimizer.zero_grad()
        outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (B,Nv,C_out)
        loss = lossfunc(outputs.transpose(2,1), labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=2)
        acc = torch.sum((preds == labels).sum(dim=1) / labels.shape[1])
        Loss += loss.detach().item()
        Acc += acc.detach().item()
    
    Loss /= dataset.__len__()
    Acc /= dataset.__len__()

    print('train: ce_loss = {:.6f} | acc = {:.2f} %'.format(Loss, 100*Acc))

    return Loss, Acc

def val(net, dataset, batch_size, epoch, device):
    net.eval()
    Loss = 0
    Acc = 0
    Dice = 0

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

    with torch.no_grad():
        for _, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
            coords = coords.to(torch.device(device))
            labels = labels.to(torch.device(device))
            mass = mass.to(torch.device(device))
            evals = evals.to(torch.device(device))
            evecs = evecs.to(torch.device(device))
            gradX = gradX.to(torch.device(device))
            gradY = gradY.to(torch.device(device))

            outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (B,Nv,C_out)
            preds = torch.argmax(outputs, dim=2)
            acc = torch.sum((preds == labels).sum(dim=1) / labels.shape[1])
            loss = lossfunc(outputs.transpose(2,1), labels)

            for b in range(batch_size):
                Dice +=  compute_dice(preds[b], labels[b], net.out_channels)

            Loss += loss.item()
            Acc += acc.item()
    
    Loss /= dataset.__len__()
    Acc /= dataset.__len__()
    Dice /= dataset.__len__()

    print('val: ce_loss = {:.6f} | acc = {:.2f} % | dice = {:.2f} %'.format(Loss, 100*Acc, 100*Dice))

    return Loss, Acc, Dice

def test(net, dataset, epoch, device, source_vtk_root, target_vtk_root):
    net.eval()
    #acc = []
    Dice = []

    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False)

    with torch.no_grad():
        for basename, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
            coords = coords.to(torch.device(device))
            labels = labels.to(torch.device(device))
            mass = mass.to(torch.device(device))
            evals = evals.to(torch.device(device))
            evecs = evecs.to(torch.device(device))
            gradX = gradX.to(torch.device(device))
            gradY = gradY.to(torch.device(device))

            outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (1,Nv,C_out)
            preds = torch.argmax(outputs, dim=2).squeeze(0)  # (Nv)

            source_vtk = read_vtk(os.path.join(source_vtk_root, basename[0] + '.vtk'))
            target_vtk = dict()
            target_vtk['vertices'] = coords_normalize(source_vtk['vertices'])
            target_vtk['faces'] = source_vtk['faces']
            target_vtk['label'] = preds.cpu().numpy()

            write_vtk(target_vtk, 'vertices', os.path.join(target_vtk_root, basename[0] + '.vtk'))
            #dice = compute_dice(preds[0], labels[0], net.out_channels)
            #Dice.append(dice)
            #with open(os.path.join(checkpoint, "subject_logs.txt"), 'a+') as f:
            #    print(basename[0] + ' : dice = {:.2f} %'.format(100*np.mean(dice)),file=f)
            
            #acc.append(torch.sum((preds == labels).sum(dim=1) / labels.shape[1]).item())
            #dice.append(compute_dice(preds[0], labels[0], net.out_channels))

    #return Dice
