import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.metric import *
from src.omt import InverseOMT
from src.utils import init_record


# =============================== Single training ===============================
def train_step(ep, model, train_loader, optimizer, criterion, act, size, device, brain_class):
    model.train()
    record = init_record()
    train_bar = tqdm(train_loader, desc=f'Training {ep}')
    for data in train_bar:
        optimizer.zero_grad()
        image = data['image'][:,brain_class,:,:,:].to(device)
        label = data['label'].to(device)
        pred = model(image)
        loss = criterion(pred, label)
        acc = compute_acc(pred, label, act)

        loss.backward()
        optimizer.step()

        batch_size = image.shape[0]
        record['CLoss'].update(loss.item(), batch_size)
        record['CAcc'].update(acc.item(), batch_size)

        del image, label, pred
        train_bar.set_postfix({
            'CLoss': f"{record['CLoss'].avg:.5f}",
            'CAcc': f"{record['CAcc'].avg:.5f}"
        })

    train_bar.close()
    train_record = {
        'CLoss': f"{record['CLoss'].avg:.5f}",
        'CAcc': f"{record['CAcc'].avg:.5f}",
        'BLoss': np.nan,
        'BAcc': np.nan,
    }

    return train_record


def val_step(ep, model, val_loader, optimizer, criterion, act, size, device, brain_class):
    model.eval()
    record = init_record()
    val_bar = tqdm(val_loader, desc=f'Validaion {ep}')
    with torch.no_grad():
        for index, data in enumerate(val_bar):
            brain_id = data['id'][0]
            image = data['image'][:,brain_class,:,:,:].to(device)
            label = data['label'].to(device)
            rawimg = data['rawimg'].to(device)
            rawlab = data['rawlab'].to(device)
            idx = data['Idx'].to(device)
            inv_idx = data['InvIdx'].to(device)
            pred = model(image)
            loss = criterion(pred, label)
            acc = compute_acc(pred, label, act)

            # convert cube to brain
            brain_pred = InverseOMT(pred, rawimg, idx, inv_idx, device)
            brain_loss = criterion(brain_pred, rawlab)
            brain_acc = compute_acc(brain_pred, rawlab, act)

            record['CLoss'].update(loss.item())
            record['CAcc'].update(acc.item())
            record['BLoss'].update(brain_loss.item())
            record['BAcc'].update(brain_acc.item())

            del image, label, pred, rawimg, rawlab, idx, inv_idx
            val_bar.set_postfix({
                'CAcc': f"{record['CAcc'].avg:.5f}",
                'BAcc': f"{record['BAcc'].avg:.5f}",
            })

    val_bar.close()
    val_record = {
        'CLoss': f"{record['CLoss'].avg:.5f}",
        'CAcc': f"{record['CAcc'].avg:.5f}",
        'BLoss': f"{record['BLoss'].avg:.5f}",
        'BAcc': f"{record['BAcc'].avg:.5f}"
    }

    return val_record


# =============================== Multiple training ===============================
def multi_train_step(ep, model, train_loader, optimizer, criterion, act, size, device, brain_class):
    model.train()
    record = init_record(mode='multiple')
    train_bar = tqdm(train_loader, desc=f'Training {ep}')
    for data in train_bar:
        optimizer.zero_grad()
        image = data['image'][:, brain_class, :, :, :].to(device)
        label = data['label'].to(device)

        if act == 'softmax':
            wt_pred, tc_pred, et_pred = model(image)

        else:
            pred = model(image)
            wt_pred, tc_pred, et_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1), pred[:, 2].unsqueeze(1)

        wt_label = label[:, 0].unsqueeze(1)
        tc_label = label[:, 1].unsqueeze(1)
        et_label = label[:, 2].unsqueeze(1)

        wt_loss = criterion(wt_pred, wt_label)
        tc_loss = criterion(tc_pred, tc_label)
        et_loss = criterion(et_pred, et_label)
        loss = wt_loss + tc_loss + et_loss

        wt_acc = compute_acc(wt_pred, wt_label, act)
        tc_acc = compute_acc(tc_pred, tc_label, act)
        et_acc = compute_acc(et_pred, et_label, act)

        loss.backward()
        optimizer.step()

        batch_size = image.shape[0]
        record['CLoss'].update(loss, batch_size)
        record['wt_CAcc'].update(wt_acc.item(), batch_size)
        record['tc_CAcc'].update(tc_acc.item(), batch_size)
        record['et_CAcc'].update(et_acc.item(), batch_size)

        del image, label, wt_pred, tc_pred, et_pred
        train_bar.set_postfix({
            'CLoss': f"{record['CLoss'].avg:.5f}",
            'wt_CAcc': f"{record['wt_CAcc'].avg:.5f}",
            'tc_CAcc': f"{record['tc_CAcc'].avg:.5f}",
            'et_CAcc': f"{record['et_CAcc'].avg:.5f}"
        })

    train_bar.close()
    train_record = {
        'CLoss': f"{record['CLoss'].avg:.5f}",
        'wt_CAcc': f"{record['wt_CAcc'].avg:.5f}",
        'tc_CAcc': f"{record['tc_CAcc'].avg:.5f}",
        'et_CAcc': f"{record['et_CAcc'].avg:.5f}",
        'wt_BAcc': np.nan,
        'tc_BAcc': np.nan,
        'et_BAcc': np.nan
    }

    return train_record


def multi_val_step(ep, model, val_loader, optimizer, criterion, act, size, device, brain_class):
    model.eval()
    record = init_record(mode='multiple')
    val_bar = tqdm(val_loader, desc=f'Validaion {ep}')
    with torch.no_grad():
        for index, data in enumerate(val_bar):
            brain_id = data['id'][0]
            image = data['image'][:, brain_class, :, :, :].to(device)
            label = data['label'].to(device)
            rawimg = data['rawimg'].to(device)
            rawlab = data['rawlab'].to(device)
            idx = data['Idx'].to(device)
            inv_idx = data['InvIdx'].to(device)

            if act == 'softmax':
                wt_pred, tc_pred, et_pred = model(image)

            else:
                pred = model(image)
                wt_pred, tc_pred, et_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1), pred[:, 2].unsqueeze(1)

            wt_label = label[:, 0].unsqueeze(1)
            tc_label = label[:, 1].unsqueeze(1)
            et_label = label[:, 2].unsqueeze(1)

            wt_rawlab = rawlab[:, 0].unsqueeze(1)
            tc_rawlab = rawlab[:, 1].unsqueeze(1)
            et_rawlab = rawlab[:, 2].unsqueeze(1)

            wt_loss = criterion(wt_pred, wt_label)
            tc_loss = criterion(tc_pred, tc_label)
            et_loss = criterion(et_pred, et_label)
            loss = wt_loss + tc_loss + et_loss

            wt_acc = compute_acc(wt_pred, wt_label, act)
            tc_acc = compute_acc(tc_pred, tc_label, act)
            et_acc = compute_acc(et_pred, et_label, act)

            # convert cube to brain
            wt_brain_pred = InverseOMT(wt_pred, rawimg, idx, inv_idx, device)
            tc_brain_pred = InverseOMT(tc_pred, rawimg, idx, inv_idx, device)
            et_brain_pred = InverseOMT(et_pred, rawimg, idx, inv_idx, device)

            wt_brain_acc = compute_acc(wt_brain_pred, wt_rawlab, act)
            tc_brain_acc = compute_acc(tc_brain_pred, tc_rawlab, act)
            et_brain_acc = compute_acc(et_brain_pred, et_rawlab, act)

            record['CLoss'].update(loss.item())
            record['wt_CAcc'].update(wt_acc.item())
            record['tc_CAcc'].update(tc_acc.item())
            record['et_CAcc'].update(et_acc.item())
            record['wt_BAcc'].update(wt_brain_acc.item())
            record['tc_BAcc'].update(tc_brain_acc.item())
            record['et_BAcc'].update(et_brain_acc.item())

            del image, label, wt_pred, tc_pred, et_pred, rawimg, rawlab, idx, inv_idx
            val_bar.set_postfix({
                'wt_BAcc': f"{record['wt_BAcc'].avg:.5f}",
                'tc_BAcc': f"{record['tc_BAcc'].avg:.5f}",
                'et_BAcc': f"{record['et_BAcc'].avg:.5f}"
            })

    val_bar.close()
    val_record = {
        'CLoss': f"{record['CLoss'].avg:.5f}",
        'wt_CAcc': f"{record['wt_CAcc'].avg:.5f}",
        'tc_CAcc': f"{record['tc_CAcc'].avg:.5f}",
        'et_CAcc': f"{record['et_CAcc'].avg:.5f}",
        'wt_BAcc': f"{record['wt_BAcc'].avg:.5f}",
        'tc_BAcc': f"{record['tc_BAcc'].avg:.5f}",
        'et_BAcc': f"{record['et_BAcc'].avg:.5f}"
    }

    return val_record
