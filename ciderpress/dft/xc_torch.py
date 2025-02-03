#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

import numpy as np
import torch
import torch.nn as nn


class XNet(nn.Module):
    def __init__(self, ninp, nlayer=5, nhidden=30):
        super(XNet, self).__init__()
        n = nhidden
        self.l0 = nn.Linear(ninp, n)
        self.lls = nn.ModuleList([nn.Linear(n, n) for i in range(nlayer)])
        self.l2s = nn.ModuleList([nn.Linear(ninp, n) for i in range(nlayer)])
        self.last = nn.Linear(n, 1)
        self.af = nn.Tanh()

    def forward(self, x):
        x0 = x
        x = self.l0(x)
        for ll, l2 in zip(self.lls, self.l2s):
            x = l2(x0) * x + ll(x)
            x = self.af(x)
        return torch.squeeze(self.last(x))


class XNet2(nn.Module):
    def __init__(self, ninp, nlayer=5, nhidden=30):
        super(XNet2, self).__init__()
        n = nhidden
        self.l0 = nn.Linear(ninp, n)
        self.lls = nn.ModuleList([nn.Linear(n, n) for i in range(nlayer)])
        self.l2s = nn.ModuleList([nn.Linear(ninp, n) for i in range(nlayer)])
        self.last = nn.Linear(n, 1)
        self.af = nn.Tanh()

    def forward(self, x):
        x0 = x
        x = self.l0(x)
        for ll, l2 in zip(self.lls, self.l2s):
            x = l2(x0) + ll(x) + x
            x = self.af(x)
        return torch.squeeze(self.last(x))


class XNet3(nn.Module):
    def __init__(self, ninp, nlayer=5, nhidden=30):
        super(XNet3, self).__init__()
        n = nhidden
        self.l0 = nn.Linear(ninp, n)
        self.lls = nn.ModuleList([nn.Linear(n, n) for i in range(nlayer)])
        self.l2s = nn.ModuleList([nn.Linear(ninp, n) for i in range(nlayer)])
        self.last = nn.Linear(n, 1)
        self.af = nn.Tanh()

    def forward(self, x):
        x0 = x
        x = self.l0(x)
        for ll, l2 in zip(self.lls, self.l2s):
            x = l2(x0) + ll(x)
            x = self.af(x)
        return torch.squeeze(self.last(x))


def simple_map(
    kernel, use_grad=True, nlayer=5, nhidden=30, nstep=3000, nncls=XNet, lr=1.0
):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # gp = joblib.load(fname)
    # X = gp.kernels[0].X1ctrl
    X = kernel.X1ctrl
    alpha = kernel.alpha
    # alpha = gp.kernels[0].alpha
    # k = gp.kernels[0].kernel(X, X)
    k = kernel.kernel(X, X)
    y = alpha.dot(k)
    Xtr = X.copy()

    def eval_gp(Xtst):
        k = kernel.kernel(Xtr, Xtst)
        return alpha.dot(k)

    my_nn = nncls(X.shape[1], nlayer=nlayer, nhidden=nhidden)
    my_nn = my_nn.double().to(device)

    loss = nn.MSELoss()
    # optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-2)
    optimizer = torch.optim.LBFGS(my_nn.parameters(), lr=lr)

    def train(X, y, model, loss_fn, optimizer, t, need_closure=False):
        size = len(X)
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        def closure_in():
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss

        if need_closure:
            optimizer.step(closure_in)
        else:
            optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        if t % 10 == 0:
            print(f"loss: {loss:>7f}  [{t:>5d}/{size:>5d}]")

    def train_grad(
        X, y, model, loss_fn, optimizer, t, delta, nfeat, nsamp, need_closure=False
    ):
        size = len(X)
        model.train()
        pred = model(X)
        fac = 1.0 / nfeat
        loss = loss_fn(pred[:nsamp], y[:nsamp])
        for i in range(nfeat):
            deriv = (
                pred[(2 * i + 1) * nsamp : (2 * i + 2) * nsamp]
                - pred[(2 * i + 2) * nsamp : (2 * i + 3) * nsamp]
            ) / delta
            loss = loss + fac * loss_fn(deriv, y[(i + 1) * nsamp : (i + 2) * nsamp])
        loss.backward()

        def closure_in():
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred[:nsamp], y[:nsamp])
            for i in range(nfeat):
                deriv = (
                    pred[(2 * i + 1) * nsamp : (2 * i + 2) * nsamp]
                    - pred[(2 * i + 2) * nsamp : (2 * i + 3) * nsamp]
                ) / delta
                loss = loss + fac * loss_fn(deriv, y[(i + 1) * nsamp : (i + 2) * nsamp])
            loss.backward()
            return loss

        if need_closure:
            optimizer.step(closure_in)
        else:
            optimizer.step()
        optimizer.zero_grad()
        loss = loss.item(), 0
        if t % 10 == 0:
            print(f"loss: {loss:>7f}  [{t:>5d}/{size:>5d}]")

    delta = 1e-4
    nsamp = X.shape[0]
    Xlst = [X]
    ylst = [y]
    for i in range(X.shape[-1]):
        Xp = X.copy()
        Xp[:, i] += 0.5 * delta
        Xm = X.copy()
        Xm[:, i] -= 0.5 * delta
        Xlst.append(Xp)
        Xlst.append(Xm)
        dy = (eval_gp(Xp) - eval_gp(Xm)) / delta
        ylst.append(dy)
    X = np.concatenate(Xlst, axis=0)
    y = np.concatenate(ylst)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    for t in range(nstep):
        if use_grad:
            train_grad(
                X,
                y,
                my_nn,
                loss,
                optimizer,
                t,
                delta,
                X.shape[1],
                nsamp,
                need_closure=True,
            )
        else:
            train(X, y, my_nn, loss, optimizer, t, need_closure=True)
    my_nn.eval()
    for i in range(100):
        pred = my_nn(X[:nsamp])
    diff = (y[:nsamp] - pred).cpu().detach().numpy()
    print(
        "Err", np.mean(np.abs(diff)), np.sqrt(np.mean(diff**2)), np.max(np.abs(diff))
    )
    return my_nn.to(torch.device("cpu"))
