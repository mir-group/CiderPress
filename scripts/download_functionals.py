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

import os
import subprocess

basedir = __file__
savedir = os.path.basename(os.path.join(basedir, "../functionals"))

os.makedirs(savedir, exist_ok=True)
links = {
    "CIDER23X_NL_GGA": "16IF81oS3Snqp96KdS43orSmpmNyRKFEs",
    "CIDER23X_NL_MGGA_DTR": "1njP58rnZL0L523hEiwXdQ9YoBTlp0Uev",
    "CIDER23X_NL_MGGA_PBE": "1sVofPzXu4eN2N0cDSC0wawnvNjqCoXhi",
    "CIDER23X_NL_MGGA": "1Y_pGWGnuzH7PiTT1UntnCga4NPIXBh0m",
    "CIDER23X_SL_GGA": "1laNkInAIq1EfHACfOf5hUpqL7eWFpoA1",
    "CIDER23X_SL_MGGA": "1kmYuG50OhTCNB8QVIEHLm-cEQ8mMjwT8",
    "CIDER24Xe": "1tFKAMvfYJcUZEz11uFSl-DXGW-47l9SV",
    "CIDER24Xne": "1OUcWv18e2vM0eSDyiSl1dm9g3A7Wu5_3",
}

os.chdir(savedir)
cmd = "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O {}"
for name, linkid in links.items():
    fname = "{}.yaml".format(name)
    mycmd = cmd.format(linkid, fname)
    subprocess.call(mycmd, shell=True)
