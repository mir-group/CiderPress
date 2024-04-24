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
    "NL_GGA": "1UNxxvYpxc8RkywxIynbenkLEMGviIce1",
    "NL_MGGA_DTR": "1-233LBivti-UDfbSuOMdPZuyC_Gibto0",
    "NL_MGGA_PBE": "15x8s_Pf3_iYoAHl_sWChB2q1HFif1jFA",
    "NL_MGGA": "1wwYFSJsZX_jhBSlsXVeggQutODXC2KP0",
    "SL_GGA": "1qD_IpXQNYUbT8Y7nLt4e4pA1E1dNntu7",
    "SL_MGGA": "18OFNglYpcPXHzXwFcRXeJn9VCLFtC7Zd",
}

os.chdir(savedir)
cmd = "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O {}"
for name, linkid in links.items():
    fname = "CIDER23_{}.yaml".format(name)
    mycmd = cmd.format(linkid, fname)
    subprocess.call(mycmd, shell=True)
