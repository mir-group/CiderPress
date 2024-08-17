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

os.chdir(savedir)
cmd = "wget 'https://zenodo.org/api/records/13336814/files-archive' -O cider_functionals.zip"
unzip = "unzip cider_functionals.zip"
rm = "rm -r cider_functionals.zip"
subprocess.call(cmd, shell=True)
subprocess.call(unzip, shell=True)
subprocess.call(rm, shell=True)
