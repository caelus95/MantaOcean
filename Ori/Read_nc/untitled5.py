#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:10:56 2021

@author: caelus
"""

import pandas as pd
import numpy as np

data = pd.read_csv('/home/caelus/wormhole/mix_elnino.csv')


a,b = data.shape

ep_index = data.values.reshape(-1)

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/latest/sigs/mix_index',ep_index)


