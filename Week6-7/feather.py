# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:43:46 2017

@author: daniel
"""

import feather
path = 'my_data.feather'
feather.write_dataframe(df, path)
df = feather.read_dataframe(path)