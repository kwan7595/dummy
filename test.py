import os
os.environ['PT_HPU_LAZY_MODE'] = '1'
os.environ['LOG_LEVEL_PT_FALLBACK'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '1'
os.environ['LOG_LEVEL_ALL'] = '3'
os.environ['ENABLE_CONSOLE'] = 'true'
import habana_frameworks.torch.core as htcore
import torch
with torch.no_grad():
        conv = torch.nn.Conv2d(3,64,(32,32)).to('hpu')
        test_conv = torch.nn.Conv2d(3,64,(32,32))
        test_conv.weight.data = conv.weight.data
output
