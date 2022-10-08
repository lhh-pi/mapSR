import torch
import os
import collections
from models import swinsrv1

model = torch.load("configs/models/swin_tiny_patch4_window7_224.pth")
# model = torch.load("save/_swinir_x4_train1/epoch-best.pth")
print(model["model"].keys())

model1 = swinsr.make_swinsr()
print(model1.state_dict().keys())

# print(model1['params'].keys())

# keys = list(model.keys())
# model2 = collections.OrderedDict()
# for key in keys:
#     s = str(key)
#     model2[s[7:]] = model[s]
# print(model2.keys())
#
# save_path = 'save/_dbpn_x8_pretrain'
# # #
# sv_file = {'model': {'name': 'dbpn', 'args': {'scale': 8}, 'sd': model2}}
# #
# torch.save(sv_file,
#            os.path.join(save_path, 'dbpn_x8.pth'))


