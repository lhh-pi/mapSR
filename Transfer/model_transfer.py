import torch
from models import mapsr

load_pretrain = torch.load("../save/_swinir_L_x3_train2/epoch-last.pth")
# print(load_pretrain['model']['sd'].keys())
# model:
#   name:
#   args:
#       scale:
#   sd:
# optimizer:
# epoch:
# 经典模型预训练模型
pretrain_net = load_pretrain['model']['sd']
# 改进模型
new_net = mapsr.make_mapsr(scale=3)
model_dict = new_net.state_dict()  # 新模型的k, v
# 更新为预训练权重
state_dict = {k: v for k, v in pretrain_net.items() if k in model_dict.keys()}  # 公共模块权重
model_dict.update(state_dict)  # 更新权重
new_net.load_state_dict(model_dict)  # 载入权重

torch.save(new_net, './mapsr/mapsr_x3.pth')
save_net = {'model': {'name': 'mapsr', 'args': {'scale': 3}, 'sd': model_dict}}
torch.save(save_net, './mapsr/mapsr_x3_pretrain.pth')
