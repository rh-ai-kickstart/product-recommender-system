from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from recommendation_core.models.item_tower import ItemTower
from recommendation_core.models.user_tower import UserTower


class TwoTowerModel(nn.Module):
    def __init__(self, item_tower: ItemTower, user_tower: UserTower):
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower

    def forward(self, items_dict: Dict[str, Tensor], users_dict: Dict[str, Tensor]):
        items_embed = self.item_tower(**items_dict)  # shape -> bs, dim
        users_embed = self.user_tower(**users_dict)  # shape -> bs, dim

        return torch.norm(items_embed - users_embed, dim=-1)  # shape -> bs
