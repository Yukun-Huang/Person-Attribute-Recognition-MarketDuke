from .models import Backbone_nFC, Backbone_nFC_Id


def get_model(model_name, num_label, use_id=False, num_id=None):
    if not use_id:
        return Backbone_nFC(num_label, model_name)
    else:
        return Backbone_nFC_Id(num_label, num_id, model_name)

