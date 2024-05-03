import os
# import gdown
import torch
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

# OMNIDATA_NORMALS_WEIGHTS_URL = "https://drive.google.com/uc?id=1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR"
OMNIDATA_NORMALS_WEIGHTS_URL = "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1"

# OMNIDATA_NORMALS_WEIGHTS_PATH = "/tmp/omnidata_surface_normal_models/"
OMNIDATA_NORMALS_WEIGHTS_DIR = torch.hub.get_dir() + '/omnidata/'
OMNIDATA_NORMALS_WEIGHTS_PATH = OMNIDATA_NORMALS_WEIGHTS_DIR + '/omnidata_dpt_normal_v2.ckpt'

def load_omni_model():
    """Download omnidata normals weights from Google Drive and load omnidata_tools model DPTDepthModel.

    returns:
        model (omnidata_tools.torch.modules.midas.dpt_depth.DPTDepthModel): the model with weights loaded
    """
    # download weights
    if not os.path.exists(OMNIDATA_NORMALS_WEIGHTS_PATH):
        os.makedirs(OMNIDATA_NORMALS_WEIGHTS_DIR, exist_ok=True)
        os.system(f'wget {OMNIDATA_NORMALS_WEIGHTS_URL} -O {OMNIDATA_NORMALS_WEIGHTS_PATH}')

        # gdown.download(url=OMNIDATA_NORMALS_WEIGHTS_URL, output=OMNIDATA_NORMALS_WEIGHTS_DIR)

    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(OMNIDATA_NORMALS_WEIGHTS_PATH, map_location='cuda')

    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to('cuda')

    return model

