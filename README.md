## This is a forked version of the Omnidata repository
#### to install
* create a python3 virtualenv (`python3 -m venv venv`) and start it (`source venv/bin/activate`)
* to use methods and classes from "omnidata_tools" (https://github.com/CCareaga/omnidata), run `pip install https://github.com/CCareaga/omnidata/archive/main.zip`

#### to use this in code:
We created a script in omnidata_tools called "model_util" which loads `DPTDepthModel`. To use this function in a python script, simply:
* install the omnidata_tools package
* in a python script, `from omnidata_tools.model_util import load_omni_model`
* in the python script, call load_omni_model: `model = load_omni_model()`. The function downloads the omnidata normals weights from Google Drive, then loads them into `DPTDepthModel` with backbone "vitb_rn50_384".

