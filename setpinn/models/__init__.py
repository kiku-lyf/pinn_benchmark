from .fls import FLS
from .pinn import PINNs
from .pinnsformer import PINNsformer
from .pinnsformer_enc import PINNSFormer_Enc
from .qres import QRes
from .setpinns import SetPinns
from .tf_enc import Transformer
from .kan import KANN
from .pinn_gpt import PINNGPT

__all__ = [
    "get_model",
    "Transformer",
    "KANN",
    "SetPinns",
    "QRes",
    "PINNSFormer_Enc",
    "PINNsformer",
    "PINNs",
    "FLS",
]


def get_model(exp_name):
    model_name = exp_name.split("-")[0]
    if model_name == "fls":
        model = FLS(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)
    elif model_name == "pinns":
        model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)
    elif model_name == "pinnsformer":
        model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2)
    elif model_name == "pinnsformer_enc":
        model = PINNSFormer_Enc(d_out=1, d_hidden=512, d_model=32, N=1, heads=2)
    elif model_name == "qres":
        model = QRes(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4)
    elif model_name == "setpinns":
        model = SetPinns(d_out=1, d_hidden=512, d_model=32, N=1, heads=2)
    elif model_name == "kan":
        model = KANN(
            width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25
        )
    elif model_name == "pinngpt":
        model = PINNGPT()
    else:
        raise NotImplementedError
    return model
