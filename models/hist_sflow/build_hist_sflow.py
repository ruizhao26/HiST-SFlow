from .core.network import HiST_SFlow

def build_hist_sflow(args):
    model = HiST_SFlow(args)
    return model