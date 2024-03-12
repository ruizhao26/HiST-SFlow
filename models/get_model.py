from .hist_sflow.build_hist_sflow import build_hist_sflow

def get_model(args):
    if args.arch.upper() == 'hist_sflow'.upper():
        model = build_hist_sflow(args)

    return model