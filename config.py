dense = {
    "act": 'selu',
    "layers": [
        [128, 0.2],
        [64, 0.2],
        [32, 0.3]
    ],
}

seriesnet = {
    "filters": 12,
    "d_rate": 3,  # use 4 if seq_len is 24
    "kernel_size": 2,
    "dropout": 0.8
}

seq2seq = {
    "encoder": 32,  # 64 before
    "decoder": 32,
    "alpha": 1e-6,
    "beta": 1e-6,
    "attn": False
}

darnn = {
    "encoder": 64,  # 64 before
    "decoder": 64,
    "beta": 1e-6,
    "alpha": 1e-6,
    "attn": True
}

rnn = {
    "encoder": 64,
    "alpha": 1e-6,  # 1e-6,
    "beta": 1e-6,  # 1e-6
    "attn":True,
}

wavenet = {
    'encoder': 128,
    'layers': 3,
    'res': 32,
    'skip': 32,
    'd_rate': 3
}

np = {
    'encoder': 32,
    'decoder': 32,
    'latent': 4,
    'samples': 20,
    'embedding': 12,
    'bias': .8,
    'h_dim': 3  # size of contraction in the aggregation step
}

parserconfig = {
    "model": "rnn",
    "path": "tf",
    "loss": "smape",
    "mode": "train",
    "restore_path": None,
    "note": None,
    "steps": int(1e5),
    "seed": 200,
    "batch_size": 8,
    "keep_prob": .8,
    "clip": 20.,
    "lr": 1e-3,
    "decay_rate": 0.7,
    "random_start": True
}

trainconfig = {
    "seq_len": 12,
    "pred_len": 6,
    "test_len": 6,
    "ar": False,
    "use_x": False,
    "decay_steps": 1e10,
    "preprocess": "normalize",
    "summarize_every": 1,  # in epochs
    "test_every": 50,
    "save_every": 50,
    "report_every": 50,
    "test_mode": "fixed",  # or daily
    "lags": (12,),
    "dataset": "sml"
}
