import os
from omegaconf import OmegaConf


cfg = OmegaConf.create()
cfg.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cfg.datasets_dir = os.path.join(cfg.root_dir, "datasets")  # default directory for loading datasets
cfg.pretrained_checkpoints_dir = os.path.join(cfg.root_dir, "pretrained_checkpoints")  # default directory for saving and loading pretrained checkpoints
#cfg.pretrained_checkpoints_dir = os.path.join(cfg.root_dir, "experiments","tmp_sevir_ftsno_w_gn_cord_64","checkpoints")
cfg.exps_dir = os.path.join(cfg.root_dir, "experiments")  # default directory for saving experiment results
os.makedirs(cfg.exps_dir, exist_ok=True)
