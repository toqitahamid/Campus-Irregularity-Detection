import mmdet, mmcv
print(mmdet.__version__)
print(mmcv.__version__)

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules

cfg = Config.fromfile("faster-rcnn_r50_fpn_1x_coco.py")

cfg.work_dir = "work_dirs"
runner = Runner.from_cfg(cfg)

runner.train()

runner.test()