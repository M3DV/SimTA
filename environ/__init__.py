import os
import json

from easydict import EasyDict


def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "File {} doesn't exist.".format(cfg_path)
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ

ENVIRON_PATH = EasyDict(parse_environ())
