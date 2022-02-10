import pickle
import subprocess
from typing import Any, Dict, List, Union, Tuple

import torch
import torch.nn as nn


def dic_to_device(dic: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value

    return dic


def set_cuda_visible_device(num_gpus: int, max_num_gpus: int = 16) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding="utf-8")
            out = proc.communicate()

            if "No devices were found" in out[0]:
                break

            if "No running" in out[0]:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


def initialize_model(
    model: nn.Module, device: torch.device, load_save_file: bool = False
) -> nn.Module:
    if load_save_file:
        if device.type == "cpu":
            model.load_state_dict(
                torch.load(load_save_file, map_location="cpu"), strict=False
            )
        else:
            model.load_state_dict(torch.load(load_save_file), strict=False)
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    return model


def read_data(
    filename: str, key_dir: str, train: bool = True
) -> Tuple[Union[List[str], Dict[str, float]]]:
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        id_to_y = {l[0]: float(l[1]) for l in lines}
    with open(f"{key_dir}/test_keys.pkl", "rb") as f:
        test_keys = pickle.load(f)
    if train:
        with open(f"{key_dir}/train_keys.pkl", "rb") as f:
            train_keys = pickle.load(f)
        return train_keys, test_keys, id_to_y
    else:
        return test_keys, id_to_y


def write_result(
    filename: str, pred: Dict[str, List[float]], true: Dict[str, List[float]]
) -> None:
    with open(filename, "w") as w:
        for k in pred.keys():
            w.write(f"{k}\t{true[k]:.3f}\t")
            w.write(f"{pred[k].sum():.3f}\t")
            w.write(f"{0.0}\t")
            for j in range(pred[k].shape[0]):
                w.write(f"{pred[k][j]:.3f}\t")
            w.write("\n")
    return
