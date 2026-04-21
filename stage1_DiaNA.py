import os
import sys
import random
import argparse

from diana_utils import calculate_source_centroids, static_separate_subsets
import numpy as np
from PIL import ImageFile

from source.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.converter import AttnLabelConverter, CTCLabelConverter
import utils.utils_HDGE as utils
from utils.load_config import load_config

from modules.discriminators import define_Dis

import source.HDGE as md
from source.stratify import DomainStratifying
from source.dataset import get_dataloader, hierarchical_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
	# ---- Static DiaNA separation (UDA) ----
	dashed_line = "-" * 80
	# k_top = getattr(args, "k_feat", 32)
	# conf_thr = getattr(args, "sele_conf_thred", 0.95)

	# Load pre-trained model
	""" Model configuration """
	if args.Prediction == "CTC":
		converter = CTCLabelConverter(args.character)
	else:
		converter = AttnLabelConverter(args.character)
		args.sos_token_index = converter.dict["[SOS]"]
		args.eos_token_index = converter.dict["[EOS]"]
	args.num_class = len(converter.character)

	print(dashed_line)
	print("Init model")
	print("Using device:", device)
	model = Model(args).to(device)

	# load pretrained model
	try:
		# pretrained = torch.load(args.saved_model)
		# model.load_state_dict(pretrained)
		pretrained = torch.load(args.saved_model, map_location=device)
		missing, unexpected = model.load_state_dict(pretrained, strict=False)

		print("Missing keys:", missing)
		print("Unexpected keys:", unexpected)
	except:
		raise ValueError("The pre-trained weights do not match the model! Carefully check!")

	## Load source data
	print("Load source domain data ...")
	source_data, source_data_log = hierarchical_dataset(args.source_data, args)
	source_loader = get_dataloader(args, source_data, args.batch_size, shuffle=True)
	print(source_data_log, end="")

	# Load target data
	print("Load target domain data ...")
	target_data, target_data_log = hierarchical_dataset(args.target_data, args, mode = "raw")
	target_loader = get_dataloader(args, target_data, args.batch_size, shuffle=True, aug=True)
	print(target_data_log, end="")
	print(dashed_line)

	# Compute source centroids
	source_centroids = calculate_source_centroids(model, source_loader, device, converter)

	# Compute target centroids
	subsets, U, D = static_separate_subsets(model, target_loader, source_centroids, device)

	cc_idx = []
	uc_idx = []
	ui_idx = []
	ci_idx = []
		
	# Iterate through the returned 'subsets' list with their index (idx)
	for idx, label in enumerate(subsets):
		if label == "cc":
			cc_idx.append(idx)
		elif label == "uc":
			uc_idx.append(idx)
		elif label == "ui":
			ui_idx.append(idx)
		elif label == "ci":
			ci_idx.append(idx)

	# Persist subsets to use in Stage 2 (progressive training)
	save_root = f"stratify/{args.num_subsets}_subsets"
	os.makedirs(save_root, exist_ok=True)

	# Save the index lists
	np.save(f"{save_root}/subset_cc_idx.npy", np.array(cc_idx, dtype=np.int64))
	np.save(f"{save_root}/subset_uc_idx.npy", np.array(uc_idx, dtype=np.int64))
	np.save(f"{save_root}/subset_ui_idx.npy", np.array(ui_idx, dtype=np.int64))
	np.save(f"{save_root}/subset_ci_idx.npy", np.array(ci_idx, dtype=np.int64))

	print(dashed_line)
	print(f"[DiaNA-static] target split sizes  CC:{len(cc_idx)}  UC:{len(uc_idx)}  UI:{len(ui_idx)}  CI:{len(ci_idx)}")
	print(dashed_line)

	return


if __name__ == "__main__":
	""" Argument """
	parser = argparse.ArgumentParser()
	config = load_config("config/STR.yaml")
	parser.set_defaults(**config)
	
	parser.add_argument(
		"--source_data", default="data/train/synth/", help="path to source domain data",
	)
	parser.add_argument(
		"--target_data", default="data/train/real/", help="path to target domain data",
	)
	parser.add_argument(
		"--select_data",
		required=True,
		help="path to select data",
	)
	parser.add_argument(
		"--checkpoint_dir", type=str, default="stratify/HDGE", help="models are saved here",
	)
	parser.add_argument(
		"--batch_size", type=int, default=16, help="input batch size",
	)
	parser.add_argument(
		"--batch_size_val", type=int, default=128, help="input batch size val",
	)
	parser.add_argument(
		"--epochs", type=int, default=20, help="number of epochs to train for",
	)
	parser.add_argument(
		"--no_dropout", action="store_true", help="no dropout for the generator",
	)
	parser.add_argument(
		"--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2",
	)
	""" Adaptation """
	parser.add_argument(
		"--num_subsets",
		type=int,
		required=True,
		help="hyper-parameter n, number of subsets partitioned from target domain data",
	)
	parser.add_argument(
		"--beta",
		type=float,
		help="hyper-parameter beta in HDGE formula, 0<beta<1",
	)
	parser.add_argument(
		"--train", action="store_true", default=False, help="training or not",
	)
	""" Model Architecture """
	parser.add_argument(
		"--model",
		type=str,
		required=True,
		help="CRNN|TRBA",
	)
	parser.add_argument(
		"--saved_model",
		required=True, 
		help="path to source-trained model for adaptation",
	)

	args = parser.parse_args()

	if args.model == "CRNN":  # CRNN = NVBC
		args.Transformation = "None"
		args.FeatureExtraction = "VGG"
		args.SequenceModeling = "BiLSTM"
		args.Prediction = "CTC"

	elif args.model == "TRBA":  # TRBA
		args.Transformation = "TPS"
		args.FeatureExtraction = "ResNet"
		args.SequenceModeling = "BiLSTM"
		args.Prediction = "Attn"
	
	""" Seed and GPU setting """
	random.seed(args.manual_seed)
	np.random.seed(args.manual_seed)
	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed(args.manual_seed)

	# Set num_worker = 0 to prevent multiprocessing on macos
	args.workers = 0

	# cudnn.benchmark = True  # it fasten training
	# cudnn.deterministic = True

	# if sys.platform == "win32":
	# 	args.workers = 0

	# args.gpu_name = "_".join(torch.cuda.get_device_name().split())
	# if sys.platform == "linux":
	# 	args.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
	# else:
	# 	args.CUDA_VISIBLE_DEVICES = 0  # for convenience

	# command_line_input = " ".join(sys.argv)
	# print(
	# 	f"Command line input: CUDA_VISIBLE_DEVICES={args.CUDA_VISIBLE_DEVICES} python {command_line_input}"
	# )

	main(args)
