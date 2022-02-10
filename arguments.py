import argparse


def parser(command):
    arg_command = command[1:]
    home = "/home/share/DTI_PDBbind2/data_pdbbind2"
    parser = argparse.ArgumentParser(description="parser for train and test")

    parser.add_argument("--dim_gnn", help="dim_gnn", type=int, default=128)
    parser.add_argument("--n_gnn", help="depth of gnn layer", type=int, default=3)
    parser.add_argument("--ngpu", help="ngpu", type=int, default=1)
    parser.add_argument("--restart_file", help="restart file", type=str, default=None)
    parser.add_argument(
        "--potential",
        help="potential",
        type=str,
        default="harmonic",
        choices=["morse", "harmonic", "morse_all_pair", "gnn", "cnn3d_kdeep"],
    )
    parser.add_argument(
        "--pos_noise_std",
        help="std of noise added to the position",
        type=float,
        default=0.0,
    )
    parser.add_argument("--filter_gamma", help="filter gamma", type=float, default=10.0)
    parser.add_argument(
        "--filter_spacing", help="filter spacing", type=float, default=0.5
    )
    parser.add_argument("--edgeconv", action="store_true", help="edge conv")
    parser.add_argument("--no_rotor_penalty", action="store_true", help="rotor penaly")
    parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.0)
    parser.add_argument("--vdw_N", help="vdw N", type=float, default=6.0)
    parser.add_argument(
        "--max_vdw_interaction", help="max vdw _interaction", type=float, default=0.0356
    )
    parser.add_argument(
        "--min_vdw_interaction", help="min vdw _interaction", type=float, default=0.0178
    )
    parser.add_argument(
        "--dev_vdw_radius", help="deviation of vdw radius", type=float, default=0.2
    )
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--lattice_dim", type=int, default=24)
    parser.add_argument("--grid_rotation", action="store_true")
    if "train.py" in command[0] or "test.py" in command[0]:
        parser.add_argument(
            "--filename",
            help="filename",
            default="/home/wykgroup/mseok/data/DTI_PDBbind/CASF-2016/scoring/pdb_to_affinity.txt",
        )
        parser.add_argument(
            "--key_dir",
            help="key directory",
            type=str,
            default="/home/wykgroup/mseok/work/DL/DTI_PDBbind/debug1/",
        )
        parser.add_argument(
            "--data_dir",
            help="data file path",
            type=str,
            default="/home/wykgroup/mseok/data/DTI_PDBbind/CASF-2016/scoring/data/",
        )
        parser.add_argument("--batch_size", help="batch size", type=int, default=1)
        parser.add_argument(
            "--num_workers", help="number of workers", type=int, default=1
        )

    # for train
    if "train.py" in command[0]:
        parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
        parser.add_argument(
            "--lr_decay", help="learning rate decay", type=float, default=1.0
        )
        parser.add_argument(
            "--weight_decay", help="weight decay", type=float, default=0.0
        )
        parser.add_argument(
            "--num_epochs", help="number of epochs", type=int, default=100
        )
        parser.add_argument(
            "--train_result_filename",
            help="train result filename",
            type=str,
            default="train_result.txt",
        )
        parser.add_argument(
            "--test_result_filename",
            help="test result filename",
            type=str,
            default="test_result.txt",
        )
        parser.add_argument(
            "--train_result_docking_filename",
            help="train result docking_filename",
            type=str,
            default="train_result_docking.txt",
        )
        parser.add_argument(
            "--test_result_docking_filename",
            help="test result docking filename",
            type=str,
            default="test_result_docking.txt",
        )
        parser.add_argument(
            "--train_result_screening_filename",
            help="train result screening filename",
            type=str,
            default="train_result_screening.txt",
        )
        parser.add_argument(
            "--test_result_screening_filename",
            help="test result screening filename",
            type=str,
            default="test_result_screening.txt",
        )
        parser.add_argument(
            "--loss_der1_ratio", help="loss der1 ratio", type=float, default=1.0
        )
        parser.add_argument(
            "--loss_der2_ratio", help="loss der2 ratio", type=float, default=1.0
        )
        parser.add_argument(
            "--min_loss_der2", help="min loss der2", type=float, default=-100000000.0
        )
        parser.add_argument(
            "--loss_docking_ratio", help="loss docking ratio", type=float, default=1.0
        )
        parser.add_argument(
            "--min_loss_docking", help="min loss docking", type=float, default=-1.0
        )
        parser.add_argument(
            "--loss_screening_ratio",
            help="loss screening ratio",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--loss_screening2_ratio",
            help="loss screening ratio",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--save_dir", help="save directory of model save files", type=str
        )
        parser.add_argument(
            "--save_every", help="saver every n epoch", type=int, default=1
        )
        parser.add_argument(
            "--tensorboard_dir",
            help="save directory of tensorboard log files",
            type=str,
        )
        parser.add_argument(
            "--filename2", help="filename2", default=home + "/pdb_to_affinity.txt"
        )
        parser.add_argument(
            "--key_dir2",
            help="key directory",
            type=str,
            default="/home/udg/msh/urp/DTI_PDBbind/keys",
        )
        parser.add_argument(
            "--data_dir2", help="data file path", type=str, default=home + "/data/"
        )
        parser.add_argument(
            "--filename3", help="filename", default=home + "/pdb_to_affinity.txt"
        )
        parser.add_argument(
            "--key_dir3",
            help="key directory",
            type=str,
            default="/home/udg/msh/urp/DTI_PDBbind/keys",
        )
        parser.add_argument(
            "--data_dir3", help="data file path", type=str, default=home + "/data/"
        )
        parser.add_argument(
            "--filename4", help="filename", default=home + "/pdb_to_affinity.txt"
        )
        parser.add_argument(
            "--key_dir4",
            help="key directory",
            type=str,
            default="/home/udg/msh/urp/DTI_PDBbind/keys",
        )
        parser.add_argument(
            "--data_dir4", help="data file path", type=str, default=home + "/data/"
        )

    # for test
    if "test.py" in command[0]:
        parser.add_argument(
            "--test_result_filename",
            help="test result filename",
            type=str,
            default="test_result.txt",
        )

    # for predict
    if "predict.py" in command[0]:
        parser.add_argument(
            "--ligand_files",
            help="list of ligand file",
            nargs="+",
            type=str,
        )
        parser.add_argument(
            "--protein_files",
            help="list of protein file",
            nargs="+",
            type=str,
        )
        parser.add_argument(
            "--output_files",
            help="list of output file",
            nargs="+",
            type=str,
        )

    args = parser.parse_args(arg_command)
    return args
