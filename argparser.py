import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="paper103")
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=[0, 1, 2, 3],
        help="",
    )
    parser.add_argument("--config_file", type=str, default="./configs/Rtransformer.json")
    parser.add_argument("--profile_bandwidth", action="store_true")
    parser.add_argument("--profile_flops", action="store_true")
    args = parser.parse_args()

    return args