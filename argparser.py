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
    args = parser.parse_args()

    return args