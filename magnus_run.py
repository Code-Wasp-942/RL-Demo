import os

import magnus
import ppo_train


def main() -> None:
    cfg = ppo_train.parse_args()
    ppo_train.main()

    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"training output file not found: {cfg.ckpt_path}")

    file_secret = magnus.custody_file(cfg.ckpt_path)
    print(f"file secret: {file_secret}")


if __name__ == "__main__":
    main()
