import subprocess
import sys

import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use("ggplot")
    seed = int(sys.argv[1])
    file = f"{seed:04}"

    subprocess.run("cargo build --features local --release", shell=True)
    subprocess.run(
        "./target/release/ahc031" + f"< tools/in/{file}.txt > tools/out/{file}.txt",
        shell=True,
    )
    subprocess.run(
        "./tools/target/release/vis" + f" tools/in/{file}.txt tools/out/{file}.txt",
        shell=True,
    )
    subprocess.run(f"pbcopy < tools/out/{file}.txt", shell=True)

    # 過去ログとの比較
    import pandas as pd

    df = pd.read_csv("./log/database.csv")
    print(
        df[(df.input_file == f"tools/in/{file}.txt")][
            ["solver_version", "score"]
        ].sort_values(by="score")[:20]
    )
