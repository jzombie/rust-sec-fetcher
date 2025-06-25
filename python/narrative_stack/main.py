import typer
from config import project_paths
from dataclasses import asdict

# https://pypi.org/project/typer/
app = typer.Typer()

# TODO: Maybe `hydra` is a better direction for main.py? https://hydra.cc/docs/1.3/intro/


@app.command()
def paths():
    print("")
    for key, path in asdict(project_paths).items():
        if key == "python_root":
            continue
        

        print(f"{key}\t{path}")
    print("")

# @app.command()
# def preprocess(stage: int):
#     if stage == 1:
#         print("stage 1..")
    
#     else:
#         raise NotImplementedError(f"Stage {stage} not implemented")

if __name__ == "__main__":
    app()
