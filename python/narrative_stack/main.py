import typer
from config import project_paths
from dataclasses import asdict

# https://pypi.org/project/typer/
app = typer.Typer()

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
