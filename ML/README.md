## Paper Title - Machine Learning instructions

### Installation

After clone the repository, all commands can be executed at the root level

#### Using Virtual Enviroment to avoid global installation

It is possible to ignore these commands, but the global package installation can lead your enviroment to unstable statuses (Tested at Ubuntu/Linux and Windows 11)

- `python -m venv envML`
- `envML\Scripts\activate.ps1` (Windows-Powershell) or `source envML/bin/activate` (Bash-Linux)
- (envML)`python -m pip install --upgrade pip`

#### Install dependencies

- `pip install -r requirements.txt`
NOTE: The requirements file includes some development dependencies to execute Jupyter notebooks in a VSCode enviroment.
- Go to the directory `ML/Notebooks/` and execute it to use the tool.
