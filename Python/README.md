### Installation

After cloning the repository, all commands can be executed at the root level.

#### Using Virtual Environment to avoid global installation

It is possible to ignore these commands, but the global package installation can lead your environment to unstable statuses (Tested at Ubuntu/Linux and Windows 11)

- `python -m venv ENV`
    - This command will creat a `ENV` folder in the root level
- Execute `ENV\Scripts\activate.ps1` (Windows-Powershell) or `source ENV/bin/activate` (Bash-Linux)
- (ENV)`python -m pip install --upgrade pip`

#### Install dependencies

- `pip install -r requirements.txt`
NOTE: The requirements file includes some development dependencies to execute Jupyter notebooks in a VSCode environment.
- Your enviroment is ready to execute all scripts into `Python` folder in the virtual enviorment