## Paper Title - Machine Learning Instructions

### Installation

After cloning the repository, all commands can be executed at the root level.

#### Using Virtual Environment to avoid global installation

It is possible to ignore these commands, but the global package installation can lead your environment to unstable statuses (Tested at Ubuntu/Linux and Windows 11)

- `python -m venv ENV`
- `ENV\Scripts\activate.ps1` (Windows-Powershell) or `source ENV/bin/activate` (Bash-Linux)
- (ENV)`python -m pip install --upgrade pip`

#### Install dependencies

- `pip install -r requirements.txt`
NOTE: The requirements file includes some development dependencies to execute Jupyter notebooks in a VSCode environment.
- Go to the `Python\Exploratory_Notebooks` directory and execute it to use the tool.