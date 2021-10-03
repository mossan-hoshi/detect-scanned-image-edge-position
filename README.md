# python_opencv
detect edge position of image scanned by flat head scanner.

## Requirements
- Windows 10 / Linux
  - It is expected to work on Macs, but we have not tested it yet.
- python 3.9
- pipenv
- Visual Studio Code(Optional)

## Installation
1. [Optional] Install Pyenv
   - It is recommended to use pyenv to enable multiple versions of python on your system.
   -  [Linux] [pyenv](https://github.com/pyenv/pyenv)
   -  [Windows] [pyenv-win](https://github.com/pyenv-win/pyenv-win)
2. Install [pipenv](https://github.com/pypa/pipenv)
3. Generate virtual environment
   - Create a virtual environment using pipenv.
     - [It is recommended to set the environment variables to be placed under the project in advance.](https://pipenv-fork.readthedocs.io/en/latest/advanced.html#custom-virtual-environment-location)
   - [Release Environment] `pipenv install`
   - [Debug Environment] `pipenv install --dev`
4. [Optional] Install [VSCode](https://code.visualstudio.com/)
  

## How to Execute
### When Using VSCode
1. Open repository folder by VSCode, and activate virtual environment.
  ![](image/2021-10-02-21-56-48.png)
2. Run by using VSCode's debug function.

### When Using from Shell
1. Open repository folder by shell terminal.
2. launch pipenv shell.
  - execute command `pipenv shell`
3. Run the program.
```shell
python `
--in_dir_path          "INPUT DIRECTORY PATH(default:"data/")" `
--in_suffix            "INPUT IMAGE FILE EXTENSION(ex:".jpg")" `
--background_is_black  "BACKGROUND COLOR(False:White, True:Black)" `
--out_dir_path         "OUTPUT DIRECTORY PATH(default:"out/")" `
--out_suffix           "OUTPUT IMAGE FILE EXTENSION(ex:".png")" `
--out_cvs_file_name    "OUTPUT CSV FILE NAME(default:"result.csv")"
```
