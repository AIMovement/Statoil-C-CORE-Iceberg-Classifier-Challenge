# README #

## What is this repository for? ##

This repo is used as a collaboration platform for the Kaggle competition
[Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge).
The programming language for the repo is Python.

## How do I get set up? ##

### Git

If developing on windows, [Git bash](https://git-scm.com/download/win) is
recommended. It includes a small linux-like environment based on
[MSYS2](http://www.msys2.org/) (which in turn is based on
[Cygwin](https://www.cygwin.com/).)


### Python

Python requires an IDE or text editor of choice e.g.: 
* PyCharm (https://www.jetbrains.com/pycharm/) 		
* Sublime Text (https://www.sublimetext.com/) 		

Install Python 3.5.X or 3.6.X as Tensorflow requires it. 

	For Windows: 
		•	Download Windows x86-64 executable installer from [python.org](https://www.python.org/downloads/)
		•	Tick the “Add Python to PATH” option 
		•	Click “Custom Installation” and leave the first page unchanged. 
		•	Choose to install python to the directory “C:/Python3X” where X is the Python version preferred. 
		•	Click “Install” 
		•	Check if the installation was successful by opening the command prompt and type: python3 and verify that a similar response is obtained as below: 

	For OSX: 
		Follow: https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072

#### First time setup

Before starting development a so-called virtual environment is good to setup.
Do this by executing

`setup/setup_virtualenv.sh`

This will install some nice-to-have python modules like Keras, Tensorflow and
Matplotlib.

Now do the steps below under
[Every time initialization](#every-time-initialization)

#### Every time initialization

To activate the local python environment, go to the root of the Git repo and
execute

on linux: `source env/bin/activate`

on windows: `source env/Scripts/activate`

Your prompt should now show `(env)` as the first characters to the left.


## Install python packages

In order to install new python packages, run

`pip install <package>`

If doing this on a Volvo computer inside the Volvo network, you might have
problems with the proxy. Try running

`export http_proxy=proxy.volvocars.net:83 https_proxy=$http_proxy`

and then retry.

If you want others to use the addon, add it to the list of modules to install
by running `pip freeze > setup/REQUIREMENTS.txt` and commit the updated file.

		
## Contribution guidelines ##
Don't push competition data to repo. 

## Who do I talk to? ##

Slack-workspace: "Kaggle-ED"
