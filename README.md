# Create virtual environment using conda and install GLF
### -------------------------------------

### To create a virtual environment using conda, follow these steps:

### 1. Open the Anaconda Prompt or your preferred terminal.
### 2. Navigate to the project directory using the `cd` command.
### 3. Run the following command to create a new virtual environment:
####    `conda create --name myenv python=3.10.12`
###    Replace `myenv` with the desired name for your virtual environment.
###    You can also specify a different Python version if needed.
### 4. Activate the virtual environment by running:
####    `conda activate myenv`
###    Replace `myenv` with the name of your virtual environment.
### 5. Your virtual environment is now active and ready to use.

### Install project dependencies from requirements.txt
### -------------------------------------------------

### Once you have created and activated the virtual environment, you can install the project dependencies using the requirements.txt file. Here's how:

## 1. Make sure you are in the project directory and your virtual environment is active.
## 2. Run the following command to install the dependencies:
####    `pip install -r requirements.txt`
###    This command will install all the packages listed in the requirements.txt file.
## 3. Wait for the installation to complete. Once it's done, you're ready to work on the project.

### Remember to activate the virtual environment every time you work on the project to ensure that you are using the correct dependencies.

### Note: If you don't have conda installed, you can install it from the Anaconda website: https://www.anaconda.com/products/individual

### Note: If you don't have a requirements.txt file, you can create one by running the following command:
####       `pip freeze > requirements.txt`
###       This will generate a requirements.txt file with all the currently installed packages in your virtual environment.
### Requirements to run all python and jupyter notebook files

## 4. Install Git Large File Storage using [GLF Link](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
