# NMR-onion
Hello and welcome to the NMR-onion github repo. Prior to installing the program please make sure you have downloaded the packages listed in addational requriements before proceeding

# Installing python
Getting python via anconda <https://www.anaconda.com/products/distribution>

# Additonal packages required
1) pytorch <https://pytorch.org/>
2) NMRglue <https://nmrglue.readthedocs.io/en/latest/tutorial.html>
3) sklearn <https://scikit-learn.org/stable/install.html>
4) rle <https://pypi.org/project/python-rle/>

Each of these can be installed by opening a command prompt/terminal and type:
```
python -m pip install nmrglue
pip install -U scikit-learn
pip install python-rle
```

For pytorch following the link of <https://pytorch.org/> and select your operating system
Note: If you installed anconda it is proably most easy to install from the anaconda prompt on Windows

# Downloading the program
Having installed python and the addtional packages our program can now be downloaded by cloning the repo. A guide on how to do this is shown in the link below

Github clone guide: <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>

To download our repository type the following command:

```
git clonehttps://github.com/Mabso1/NMR-onion_test.git
```

Having trouble installing, checkout our youtube video 

# Running the program
To run the program follow the instructions in the example folder using this link: 

To download the data for running the examples either clone the repository or followig this guide <https://www.wikihow.com/Download-a-GitHub-Folder> on how to download a single folder.

Note, currently only data recorded on Bruker instruments are supported for the native data import function. You can however use the various NMRgule data import features for loading different data format, but you have to write the import data function yourself. Alternatively save the specrum as a txt file and manually do an inverse fft to get the time domain data, while also settting the spectral parameters manually. 

# Additional information
For more information on the background behind the program read our article at




