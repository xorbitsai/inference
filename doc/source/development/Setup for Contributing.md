# Setup for Contributing



Table of contents:

- [Getting startted with Git](#Getting-startted-with-Git)
- [Setting up Conda environment](#Setting-up-Conda-environment)
- [Xinference Installation](#Xinference-Installation)
- [Frontend Compilation](#Frontend-Compilation)



## [Getting startted with Git](#Setup-for-Contributing)



For more details, refer to [Working with the code](https://github.com/xorbitsai/xorbits/blob/main/doc/source/development/contributing.rst#working-with-the-code).

If the speed of `git clone` is slow, you can use the following command to add a proxy:

```
export https_proxy=YourProxyAddress
```



## [Setting up Conda environment](#Setup-for-Contributing)



Before formally installing Xinference, it's recommended to create a new Conda environment for ease of subsequent operations.

If you're using a campus-level public computing cloud platform, the setup command is as follows:

```
mkdir /fs/fast/ustudentID/envs
conda create --prefix=/fs/fast/ustudentID/envs/xinf
conda activate /fs/fast/ustudentID/envs/xinf
```

The `studentID` needs to be replaced with the corresponding student ID of your server account, and `xinf` can be replaced with a custom Conda environment name.

Afterward, you'll need to install Python and npm in the newly created Conda environment. Here are the commands:

```
conda install python=3.10
conda install nodejs
```



## [Xinference Installation](#Setup-for-Contributing)



For more details, refer to [installation](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)。

After the initial installation of Xinference, you need to run the following commands in the `/inference/` directory to check if it can run properly:

```
pip install -e .
xinference-local
```

If errors occur or the process freezes during execution, the next step is to compile the frontend, refer to [Frontend-Compilation](#Frontend-Compilation).


If errors occur or the process freezes during execution, the next step is to compile the frontend.

If the commands run successfully, you can use Xinference normally. For detailed usage instructions, refer to [using_xinference](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html).



## [Frontend Compilation](#Setup-for-Contributing)



Firstly, navigate to the `/inference/xinference/web/ui` directory. If the `/node_modules/` folder already exists in this directory, it's recommended to manually delete it. Then, execute the following command to clear the cache:

```
npm cache clean
```

Next, execute the following command in this directory to compile the frontend:

```
npm install
npm run build
```


After compiling the frontend, you can retry running Xinference.

At this point, all the necessary environment setup for development has been completed.
