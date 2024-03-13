==================================
Creating a development environment
==================================

.. contents:: Table of contents:
   :local:

Before proceeding with any code modifications, it's essential to set up the necessary environment for Xinference development,
which includes familiarizing yourself with Git usage, establishing an isolated environment, installing Xinference, and compiling the frontend.

Getting startted with Git
-------------------------

Now that you have identified an issue you wish to resolve, an enhancement to incorporate, or documentation to enhance,
it's crucial to acquaint yourself with GitHub and the Xinference codebase.

To the new user, working with Git is one of the more intimidating aspects of contributing to Xinference.
It can very quickly become overwhelming, but sticking to the guidelines below will help simplify the process 
and minimize potential issues. As always, if you are having difficulties please
feel free to ask for help.

The code is hosted on `GitHub <https://github.com/xorbitsai/inference>`_. To
contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_. We use `Git <https://git-scm.com/>`_ for
version control to allow many people to work together on the project.

`GitHub has instructions <https://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git. All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

Some great resources for learning Git:

* `Official Git Documentation <https://git-scm.com/doc>`_
* `Pro Git Book <https://git-scm.com/book/en/v2>`_
* `Git Tutorial by Atlassian <https://www.atlassian.com/git/tutorials>`_
* `Git - Concise Guide <http://rogerdudler.github.io/git-guide/index.zh.html>`_

.. note::
   If the speed of ``git clone`` is slow, you can use the following command
   to add a proxy:

   ::

      export https_proxy=YourProxyAddress

Creating an isolated environment
--------------------------------

Before formally installing Xinference, it's recommended to create an isolated 
environment, using Conda recommended, for ease of subsequent operations.

::

   conda create --name xinf
   conda activate xinf

``xinf`` can be replaced with a custom Conda environment name.

Afterward, you'll need to install Python and Node.js (npm) in the newly created
Conda environment. Here are the commands:

::

   conda install python=3.10
   conda install nodejs

Install from source code
------------------------

Before we begin, please make sure that you have cloned the repository. 
Suppose you clone the repository as ``inference`` directory,  ``cd`` to this directory
where the ``setup.cfg`` and ``setup.py`` files are located, and run the following command:

::

   pip install -e .
   xinference-local

If the commands run successfully, you can use Xinference normally. For
detailed usage instructions, refer to
`using_xinference <https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html>`__.

If errors occur or the process freezes during execution, the next step
is to compile the frontend.

Frontend Compilation
--------------------

Navigate to the ``inference/xinference/web/ui`` directory. Then, execute the following command
to clear the cache:

::

   npm cache clean

If the command fails to execute, you can try adding the ``--force`` option.

.. note::
   If the ``node_modules`` folder already exists in this directory,
   it's recommended to manually delete it before cleaning the cache.

Next, execute the following command in this directory to compile the
frontend:

::

   npm install
   npm run build

Still, if the first command fails to execute, you can try adding the ``--force`` option.

After compiling the frontend, you can ``cd`` back to the directory
where the ``setup.cfg`` and ``setup.py`` files are located,
and install Xinference via ``pip install -e .``.
