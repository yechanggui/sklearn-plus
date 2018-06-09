============
Sklearn Plus
============

.. image:: https://img.shields.io/pypi/v/sklearn_plus.svg
    :target: https://pypi.python.org/pypi/sklearn_plus

.. image:: https://api.travis-ci.org/ybbaigo/sklearn-plus.svg
    :target: https://travis-ci.org/ybbaigo/sklearn_plus

.. image:: https://readthedocs.org/projects/sklearn-plus/badge/?version=latest
    :target: https://sklearn-plus.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


    A set of python modules for Natural Language Processing


    * Free software: MIT license
    * Documentation: https://sklearn-plus.readthedocs.io.


Sklearn plus
----------------

Sklearn plus is an end-to-end neural network API, written in Python. Currently we focus on Natural Language Processing(NLP) tasks. It is based on Tensoflow_ and scikit-learn_. It assembles many preprocessing function, utils and deep learning models about NLP tasks.

It was developed with a focus on expanding sklearn and making using deep learning model to handle NLP tasks easier.

Features
----------------

**End-To-End**. Sklearn plus is based on Tensorflow and implements many classical models. Also it offers easy and consistent API with sklearn style for specific NLP task.

**Easy extensibility**. Sklearn plus inherits sklearn base classes and follows sklearn API design principles. It is easy to assemble new models, preprocessing functions and utils in it.

.. _Tensoflow: https://www.tensorflow.org/
.. _scikit-learn: http://scikit-learn.org/stable/


Quick Start
----------------

* TODO

Installation
----------------

There are two ways to install Sklearn plus:

1. **Install sklearn plus from PyPI(recommended)**:

::

  sudo pip install sklearn-plus

If you are using a virtualenv, you may want to avoid using sudo:

::

  pip install sklearn-plus

2. **Install sklearn plus from the GitHub source:**

::

  git clone https://github.com/ybbaigo/sklearn-plus.git
  cd sklearn-plus
  sudo python setup.py install


Contributing pull requests
--------------------------------

Here's a quick guide to submitting your improvements:

1. **Write the code.** There are three base modules in sklearn plus: preprocess, utils and nerual_network. Write your code in the three modules and reference to the samples in them. We use PEP8 syntax conventions.
2. **Write the docstrings.** Make sure any code you touch still has up-to-date docstrings and documentation. Please follow the numpydoc_style_.

3. **Write tests.** Your code should have full unit test coverage. If you want to see your PR merged promptly, this is crucial.

4. **Make sure all tests are passing.** Make sure your code tests can pass on Python 2.7 and Python 3.6 with Tensorflow 1.1.0.

5. **Make sure that your PR does not add PEP8 violations.** Make sure that your PR does not add PEP8 violations you can check it by flake8_:

  * install flake8: ```pip install flake8```
  * check: ```flake8 path/to/code/```

6. **Commit, use appropriate, descriptive commit messages.**

7. **Submit your PR.** If you have complete (and passing) unit tests as well as proper docstrings/documentation, your PR is likely to be merged promptly.

.. _numpydoc_style: https://numpydoc.readthedocs.io/en/latest/format.html#overview
.. _flake8: http://flake8.pycqa.org/en/latest/index.html#quickstart)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
