.. _contributing-code:

Contributing Code to Brails++
=============================

.. warning::

   Documentation under development.
   Statements reflect the personal views of JVM, but really this serves as a template and definitely feel free to change anything.

Contributors should `fork`_ the GitHub `repository`_ and create `pull requests`_.

.. note::

   For bug reports, if you are having trouble following the installation instructions on this page, please `open an issue`_.
   We would be happy to help.
   For bug fixes or even fixing typos/inconsistencies in this documentation, please submit a `pull request`_.
   Contributions are welcome!

.. todo::

   For GitHub issues, create appropriate templates so that we don't have to ask users to follow-up with the same kind additional information (OS, version, ...) each time an issue is opened.

.. _fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
.. _repository: https://github.com/NHERI-SimCenter/BrailsPlusPlus
.. _pull requests: https://docs.github.com/en/pull-requests
.. _pull request: https://docs.github.com/en/pull-requests
.. _open an issue: https://github.com/NHERI-SimCenter/BrailsPlusPlus/issues/new

Installation Instructions
-------------------------

We provide instructions for creating a development setup that includes creating a virtual environment.
Doing so isolates the setup from other Python projects and avoids modifying the system's Python setup, which may lead to unexpected behavior on some systems.
We assume that you already have ``git`` and ``python`` installed.

As a first step, please `fork the repository`_ on GitHub.
Then clone it at your desired location. The location you chose is important, since the module's code will be evaluated from there.
Moving the local repository will break the environment setup, but it can be easily torn down and recreated by repeating the relevant steps on this page.

.. _fork the repository: https://github.com/NHERI-SimCenter/BrailsPlusPlus


.. code-block::
   :caption: Clone the repository

   cd {/desired/path/to/repo}
   git clone https://github.com/{your_account}/BrailsPlusPlus
   cd BrailsPlusPlus


Similar to the repo's location, the path to the virtual environment is important.
Developers are advised to store all their virtual environments in a specific folder.

.. code-block::
   :caption: Create a Python virtual environment

   # create the environment.
   # We assume you store your virtual environments in `envs/`
   # and you name this one `brails_env`.
   python -m venv {/desired/path}/envs/brails_env

   # activate
   source {/desired/path}/envs/brails_env/bin/activate
   # make sure that your prompt changes to reflect the environment activation
   
   # check
   which python
   # Should output the path of the environment instead of your system's Python.
   
Install ``brails++``

.. code-block::
   :caption: Install

   python -m pip install -e .[development]

``-e`` installs ``brails++`` in `editable mode`_. This makes it so that the module's code is executed from within the installation directory, and any changes to the code take effect immediately, without needing to re-install.
``[development]`` ensures that additional development dependencies get installed; packages required for linting, testing, building and deploying ``brails++``.
They can be found in ``setup.py`` under ``extras_require``.

After issuing the command and reviewing the standard output---assuming no errors---you can perform the following checks:

.. code-block::
   :caption: Checks

    (brails-env) $ python
    >>> import brails
    >>> brails.__file__
    # It should return a path that points to the local repository
    >>> import sys
    >>> sys.executable
    # It should return the path to `python` in the virtual environment

You now have a working local development repository of ``brails++``.

.. _editable mode: https://pip.pypa.io/en/stable/cli/pip_install/

In case you wish to tear down (remove and clean up) the environment, you can simply remove its directory.

.. code-block::
   :caption: Checks

   # Unix-like
   rm -r {/desired/path}/envs/brails_env
   # Windows
   rd /s /q {C:\desired\path}\envs\brails_env


Code Quality and Style
----------------------
Striving to maintain a high code quality, we require all code contributed to ``brails++`` to be compliant with `PEP 8`_ and include unit tests.
This is easy to achieve with proper linting; instructions are provided in the sections that follow.
Linting and testing the code before commits is a great way to avoid introducing bugs that require additional commits to address, or (much worse) remain dormant in the code base until an unfortunate user has to deal with them.
Maintaining good code quality, comprehensive unit tests and documentation in parallel with additional features is much more time-efficient than reviving outdated test suites, documentation and fixing long-standing formatting issues.

.. _PEP 8: https://peps.python.org/pep-0008/


Linting and formatting
----------------------

For linting, we use Ruff.
It is installed in the environment when specifying the ``[development]`` argument when installing ``brails++``.
The behavior of the linter is configured with ``pyproject.toml``.

The following commands can be used to lint the code from the command line.

.. code-block::
   :caption: Linting the code

   ruff check  # Default command
   ruff check --output-format concise  # Concise output
   ruff check help  # Learn more about ruff check.

Ruff can automatically fix certain issues.

.. code-block::
   :caption: Auto-fix

   ruff check --fix

If needed, warnings can be silenced by adding ``# noqa`` directives at the offending lines.
This can be done by hand, or automatically with the following command:

.. code-block::
   :caption: Add # noqa: ...

   ruff check --add-noqa

While the command-line approach definitely works, we recommend setting up linter integration with your text editor of choice.

We also use Ruff for code formatting.

Formatting enforces a consistent formatting style to the code.
Using it on specific regions or entire files that you are working with can save you a lot of time addressing style-related linter warnings.
Formatting of specific regions can be turned off with specific comments (``# fmt: off``, ``# fmt: on``).
This can be useful if you believe that certain lines of code are more readable with your own formatting.

.. code-block::
   :caption: Turning off formatting

   {source code that will be formatted}
   {source code that will be formatted}
   # fmt: off
   {source code that is not formatted}
   {source code that is not formatted}
   # fmt: on
   {source code that will be formatted}
   {source code that will be formatted}


To run the formatter, use the following command:
   
.. code-block::
   :caption: Code formatting

   ruff format

Unit Testing
------------

For unit testing in ``brails++`` we use ``pytest``, and it should become available when installing with ``[development]``.
The unit tests reside in the ``tests`` directory, and they can be used as an example reference for future contributions.
*Please* write unit tests for the code you contribute.

To run the entire test suite, use this command:

.. code-block::
   :caption: Running all unit tests

   # From package root (BrailsPlusPlus/)
   python -m pytest --cov=brails --cov-report html tests

This will produce a coverage report in ``html`` format, which can be viewed using your web browser of choice.
Opening ``BrailsPlusPlus/htmlcov/index.html`` you will be presented with a table containing all files and the overall coverage percentage, which is the proportion of executable lines that ran during testing.
Clicking on a file shows the source code with highlighting identifying which specific lines were covered/missed.
We strongly encourage contributions that include comprehensive testing and do not reduce the existing overall coverage score.

While this setup works on its own, similar to linting, most text editors and IDEs have integrated support for unit testing, with features that can help improve the efficiency of the unit testing process.
Consult the documentation of your editor of choice to see what functionality is available.

.. todo::

   Unit tests can be executed under an interactive session and debugged with pdb, which is **super useful** when writing them. Document how to do this.

.. todo::

   Encourage contributors to reach out to us for support on writing unit tests for their contributions.


Documentation
-------------

The documentation files of ``brals++`` exist in a dedicated branch named ``documentation``.
This isolates the documentation-related commits from code commits, making it easier for us to review code changes.
*Please* write documentation for any user-facing functionality you implement.

To compile the documentation follow these steps:

.. code-block::
   :caption: Compiling the documentation

   # From package root (BrailsPlusPlus/)
   git checkout documentation
   cd docs/
   make html

To view it, open ``BrailsPlusPlus/docs/build/html/index.html`` with your web browser of choice.

To update the documentation merge the main development or any feature branches, and update the corresponding ``.rst`` files residing in ``source/``.
Make any changes to the ``documentation`` branch of your fork, and open a separate pull request, assigning our ``documentation`` branch as a recipient.

Git Guidelines
--------------

.. todo::

   Here we will reiterate that contributors should open pull requests, and then specify our guidelines for how contributors should utilize branches, when to delete them, what branches to use when making pull requests, and so on.

