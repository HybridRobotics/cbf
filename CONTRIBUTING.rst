Ready to contribute? Here's how to set up for local development.

1. Fork the repo on Github.
2. Clone your fork locally.
3. Create the environment with::

    $ conda env create -f environment.yml

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, ensure that your code is formatted using black and isort::

    $ black .
    $ isort .

   you can add pre-commit hooks for both isort and black to make all formatting easier::

    $ pip install pre-commit

6. Commit your changes and push your branch to GitHub and submit a pull request through the GitHub website. 