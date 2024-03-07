=============================
Contributing to the code base
=============================

.. contents:: Table of contents:
   :local:

Code standards
--------------

Writing good code is not just about what you write. It is also about *how* you write it.
During Continuous Integration testing, several tools will be run to check your code for stylistic errors.
Good style is a requirement for submitting code to Xinference.

In addition, it is important that we do not make sudden changes to the code that
could have the potential to break a lot of user code as a result. Therefore
we need it to be as backwards compatible as possible to avoid mass breakages.

Backwards compatibility
-----------------------

Please try to maintain backward compatibility. If you think breakage is necessary,
clearly state why as part of the pull request. Also, be careful when changing method
signatures and add deprecation warnings where needed. Also, add the deprecated sphinx
directive to the deprecated functions or methods.

You'll also need to

1. Write a new test that asserts a warning is issued when calling with the deprecated argument
2. Update all of Xinference existing tests and code to use the new argument

Type hints
----------

Xinference strongly encourages the use of :pep:`484` style type hints. New development should
contain type hints and pull requests to annotate existing code are accepted as well!

Test-driven development
-----------------------

Xinference is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <https://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests. Often the test can be
taken from the original GitHub issue. However, it is always worth considering additional
use cases and writing corresponding tests.

Adding tests is frequently requested after code is pushed to Xinference. Thus,
it is worth getting in the habit of writing tests ahead of time so this is never an issue.