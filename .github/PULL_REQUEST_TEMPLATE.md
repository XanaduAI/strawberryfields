### Pull request guidelines

* Please do not make a pull request for minor typos/changes to the code style - make an issue instead.
* For major features, consider making an independent app that runs on top of Strawberry Fields,
  rather than modifying strawberry fields directly.

### Before submitting

Before submitting a pull request, please make sure the following is done:

* Fork the repository and create your branch from master.
* If you've fixed a bug or added code that should be tested, add a test to the test directory!
  All new features *must* include a unit test.
* Ensure that the test suite passes, by running: make test
  You can also test individual backends by running: make test-backendname
* The Strawberry Fields source code conforms to PEP8 standards.
  We check all of our code against pylint.
  To lint modify files, run: pylint strawberryfields/path/to/file

### Pull request template

When ready to submit, delete everything above the line and fill in the pull request template.

------------------------------------------------------------------------------------------------------------

**Description of the Change:**

**Benefits:**

**Possible Drawbacks:**

**Related GitHub Issues:**
