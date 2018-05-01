# Contributing to Strawberry Fields

Thank you for taking the time to contribute to Strawberry Fields!
:strawberry: :confetti_ball: :tada: :fireworks: :balloon:

Strawberry Fields is a collaborative effort with the quantum photonics and quantum computation
community - while we will continue working on adding new and exciting features to Strawberry Fields,
we invite you to join us and suggest improvements, research ideas, or even just to discuss how
Strawberry Fields fits into your workflow.

If your contribution becomes part of Strawberry Fields, or is highlighted in our Gallery, we will
send you some exclusive Xanadu Swag™ - including t-shirts, stickers, and more.

## How can I get involved in the community?

If you want to contribute but don't know where to start, start by checking out our
[documentation](https://strawberryfields.readthedocs.io). Have a go working through some of the tutorials,
and having a look at the Strawberry Fields API and code documentation to see how things work under the hood.

To chat directly with the team designing and building Strawberry Fields, as well as members of our
community - ranging from professors of quantum physics, to students, to those just interested in being a
part of a rapidly growing industry - you can join our [Slack channel](https://u.strawberryfields.ai/slack).

Available channels:

* `#strawberryfields`: For general discussion regarding Strawberry Fields
* `#sf_projects`: For discussion of research ideas and projects built on Strawberry Fields
* `#sf_bugs`: For discussion of any bugs and issues you run into using Strawberry Fields
* `#sf_interactive`: For discussion relating to the [Strawberry Fields Interactive](https://strawberryfields.ai) web application
* `#sf_docs`: For discussion of the Strawberry Fields [documentation](https://strawberryfields.readthedocs.io)

Sometimes, it might take us a couple of hours to reply - please be patient!

## How can I contribute?

It's up to you!

* **Be a part of our community** - chat to people in our Slack channel, respond to questions, and
  provide exciting updates of the projects/experiments you are investigating with Strawberry Fields

  You can even write your own Strawberry Fields tutorials, or blog about your simulation results.
  Send us the link, and we may even add it to our Gallery or our documentation as an external resource!

* **Test the cutting-edge Strawberry Fields releases** - clone our GitHub repository, and keep up to
  date with the latest features. If you run into any bugs, make a bug report on our
  [issue tracker](https://github.com/XanaduAI/strawberryfields/issues).

* **Report bugs** - even if you are not using the development branch of Strawberry Fields, if you come
  across any bugs or issues, make a bug report. See the next section for more details on the bug
  reporting procedure.

* **Suggest new features and enhancements** - hop on to our Slack channel, or use the GitHub issue tracker
  and let us know what will make Strawberry Fields even better for your workflow.

* **Contribute to our documentation, or to Strawberry Fields directly** - we are hoping for our documentation
  to become an online, open-source resource for all things continuous-variable. If you would like to add
  to it, or suggest improvements/changes, let us know - or even submit a pull request directly. All authors
  who have contributed to the Strawberry Fields codebase will be listed alongside the latest release.

* **Build an application on top of Strawberry Fields** - have an idea for an application, and Strawberry Fields
  provides the perfect computational backbone? Consider making a fully separate app that builds upon Strawberry Fields
  as a base. Ask us if you have any questions, and send a link to your application to support@xanadu.ai so we can highlight it in
  our documentation!

For some inspiration, you can also check out some preprepared [contribution challenges](.github/CHALLENGES.md).

Appetite whetted? Keep reading below for all the nitty-gritty on reporting bugs, contributing to the documentation,
and submitting pull requests.

## Reporting bugs

We use the [GitHub issue tracker](https://github.com/XanaduAI/strawberryfields/issues) to keep track of all reported
bugs and issues. If you find a bug, or have an issue with Strawberry Fields, please submit a bug report! User
reports help us make Strawberry Fields better on all fronts.

To submit a bug report, please work your way through the following checklist:

* **Search the issue tracker to make sure the bug wasn't previously reported**. If it was, you can add a comment
  to expand on the issue if you would like. If you're not sure, you can ask us in our Slack community, in the
  `#sf_bug` channel.

* **Fill out the issue template**. If you cannot find an existing issue addressing the problem, create a new
  issue by filling out the [issue template](.github/ISSUE_TEMPLATE.md). This template is added automatically to the comment
  box when you create a new issue. Please try and add as many details as possible!

* Try and make your issue as **clear, concise, and descriptive** as possible. Include a clear and descriptive title,
  and include all code snippets/commands required to reproduce the problem. If you're not sure what caused the issue,
  describe what you were doing when the issue occurred.

## Suggesting features, document additions, and enhancements

To suggest features and enhancements, you can either use the GitHub tracker, or use the `#strawberryfields` slack
channel. There is no template required for feature requests and enhancements, but here are a couple of suggestions
for things to include.

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested feature**.

  - If the feature is related to any theoretical results in continuous-variable quantum computation
    or quantum photonics, provide any relevant equations. Alternatively, provide references to papers/preprints,
    with the relevant sections/equations noted.
  - If the feature is workflow-related, or related to the use of Strawberry Fields,
    explain why the enhancement would be useful, and where/how you would like to use it.

* **For documentation additions**, point us towards any relevant equations/papers/preprints,
    with the relevant sections/equations noted. Short descriptions of its use/importance would also be useful.

## Pull requests

If you would like to contribute directly to the Strawberry Fields codebase, simply make a fork of the master branch, and
then when ready, submit a [pull request](https://help.github.com/articles/about-pull-requests). We encourage everyone
to have a go forking and modifying the Strawberry Fields source code, however, we have a couple of guidelines on pull
requests to ensure the main master branch of Strawberry Fields conforms to existing standards and quality.

### General guidelines

* **Do not make a pull request for minor typos/cosmetic code changes** - make an issue instead.
* **For major features, consider making an independent app** that runs on top of Strawberry Fields, rather than modifying
  Strawberry Fields directly.

### Before submitting

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added code that should be tested,
  add a test to the test directory!
* **All new functions and code must be clearly commented and documented.** Have a look through the source code at some of
  the existing function docstrings - the easiest approach is to simply copy an existing docstring and modify it as appropriate.
  If you do make documentation changes, make sure that the docs build and render correctly by running `make docs`.
* **Ensure that the test suite passes**, by running `make test`. You can also test individual backends by running
  `make test-backendname`. For example, if you only make changes corresponding to the Gaussian backend, you can run `make test-gaussian`.
* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.** The Strawberry Fields source code
  conforms to [PEP8 standards](https://www.python.org/dev/peps/pep-0008/). We check all of our code against
  [Pylint](https://www.pylint.org/). To lint modified files, simply install `pip install pylint`, and then from the source code
  directory, run `pylint strawberryfields/path/to/file.py`.

### Submitting the pull request
* When ready, submit your fork as a [pull request](https://help.github.com/articles/about-pull-requests) to the Strawberry
  Fields repository, filling out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). This template is added automatically
  to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible regarding the changes made/new features
  added/performance improvements. If including any bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on [Travis CI](https://travis-ci.org/XanaduAI/strawberryfields)
    to ensure that the all tests continue to pass.
  - Once the test suite is finished, a **code coverage report** will be generated on
    [Codecov](https://codecov.io/gh/XanaduAI/strawberryfields). This will calculate the percentage of Strawberry Fields
    covered by the test suite, to ensure that all new code additions are adequately tested.
  - Finally, the **code quality** is calculated by [Codacy](https://app.codacy.com/app/XanaduAI/strawberryfields/dashboard),
    to ensure all new code additions adhere to our code quality standards.

  Based on these reports, we may ask you to make small changes to your branch before merging the pull request into the master branch. Alternatively, you can also
  [grant us permission to make changes to your pull request branch](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

## I've made a contribution! Now how do I claim my Xanadu Swag?

* If you've contributed code to Strawberry Fields, we'll contact you shortly after merging your pull
  request to organise delivery of your Xanadu Swag™.

* If you've created a Strawberry Fields application, produced content using Strawberry Fields, or simply
  written an awesome Strawberry Fields tutorial, and we have linked it to our Gallery, you are eligible for some
  Xanadu Swag™. We will contact you to organise delivery of your swag.

If you haven't heard back from us, don't panic! Here at Xanadu we are working on projects on the cutting edge of
quantum technology, and sometimes get a little sidetracked. Simply send a reminder email to support@xanadu.ai regarding your
swag, and we'll get back to you ASAP.

:strawberry: Thank you for contributing to Strawberry Fields! :strawberry:

\- The Strawberry Fields team
