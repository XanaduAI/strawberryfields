PYTHON := $(shell which python3 2>/dev/null)
TESTRUNNER := -m pytest tests -p no:warnings
COVERAGE := --cov=strawberryfields --cov-report=html:coverage_html_report --cov-append

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install Strawberry Fields"
	@echo "  wheel              to build the Strawberry Fields wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite for entire codebase"
	@echo "  test-[component]   to run the test suite for frontend, fock, tf, or gaussian"
	@echo "  coverage           to generate a coverage report for entire codebase"
	@echo "  coverage-[backend] to generate a coverage report for frontend, fock, tf, or gaussian"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install Strawberry Fields you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	rm -rf strawberryfields/__pycache__
	rm -rf strawberryfields/backends/__pycache__
	rm -rf strawberryfields/backends/fockbackend/__pycache__
	rm -rf strawberryfields/backends/tfbackend/__pycache__
	rm -rf strawberryfields/backends/gaussianbackend/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/backend/__pycache__
	rm -rf tests/frontend/__pycache__
	rm -rf tests/integration/__pycache__
	rm -rf dist
	rm -rf build

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean

test: test-frontend test-gaussian test-fock test-tf batch-test-tf test-gbs

test-%:
	@echo "Testing $(subst test-,,$@) backend..."
	$(PYTHON) $(TESTRUNNER) -m $(subst test-,,"$@")

batch-test-%:
	@echo "Testing $(subst batch-test-,,$@) backend in batch mode..."
	export BATCHED=1 && $(PYTHON) $(TESTRUNNER) -m $(subst batch-test-,,"$@")

coverage: coverage-frontend coverage-gaussian coverage-fock coverage-tf batch-coverage-tf coverage-gbs

coverage-%:
	@echo "Generating coverage report for $(subst coverage-,,$@)..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE) -m $(subst coverage-,,"$@")

batch-coverage-%:
	@echo "Generating coverage report for $(subst batch-coverage-,,"$@") in batch mode..."
	export BATCHED=1 &&  $(TESTRUNNER) $(COVERAGE) -m $(subst batch-coverage-,,"$@")
