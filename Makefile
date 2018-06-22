all: install

install:
	python setup.py install

test:
	python -m pytest riconv
