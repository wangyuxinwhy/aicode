.PHONY: style
style:
	blue . --exclude dataset
	isort . --skip dataset
