#!/bin/bash
sphinx-apidoc -o ./ ../pychemauth/;
make clean html;
make html;

# pip install pip-tools
# pip-compile requirements.in
