#!/bin/bash
sphinx-apidoc -o ./ ../pychemauth/;
make clean html;
make html;
