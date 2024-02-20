#!/bin/bash
poetry run pytest --cov-report html --cov-config=.coveragerc --cov sigmaepsilon.solid.material
