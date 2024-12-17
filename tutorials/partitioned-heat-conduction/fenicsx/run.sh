#!/bin/sh
set -e -u

while getopts ":dn" opt; do
  case ${opt} in
  d)
    python3 heat.py Dirichlet --error-tol 10e-3
    ;;
  n)
    python3 heat.py Neumann --error-tol 10e-3
    ;;
  \?)
    echo "Usage: cmd [-d] [-n]"
    ;;
  esac
done
