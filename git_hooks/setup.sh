#!/bin/bash
ln -s $(pwd)/post-commit $(git rev-parse --git-dir)/hooks

