#!/bin/bash
git add *
git commit -m `date +%Y%m%d-%s`
git push -u origin master 
