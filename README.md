# S14 Amazonas – openEO Workflow
![alt text](https://raw.githubusercontent.com/gisat/s14amazonas/master/amazonas_workflow.jpg)
Script repositorty for ESA S14amazonas project.

This branch contains the openEO-based implementation of the Sentinel-1 forest cover change / deforestation detection workflow developed under the Sentinel-1 for Science Amazonas (S14 Amazonas) project.

The focus of this branch is on scalable, reproducible execution of the algorithm using the openEO API, including post-processing, and threshold-based change detection.

Overview
--------
The workflow detects tree cover change (forest → non-forest) using multi-temporal Sentinel-1 SAR backscatter. It is designed to operate on large spatial and temporal extents (Amazon basin scale) and to be deployable on openEO-compliant backends.

1.  [Installation](#installation)
2.  [Dependencies](#dependencies)
3.  [Documentation](#documentation)
4.  [Hints](#hints)

Installation
------------

It is strongly recommended to use [ANACONDA](https://www.anaconda.com/distribution/) and 
[git](https://git-scm.com/downloads).

Dependencies
-----------

   - Python version 3.7 or higher
   - `numpy`, `scipy`, `gdal`, `osgeo`, `ogr`, `osr`, `pandas`, `openeo`


Main Components
-------------
1. openEO Detection Workflows

openeo_treecoverchange_detection.py
Core openEO process graph defining the tree cover change detection logic:
Loads Sentinel-1 backscatter data
Builds temporal stacks
Computes statistical features
Applies change-detection rules
Produces intermediate and final change layers

openeo_jobmanager_detection.py
Handles:
Batch job submission
Temporal and spatial parameterisation
Monitoring and management of long-running openEO jobs

2. 
Hints
-----

##### Python language:
- [A byte of Python:](http://python.swaroopch.com/) A good introductory tutorial to Python.
- [Introduction to Python Programming:](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) It covers basics of Python programming, including data types, control structures, and functions.
- [Python for Everybody:](https://www.coursera.org/learn/python-for-applied-data-science-ai) It covers basics of programming and data analysis.
