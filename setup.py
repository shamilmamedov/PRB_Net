from setuptools import setup, find_packages

NAME = "FEIN"
VERSION = "0.0.1"
AUTHORS = ""
MAINTAINER_EMAIL = ""
DESCRIPTION = "Leveraging rigid finite element discretizations forward kinematics for learning DLO dynamics"

setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=MAINTAINER_EMAIL,
    packages=find_packages()
)
