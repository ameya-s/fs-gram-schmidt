from distutils.core import setup

NAME = "fs_gram_schmidt"

DESCRIPTION = "Feature Selection Algorithm based on Gram Schmidt Orthogonalisation in Python"

KEYWORDS = "Feature Selection"

AUTHOR = "Ameya Dahale"

AUTHOR_EMAIL = "100ameya@gmail.com"

URL = "https://github.com/jundongl/fs-gram-schmidt"

VERSION = "1.0.0"


setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    keywords = KEYWORDS,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    packages =['fs_gram_schmidt', 'fs_gram_schmidt.function']
)
