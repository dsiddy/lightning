"""Boilerplate for publishing Python packages.

See <https://packaging.python.org/guides/distributing-packages-using-setuptools/> for more information.  # noqa: E501
"""


import setuptools


setuptools.setup(
    name='lightning',
    version='0.0.1',
    description='',
    # long_description='',
    url='https://www.blacksuncollective.net',
    # download_url='',
    author='Dan Siddoway',
    author_email='dan@blacksuncollective.net',
    # maintainer='',
    # maintainer_email='',
    license='UNLICENSED',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
    ],
    # keywords='',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'opencv-python',
    ],
    # platforms=[
    #     '',
    # ],
    # entry_points={
    #     '': [
    #         '',
    #     ],
    # }
)
