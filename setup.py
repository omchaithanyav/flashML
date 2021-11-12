from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]


with open("README.md", "r") as fh:
  long_description = fh.read()


setup(
  name='flashML',
  version='0.0.1',
  description='AutoML tool',
  long_description=long_description + '\n\n' + open('CHANGELOG.txt').read(),
  long_description_content_type="text/markdown",
  url='https://github.com/omchaithanyav/flashML',
  author='Om Chaithanya V',
  author_email='vomchaithanya@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='AutoML', 
  packages=find_packages(include=["flashML*"]),
  install_requires=["NumPy>=1.16.2",
    "lightgbm>=2.3.1",
    "scipy>=1.4.1",
    "pandas>=1.1.4",
    "scikit-learn>=0.24",
    "catboost>=0.26",
    "optuna"],

  python_requires=">=3.6"
)