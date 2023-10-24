from distutils.core import setup

setup(name='casados_integrators',
   version='0.1',
   python_requires='>=3.7',
   description='A CasADi Python wrapper for the acados integrators',
   author='Jonathan Frey',
   use_scm_version={
     "fallback_version": "0.1-local",
     "root": "../..",
     "relative_to": __file__
   },
   license='BSD 2-clause',
   include_package_data = True,
   setup_requires=['setuptools_scm'],
   install_requires=[
      'numpy',
      'scipy',
      'casadi<3.6',
      'matplotlib',
   ]
)