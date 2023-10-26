import os
import codecs

from setuptools import setup, find_packages

def readme():
    with codecs.open('README.rst', encoding='utf-8-sig') as f:
        return f.read()

version_file= os.path.join('mlscorecheck', '_version.py')
__version__= "0.0.0"
with open(version_file) as f:
    exec(f.read())

DISTNAME= 'mlscorecheck'
DESCRIPTION= 'ML score check: checking the validity of machine learning and computer vision scores'
LONG_DESCRIPTION= readme()
LONG_DESCRIPTION_CONTENT_TYPE='text/x-rst'
MAINTAINER= 'Gyorgy Kovacs'
MAINTAINER_EMAIL= 'gyuriofkovacs@gmail.com'
URL= 'https://github.com/gykovacs/mlscorecheck'
LICENSE= 'MIT'
DOWNLOAD_URL= 'https://github.com/gykovacs/mlscorecheck'
VERSION= __version__
CLASSIFIERS= [  'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Software Development',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS']
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES= ['numpy', 'scipy', 'scikit-learn', 'pulp']
EXTRAS_REQUIRE= {'tests': ['pytest'],
                    'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'matplotlib', 'pandas', 'pulp']}
PYTHON_REQUIRES= '>=3.5'
PACKAGE_DIR= {'mlscorecheck': 'mlscorecheck'}
SETUP_REQUIRES=['setuptools>=41.0.1', 'wheel>=0.33.4', 'pytest-runner']
TESTS_REQUIRE=['pytest']

setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        python_requires=PYTHON_REQUIRES,
        setup_requires=SETUP_REQUIRES,
        tests_require=TESTS_REQUIRE,
        package_dir=PACKAGE_DIR,
        packages=find_packages(exclude=[]),
        package_data={'mlscorecheck': [os.path.join('individual', 'solutions.json'),
                                        os.path.join('scores', 'scores.json'),
                                        os.path.join('experiments', 'machine_learning', 'common_datasets.json'),
                                        os.path.join('experiments', 'machine_learning', 'sklearn.json'),
                                        os.path.join('experiments', 'ehg', 'tpehg.json'),
                                        os.path.join('experiments', 'retina', 'drive', 'drive_test_fov.json'),
                                        os.path.join('experiments', 'retina', 'drive', 'drive_test_no_fov.json'),
                                        os.path.join('experiments', 'retina', 'drive', 'drive_train_fov.json'),
                                        os.path.join('experiments', 'retina', 'drive', 'drive_train_no_fov.json'),
                                        os.path.join('experiments', 'retina', 'chase_db1', 'manual1.json'),
                                        os.path.join('experiments', 'retina', 'chase_db1', 'manual2.json'),
                                        os.path.join('experiments', 'retina', 'diaretdb0', 'diaretdb0.json'),
                                        os.path.join('experiments', 'retina', 'diaretdb1', 'diaretdb1.json'),
                                        os.path.join('experiments', 'retina', 'drishti_gs', 'drishti_gs_test.json'),
                                        os.path.join('experiments', 'retina', 'drishti_gs', 'drishti_gs_train.json'),
                                        os.path.join('experiments', 'retina', 'hrf', 'with_fov.json'),
                                        os.path.join('experiments', 'retina', 'hrf', 'without_fov.json'),
                                        os.path.join('experiments', 'retina', 'stare', 'ah.json'),
                                        os.path.join('experiments', 'retina', 'stare', 'vk.json'),
                                        os.path.join('experiments', 'skinlesion', 'isic2016', 'isic2016.json'),
                                        os.path.join('experiments', 'skinlesion', 'isic2017', 'isic2017.json'),
                                        os.path.join('experiments', 'skinlesion', 'isic2017', 'isic2017m.json'),
                                        os.path.join('experiments', 'skinlesion', 'isic2017', 'isic2017sk.json'),]},
        include_package_data=True)
