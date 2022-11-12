from setuptools import setup

setup(
    name='SimpleClick',
    version='',
    packages=['isegm', 'isegm.data', 'isegm.data.datasets', 'isegm.model', 'isegm.model.modeling',
              'isegm.model.modeling.hrformer_helper', 'isegm.model.modeling.hrformer_helper.hrt',
              'isegm.model.modeling.hrformer_helper.hrt.modules', 'isegm.model.modeling.transformer_helper',
              'isegm.model.modeling.swin_transformer_helper', 'isegm.utils', 'isegm.utils.cython', 'isegm.engine',
              'isegm.inference', 'isegm.inference.predictors', 'isegm.inference.transforms'],
    url='',
    license='',
    author='steven.tobias',
    author_email='',
    description=''
)
