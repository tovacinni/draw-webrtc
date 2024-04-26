from setuptools import setup

PACKAGE_NAME = 'drawrtc'
DESCRIPTION = 'demo for WebRTC with LCMLoRA'
URL = 'https://github.com/tovacinni/drawrtc'
AUTHOR = 'Towaki Takikawa'
LICENSE = 'MIT License'
VERSION = '0.0.0'

if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='>=3.9',

        # Package info
        packages=['drawrtc'],
        include_package_data=True,
        zip_safe=True,
    )
