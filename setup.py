from setuptools import setup, find_packages

setup(
    name='drawing_intelligence',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.10',
    install_requires=[
        'pyyaml>=6.0',
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'pillow>=10.0.0',
        'PyMuPDF>=1.23.0',
        'paddleocr>=2.7.0',
        'easyocr>=1.7.0',
        'spacy>=3.7.0',
        'torch>=2.0.0',
        'ultralytics>=8.0.0',
        'openai>=1.3.0',
        'anthropic>=0.7.0',
        'google-generativeai>=0.3.0',
        'python-dotenv>=1.0.0',
        'tqdm>=4.66.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ]
    },
)