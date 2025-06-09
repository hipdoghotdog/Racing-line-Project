from setuptools import setup

package_name = 'f1tenth_stanley_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@domain.com',
    description='Stanley controller for Ackermann driving in ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'stanley_node = f1tenth_stanley_controller.stanley_node:main',
        ],
    },
)