Recommender System
===================


This is a simple recommender system that uses [lightfm](https://lyst.github.io/lightfm/docs/home.html) library to generate recommendation from [movie-lens 100k](https://grouplens.org/datasets/movielens/100k/) dataset. This program is almost similar to the one in the documentation. I have to write few more methods to train it on movielens-1M dataset.


External Dependencies
-------------
> - numpy
> - lightfm

To install the dependencies use pip or anaconda/miniconda.
> **Note:**  *lightfm*  works out of the box after grabbing by pip on linux. I found difficulty to make it work on windows since no wheel was available on the pip version of the module. There are some unofficial wheels for lightfm, u may make it work using those. Or download the latest latest VC++ complier. To know more about this issue click [here](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/).


