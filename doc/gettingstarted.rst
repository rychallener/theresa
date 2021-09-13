Getting Started
===============

There is an example application to the ultra-hot Jupiter WASP-76b
included in the ThERESA installation. First, download the required
ExoTransmit opacities. There is a script included to do this,
found in the example directory. Assuming you are in the top level
directory, do:

.. code-block:: bash

   cd example
   ./fetchopac.sh

This will populate example/opac with the necessary files.  Then,
ThERESA can be run from the command line with the following commands:

.. code-block:: bash

   cd ../theresa
   theresa.py 2d wasp76-example.cfg
   theresa.py 3d wasp76-example.cfg

The available configuration options are discussed in more detail in
:doc:`configuration`.
