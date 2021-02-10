Quickstart
==========

Installation
------------

.. code-block:: bash

  pip install --user -e .[dev]

Contribute
------------

To contribute to this repo, please follow this steps:

1. Create or assigne yourself an open issue
2. Go to the develop branch and pull it: 

   .. code-block:: bash

      git checkout develop
      git pull

3. Create a new branch with the issue id:
   
   .. code-block:: bash
      
      git checkout -b #123-this-feature

4. Do as much as commit as needed (don't be afraid, there is never too much commits), then push your changes.
5. [*optional*] write a test in ``tests/`` dir
6. Once your implementation is finished:
   
   - format the code:
      
      .. code-block:: bash
         
         invoke format

   - check your code format:
   
      .. code-block:: bash
         
         invoke lint
   
   - test it:

      .. code-block:: bash
         
         invoke test

7. [*optional*] fix/commit/push the changes
8. Create a pull request

.. note:: Please go to ``tasks.py`` to see the details of *invoke* commands.