Migration guides
================

The migration guides below offers guidance for dealing with major breaking
changes between Strawberry Fields releases.

Version 0.20.0
--------------

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

The following table shows the equivalent ``xcc`` (v0.20.0) command for
each ``sf`` (v0.19.0) command:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - **Strawberry Fields v0.19.0**
     - **Strawberry Fields v0.20.0**
   * - ``sf configure --token "foo"``
     - ``xcc config set REFRESH_TOKEN "foo"``
   * - ``sf --ping``
     - ``xcc ping``
   * - ``sf run "foo.xbb"``
     - ``xcc job submit --name "bar" --target "X8_01" --circuit "$(cat foo.xbb)"``

.. note::

  To submit a job using Windows PowerShell, simply replace ``cat foo.xbb`` with  ``Get-Content foo.xbb -Raw``.