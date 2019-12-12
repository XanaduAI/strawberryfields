Circuits
========

.. raw:: html

    <style>
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
    </style>


.. toctree::
    :maxdepth: 1

    circuits/glossary

Creating a quantum program
--------------------------

To construct a photonic quantum circuit in Strawberry Fields, a :class:`~.Program` object
must be created, and operations applied.

.. code-block:: python3

    import strawberryfields as sf
    from strawberryfields import ops

    # create a 3 mode quantum program
    prog = sf.program(3)

    with prog.context as q:
        ops.Sgate(0.54) | q[0]
        ops.BSgate(0.43, 0.1) | (q[1], q[2])

Here, we are creating a 3 mode program, and applying a :class:`~.Sgate` and
a :class:`~.BSgate` to various modes. Constructing quantum circuits always follows
the same structure as the above example; in particular,

* A ``with`` statement is used to populate the program with quantum operations.

* ``Program.context`` is used within the ``with`` statement to return ``q``,
  a representation of the quantum registers (qumodes or modes).

* Quantum operations are applied to the modes of the program using the syntax
  
  .. code-block:: python3

      ops.GateName(arg1, arg2) | (q[i], q[j], ...)


Simulating your program
-----------------------


Compilation
-----------


Glossary
--------

