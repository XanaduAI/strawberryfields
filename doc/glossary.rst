Glossary
========

Expressions that are grouped together are used synonymously in the Strawberry Fields documentation.


.. glossary::

   backend
      Executes quantum :term:`programs <program>`. Can be either a classical simulator or a
      quantum hardware device, local or remote.
      See: :class:`.BaseBackend`

   circuit
   quantum circuit
   program
      A sequence of quantum :term:`operations <operation>` acting on a quantum :term:`register`.
      Typically contains :term:`measurements <measurement>` that produce a stochastic classical output.
      Represents a quantum computation.
      See: :class:`.Program`

   compilation
      The process of converting a quantum :term:`program` into an :term:`equivalent program` that
      can be run on a specific :term:`device`. The compilation can fail if the device is incapable
      of running the program.

   device
      Something that can run a well-defined subclass of quantum :term:`programs <program>`.
      Each :term:`backend` corresponds to one or more devices.
      See: :class:`.DeviceSpecs`

   equivalent circuit
   equivalent program
      Two quantum :term:`circuits <circuit>` are equivalent iff they produce the same output
      probability distributions.

   engine
   frontend
      A class for executing quantum :term:`programs <program>` on a specific :term:`backend`.
      See: :class:`.BaseEngine`

   gate
   quantum gate
      A quantum :term:`operation` that changes the state of the :term:`register`.
      Analogous to a logic gate in a Boolean circuit.

   measurement
      A quantum :term:`operation` that returns classical information.

   operation
   quantum operation
      Something that acts on specific subsystems of the :term:`register`, changing its state and
      possibly returning classical information, e.g. a state preparation, :term:`gate`,
      :term:`measurement` etc.
      See: :class:`.Operation`

   register
   quantum register
      A quantum system (or a simulation of one) storing the state of the quantum computation.
      Typically consists of several subsystems (which in CV quantum computing are called modes).
