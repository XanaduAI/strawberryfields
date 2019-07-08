Glossary
========

Expressions that are grouped together are used synonymously in the Strawberry Fields documentation.


.. glossary::

   backend
      Executes quantum :term:`programs <program>`. Can be either a classical simulator or a
      quantum hardware device, local or remote. Different backends can have different capabilities
      in terms of efficiency and which :term:`class of circuits <circuit class>` they can execute.
      See: :class:`.BaseBackend`

   circuit
   quantum circuit
   program
      A sequence of quantum :term:`operations <operation>` acting on a quantum :term:`register`.
      Typically contains :term:`measurements <measurement>` that produce a stochastic classical output.
      Represents a quantum computation.
      See: :class:`.Program`

   circuit class
   circuit family
      A well-defined subset of :term:`quantum circuits <circuit>`.
      Members of a given circuit class can potentially be evaluated
      on more than one :term:`backend`, and each backend has an associated circuit class it
      can execute.
      See: :class:`~.strawberryfields.circuitspecs.CircuitSpecs`

   compilation
      The process of converting a `source` quantum :term:`program` into an :term:`equivalent program`
      that belongs in the `target` :term:`circuit class`.
      The compilation can fail if the program cannot be made equivalent to the target circuit class.

   equivalent circuit
   equivalent program
      Two quantum :term:`circuits <circuit>` are equivalent iff they produce the same output
      probability distributions. Two equivalent circuits are not necessarily in the same
      :term:`circuit class`.

   engine
      A class for executing quantum :term:`programs <program>` on a specific :term:`backend`.
      See: :class:`.BaseEngine`

   frontend
      All components of Strawberry Fields that are user-facing. Includes the user interface
      (:term:`engine`, :term:`program`, :term:`operation`);
      excludes the :term:`backends <backend>`.
      See: :class:`.BaseEngine`, :class:`.Program`, :mod:`Operation <.ops>`

   gate
   quantum gate
      A quantum :term:`operation` that changes the state of the :term:`register`.
      Analogous to a logic gate in a Boolean circuit.

   measurement
      A quantum :term:`operation` that returns classical information.

   operation
   quantum operation
      Something that acts on specific subsystems of the :term:`register`, changing its state and
      possibly returning classical information, e.g., a state preparation, :term:`gate`,
      :term:`measurement`, etc.
      See: :mod:`Operation <.ops>`

   register
   quantum register
   system
   mode
   qumode
      A quantum system (or a simulation of one) storing the state of the quantum computation.
      Typically consists of several subsystems (which in CV quantum computing are called modes).

   state preparation
   preparation
      An :term:`operation` where a quantum system or subsystem is prepared in a known fixed state.
      Typically accomplished by applying a specific sequence of gates to a known
      reference state.
