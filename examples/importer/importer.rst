.. _importer_example:

Importer
========

|app| is a modular Python framework for creating workflows that create data for SimCenter applications. To **avoid hard-coding every possible case**, |app| includes a utility class that can return a class object based on its name — the Importer class.

The ``Importer`` class has a single method, ``get_class()``, which takes a class name as input and returns the corresponding Python class, allowing you to instantiate objects dynamically.

The following example demonstrates a typical Python usage scenario:

.. literalinclude:: classic.py
   :language: python
   :linenos:


The following example shows it re-written using the ``Importer`` class.

.. literalinclude:: importer.py
   :language: python
   :linenos:

To run, for example, the **importer.py** script for Berkeley, CA, you would enter the following command in a terminal window:

.. code-block::
      
   python3 importer.py USA_FootprintScraper "Berkeley, CA"


and the application  would produce:

.. literalinclude:: output.txt
   :linenos:   
    
.. note::

   #. #. The purpose of the ``Importer`` class is to enable the development of applications for building workflows.  For example, imagine creating an application that parses the following JSON input file. By using the ``Importer`` class, the application can be written without lengthy ``if-else`` statements that switch on the type.  This is possible in |app| because all the classes that perform work inherit from :py:class:`abc.ABC` (*Abstract Base Class*), which is part of Python’s :py:mod:`abc` module.

   .. literalinclude:: workflow.json
      :language: json		       
      :linenos:

   #. An additional benefit of this approach is that when developers add new subclasses, they do not need to search the codebase to locate and update all the if-else statements required for their classes to function. The process is fully automated.

   #. When constructing a ``Importer`` object, the code automatically scans all directories within the |app| codebase to identify every available class. This means that developers can integrate their new code without making any modifications to the existing ``Importer`` class.

   #. Most of the provided examples will make use of the ``Importer`` class, both to illustrate its functionality and to serve as a means of testing its behavior.


	 
   
      

      


