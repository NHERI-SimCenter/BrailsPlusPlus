.. _importer_example:

Importer
========

|app| is a modular python framework for creating workflows that create data for SimCenter applications. In order to allow applications to be built **w/o the need to hard code every possible case**, there is a class in |app| that will return the class given the class name. This is the **Importer** class. It's has one method, **get_class()**, which will return the python class with that given name from which an object can be instantiated. 

The following example shows a more typical python example. 

.. literalinclude:: classic.py
   :language: python
   :linenos:


The following example shows it re-written using the **Importer** class.

.. literalinclude:: importer.py
   :language: python
   :linenos:

.. note::

   #. The purpose of the **Importer** class is to allow applications to be developed for building workflows. Consider developing an application that would parse the following JSON input file to such an application. Using the **Importer** class, this application could be written without a bunch of if-else statments switching on the type. This works in |app| because all the classes that do work inherit from **ABC**, **Abstract Base Class** which is part of the python abc module.

   .. literalinclude:: workflow.json
      :language: json		       
      :linenos:

   #. An additional advantage of this approach is that as developers add new subclasses, they do not have to search through the code to see all the places that need the if-else statements modified for their classes to be used.

   #. In the construction of the **Importer** object the code will search through all directories of the brails code to find all the classes that exist in the code-base. As such, no changes are needed to the class by developers wishing to add their code.
      
   #. Most of the examples presented will use the Importer class as we use the examples to test that class.

	 
   
      

      


