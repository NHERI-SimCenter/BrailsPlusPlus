"""
Brails Exceptions
"""


class BrailsError(Exception):
    """
    Custom exception for specific error cases in our application.
    """

    def __init__(self, message="An error occurred in the Brails application"):
        self.message = message
        super().__init__(self.message)


class NotFoundError(BrailsError):
    """
    Exception to be raised when something is not found.
    Note: If a file is not found, use the built-in FileNotFound
    instead.

    """

    def __init__(self, type_of_thing, name, where=None, append=None):
        """
        Initializes NotFoundError

        Args:
            type_of_thing (str): What the missing thing is, i.e.,
                class, key, parameter, etc.
            name (str): The name of the missing thing.
            where (str): Where the thing was expected to be found.
            append (str): Additional message to be appended.

        Example:
            >>> raise NotFoundError(
            ...     'class',
            ...     'ConvolutionFilter',
            ...     where='configuration_file'
            ... )
            NotFoundError: CLASS ConvolutionFilter is not found in configuration_file.

            Including additional information for context:
            >>> raise NotFoundError(
            ...     'class',
            ...     'ConvolutionFilter',
            ...     where='configuration_file',
            ...     append='Check config and retry.'
            ... )
            NotFoundError: CLASS ConvolutionFilter is not found in configuration_file.
            Check config and retry.
        
        """
        if where:
            add = f' in {where}'
        else:
            add = ''
        self.message = f'{type_of_thing.capitalize()} {name} is not found{add}.'
        if append:
            self.message += '\n' + append
        super().__init__(self.message)
