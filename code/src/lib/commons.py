class AbstractAttribute:
    """An abstract class attribute.

    Use this instead of an abstract property when you don't expect the
    attribute to be implemented by a property.

    """
    __isabstractmethod__ = True

    def __init__(self, doc=""):
        self.__doc__ = doc

    def __get__(self, obj, cls):
        return self


# def parse_kwargs(key, **kwargs):
#     try:
#         return kwargs[key]
#     except KeyError:
#         return None
