INFO 05-12 21:12:14 \[\_\_init\_\_.py:239\] Automatically detected
platform cuda. Help on package openclio:

  - NAME  
    openclio

  - PACKAGE CONTENTS  
    faissKMeans openclio opencliotypes prompts utils writeOutput

  - DATA
    
      - Callable = typing.Callable  
        Deprecated alias to collections.abc.Callable.
        
        Callable\[\[int\], str\] signifies a function that takes a
        single parameter of type int and returns a str.
        
        The subscription syntax must always be used with exactly two
        values: the argument list and the return type. The argument list
        must be a list of types, a ParamSpec, Concatenate or ellipsis.
        The return type must be a single type.
        
        There is no syntax to indicate optional or keyword arguments;
        such function types are rarely used as callback types.
    
      - Dict = typing.Dict  
        A generic version of dict.
    
    EmbeddingArray = numpy.ndarray\[tuple\[int, ...\],
    numpy.dtype\[numpy.floa... List = typing.List A generic version of
    list.
    
      - Optional = typing.Optional  
        Optional\[X\] is equivalent to Union\[X, None\].
    
      - Tuple = typing.Tuple  
        Deprecated alias to builtins.tuple.
        
        Tuple\[X, Y\] is the cross-product type of X and Y.
        
        Example: Tuple\[T1, T2\] is a tuple of two elements
        corresponding to type variables T1 and T2. Tuple\[int, float,
        str\] is a tuple of an int, a float and a string.
        
        To specify a variable-length tuple of homogeneous type, use
        Tuple\[T, ...\].
    
      - TypeAlias = typing.TypeAlias  
        Special form for marking type aliases.
        
        Use TypeAlias to indicate that an assignment should be
        recognized as a proper type alias definition by type checkers.
        
        For example:
        
            Predicate: TypeAlias = Callable[..., bool]
        
        It's invalid when used anywhere except as in the example above.
    
      - Union = typing.Union  
        Union type; Union\[X, Y\] means either X or Y.
        
        On Python 3.10 and higher, the | operator can also be used to
        denote unions; X | Y means the same thing to the type checker as
        Union\[X, Y\].
        
        To define a union, use e.g. Union\[int, str\]. Details: - The
        arguments must be types and there must be at least one. - None
        as an argument is a special case and is replaced by type(None).
        - Unions of unions are flattened, e.g.:
        
            assert Union[Union[int, str], float] == Union[int, str, float]
        
          - Unions of a single argument vanish, e.g.:
            
                assert Union[int] == int  # The constructor actually returns int
        
          - Redundant arguments are skipped, e.g.:
            
                assert Union[int, str, int] == Union[int, str]
        
          - When comparing unions, the argument order is ignored, e.g.:
            
                assert Union[int, str] == Union[str, int]
        
          - You cannot subclass or instantiate a union.
        
          - You can use Optional\[X\] as a shorthand for Union\[X,
            None\].
    
    cachedTokenizer = None genericSummaryFacets =
    \[Facet(name='Summary', question='', prefill='',... isWindows =
    False mainFacets = \[Facet(name='Request', question='What is the
    user...d a n... replacementCache = {}

  - FILE  
    /workspace/OpenClio/openclio/\_\_init\_\_.py
