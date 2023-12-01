import io
import parser

from pycparser import CParser

filename = 'Benchmarks/Linear/c/1.c'

with io.open(filename) as f:
    text = f.read()
    print(text)

parser = CParser()

parser.parse(text, filename)
print(parser)

def myparse(self, text, filename='', debug=False):
    """ Parses C code and returns an AST.

        text:
            A string containing the C source code

        filename:
            Name of the file being parsed (for meaningful
            error messages)

        debug:
            Debug flag to YACC
    """
    self.clex.filename = filename
    self.clex.reset_lineno()
    self._scope_stack = [dict()]
    self._last_yielded_token = None
    return self.cparser.parse(
        input=text,
        lexer=self.clex,
        debug=debug)
