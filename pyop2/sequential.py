# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012-2016, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""OP2 sequential backend."""

from copy import deepcopy
import os

import ctypes

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2.base import par_loop                          # noqa: F401
from pyop2.base import READ, WRITE, RW, INC, MIN, MAX    # noqa: F401
from pyop2.base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS, ALL
from pyop2.base import Map, MixedMap, DecoratedMap, Sparsity, Halo  # noqa: F401
from pyop2.base import Set, ExtrudedSet, MixedSet, Subset  # noqa: F401
from pyop2.base import DatView                           # noqa: F401
from pyop2.petsc_base import DataSet, MixedDataSet       # noqa: F401
from pyop2.petsc_base import Global, GlobalDataSet       # noqa: F401
from pyop2.petsc_base import Dat, MixedDat, Mat          # noqa: F401
from pyop2.exceptions import NameTypeError  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region
from pyop2.utils import UniqueNameGenerator, as_tuple, cached_property, get_petsc_dir
from pyop2.wrapper import DirectLayerAccess, IncrementalLayerLoop, wrapper_Arg


import coffee.system
from coffee.plan import ASTKernel


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        ast_handler = ASTKernel(ast, self._include_dirs)
        ast_handler.plan_cpu(self._opts)
        return ast_handler.gencode()


class Arg(base.Arg):

    def wrapper_args(self):
        # TODO: Use cache key to calculate types.
        c_typenames = []
        types = []
        values = []
        if self._is_mat:
            c_typenames.append("Mat")
            types.append(self.data._argtype)
            values.append(self.data.handle.handle)
        else:
            for d in self.data:
                c_typenames.append(self.ctype)
                types.append(d._argtype)
                # Cannot access a property of the Dat or we will force
                # evaluation of the trace
                values.append(d._data.ctypes.data)
        if self._is_indirect or self._is_mat:
            maps = as_tuple(self.map, Map)
            for map in maps:
                for m in map:
                    c_typenames.append("int")
                    types.append(m._argtype)
                    values.append(m._values.ctypes.data)
        return c_typenames, types, values


class JITModule(base.JITModule):

    def generate_wrapper(self):
        is_facet = (self._iteration_region == ON_INTERIOR_FACETS)

        wrapper_args = []

        unique_name = UniqueNameGenerator()
        start_layer = None

        if isinstance(self._itspace._iterset, Subset):
            wrapper_args.append("int *ssinds")
        if self._itspace._extruded:
            wrapper_args += ["int nlayers"]

            region = self._iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                start_layer = 0
                end_layer = 1
            elif region is ON_TOP:
                start_layer = "(nlayers - 1)"
                end_layer = "nlayers"
            elif region is ON_INTERIOR_FACETS:
                start_layer = 0
                end_layer = "(nlayers - 1)"
            else:
                start_layer = 0
                end_layer = "nlayers"

        def loop_body(index_name):
            kernel_args = []
            inits = []
            init_layers = []
            writebacks = []
            post_writebacks = []

            for i, arg in enumerate(self._args):
                prefix = 'arg{0}_'.format(i)
                namer = lambda sn: unique_name(prefix + sn)

                c_typenames, _1, _2 = arg.wrapper_args()

                arg_names = []
                for j, typ in enumerate(c_typenames):
                    name = namer(str(j))
                    arg_names.append(name)
                    wrapper_args.append("{typ} *{name}".format(typ=typ, name=name))  # ugh, side effect

                col_name = 'j_0' if self._itspace._extruded else None
                # layer_access = DirectLayerAccess(col_name)
                layer_access = IncrementalLayerLoop(start_layer, namer)
                arg_wrapper, kernel_arg = wrapper_Arg(arg.cache_key, arg_names, index_name, col_name, namer, layer_access, is_facet=is_facet)
                inits.extend(arg_wrapper.init)
                init_layers.extend(arg_wrapper.init_layer)
                writebacks.extend(arg_wrapper.writeback)
                post_writebacks.extend(arg_wrapper.post_writeback)
                kernel_args.append(kernel_arg)

            return inits, init_layers + [self._kernel.name + '(' + ', '.join(kernel_args) + ');'] + writebacks + post_writebacks

        base_inits, body3 = loop_body('i')
        if self._itspace._extruded and any(arg._is_indirect or arg._is_mat for arg in self._args):
            body2 = base_inits[:]
            body2.append("for (int {0} = {1}; {0} < {2}; {0}++) {{".format('j_0', start_layer, end_layer))
            body2.extend('\t' + line for line in body3)
            body2.append("}")
        else:
            body2 = base_inits + body3

        if isinstance(self._itspace._iterset, Subset):
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('n')]
            body.append('\t' + "int i = ssinds[n];")
            body.extend('\t' + line for line in body2)
            body.append("}")
        else:
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('i')]
            body.extend('\t' + line for line in body2)
            body.append("}")

        return ''.join(["void ", self._wrapper_name,
                        "(int start, int end, ",
                        ', '.join(wrapper_args),
                        ")\n{\n",
                        '\n'.join('\t' + line for line in body),
                        "\n}\n"])

    _cppargs = []
    _libraries = []
    _system_headers = []
    _extension = 'c'

    def __init__(self, kernel, itspace, *args, **kwargs):
        """
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self.comm = itspace.comm
        self._kernel = kernel
        self._fun = None
        self._code_dict = None
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = deepcopy(type(self)._cppargs)
        self._libraries = deepcopy(type(self)._libraries)
        self._system_headers = deepcopy(type(self)._system_headers)
        self.set_argtypes(itspace.iterset, *args)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def code_to_compile(self):
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = coffee.system.compiler
        externc_open = '' if not self._kernel._cpp else 'extern "C" {'
        externc_close = '' if not self._kernel._cpp else '}'
        headers = "\n".join([compiler.get('vect_header', "")])
        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %(header)s
            %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code(),
                   'header': headers}
        else:
            kernel_code = """
%(header)s
%(code)s
            """ % {'code': self._kernel.code(),
                   'header': headers}
        code_to_compile = self.generate_wrapper()

        code_to_compile = """
#include <petsc.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
%(sys_headers)s

%(kernel)s

%(externc_open)s
%(wrapper)s
%(externc_close)s
        """ % {'kernel': kernel_code,
               'wrapper': code_to_compile,
               'externc_open': externc_open,
               'externc_close': externc_close,
               'sys_headers': '\n'.join(self._kernel._headers + self._system_headers)}

        self._dump_generated_code(code_to_compile)
        return code_to_compile

    @collective
    def compile(self):
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        # If we weren't in the cache we /must/ have arguments
        compiler = coffee.system.compiler
        extension = self._extension
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        if compiler:
            cppargs += [compiler[coffee.system.isa['inst_set']]]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        if self._kernel._cpp:
            extension = "cpp"
        self._fun = compilation.load(self.code_to_compile,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     argtypes=self._argtypes,
                                     restype=ctypes.c_int,
                                     compiler=compiler.get('name'),
                                     comm=self.comm)
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._itspace
        del self._direct
        return self._fun

    def set_argtypes(self, iterset, *args):
        index_type = as_ctypes(IntType)
        argtypes = [index_type, index_type]
        if iterset.masks is not None:
            argtypes.append(iterset.masks._argtype)
        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        if iterset._extruded:
            argtypes.append(ctypes.c_int)

        for arg in args:
            _1, types, _2 = arg.wrapper_args()
            argtypes.extend(types)

        self._argtypes = argtypes


class ParLoop(petsc_base.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = []

        if isinstance(iterset, Subset):
            arglist.append(iterset._indices.ctypes.data)
        if iterset._extruded:
            arglist.append(iterset.layers - 1)

        for arg in args:
            _1, _2, values = arg.wrapper_args()
            arglist.extend(values)

        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.it_space, *self.args,
                         direct=self.is_direct, iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @collective
    def _compute(self, part, fun, *arglist):
        with timed_region("ParLoop%s" % self.iterset.name):
            fun(part.offset, part.offset + part.size, *arglist)
            self.log_flops(self.num_flops * part.size)


def generate_cell_wrapper(itspace, args, forward_args=(), kernel_name=None, wrapper_name=None):
    wrapper_fargs = ["{1} farg{0}".format(i, arg) for i, arg in enumerate(forward_args)]
    kernel_fargs = ["farg{0}".format(i) for i in range(len(forward_args))]

    unique_name = UniqueNameGenerator()
    wrapper_args = wrapper_fargs[:]

    def loop_body(index_name):
        kernel_args = kernel_fargs[:]
        inits = []
        writebacks = []

        for i, arg in enumerate(args):
            prefix = 'arg{0}_'.format(i)
            namer = lambda sn: unique_name(prefix + sn)

            c_typenames, _1, _2 = arg.wrapper_args()

            arg_names = []
            for j, typ in enumerate(c_typenames):
                name = namer(str(j))
                arg_names.append(name)
                wrapper_args.append("{typ} *{name}".format(typ=typ, name=name))  # ugh, side effect

            col_name = 'j_0' if itspace._extruded else None
            layer_access = DirectLayerAccess(col_name)
            arg_wrapper, kernel_arg = wrapper_Arg(arg.cache_key, arg_names, index_name, col_name, namer, layer_access)
            inits.extend(arg_wrapper.init)
            inits.extend(arg_wrapper.init_layer)
            writebacks.extend(arg_wrapper.writeback)
            kernel_args.append(kernel_arg)

        return inits + [kernel_name + '(' + ', '.join(kernel_args) + ');'] + writebacks

    body = loop_body('i')

    if itspace._extruded:
        body = ["int i = cell / nlayers;", "int j_0 = cell % nlayers;"] + body
        wrapper_args.append("int nlayers")
    else:
        body.insert(0, "int i = cell;")
    wrapper_args.append("int cell")

    return ''.join(["void ", wrapper_name, "(",
                    ', '.join(wrapper_args),
                    ")\n{\n",
                    '\n'.join('\t' + line for line in body),
                    "\n}\n"])
