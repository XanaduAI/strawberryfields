# BSD 3-Clause License

# Copyright (c) 2017, Pytorch contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
import re
import os

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


GALLERY_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-category="{tags}" tooltip="{tooltip}">

.. only:: html

    .. figure:: {thumbnail}

        {description}

.. raw:: html

    </div>
"""


class CustomGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.

    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.

    Example usage:

    .. customgalleryitem::
        :tooltip: I am writing this tutorial to focus specifically on NLP for people who have never written code in any deep learning framework
        :figure: /_static/img/thumbnails/babel.jpg
        :description: :doc:`/beginner/deep_learning_nlp_tutorial`

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'tooltip': directives.unchanged,
                   'figure': directives.unchanged,
                   'description': directives.unchanged,
                   'tags': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'tooltip' in self.options:
                tooltip = self.options['tooltip'][:195]
            else:
                raise ValueError('tooltip not found')

            tags = ""
            if 'tags' in self.options:
                tags = self.options['tags']

            if 'figure' in self.options:
                env = self.state.document.settings.env
                rel_figname, figname = env.relfn2path(self.options['figure'])
                thumbnail = "/" + rel_figname
            else:
                thumbnail = '/_static/code.png'

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not doc found')

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(tooltip=tooltip,
                                                thumbnail=thumbnail,
                                                description=description,
                                                tags=tags)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


DETAILS_TEMPLATE = """
.. raw:: html

    <a class="details-header collapse-header" data-toggle="collapse" href="#details" aria-expanded="false" aria-controls="details">
        <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> {title}
        </h2>
    </a>
    <div class="collapse" id="details">

{content}

.. raw:: html

    </div>
"""


class DetailsDirective(Directive):
    """Create a collapsed details section in the documentation."""
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = False
    option_spec = {'name': directives.unchanged}
    has_content = True
    add_index = False

    def run(self):
        name = self.options.get("name", "Details and conventions")
        rst = DETAILS_TEMPLATE.format(title=name, content="\n".join(self.content))
        string_list = StringList(rst.split('\n'))
        node = nodes.tbody()
        self.state.nested_parse(string_list, self.content_offset, node)
        return [node]
