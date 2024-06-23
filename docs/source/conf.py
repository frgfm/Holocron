# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, Path().cwd().parent.parent)


import holocron
from holocron import models

# -- Project information -----------------------------------------------------

master_doc = "index"
project = "holocron"
copyright = f"2019-{datetime.now().year}, François-Guillaume Fernandez"
author = "François-Guillaume Fernandez"

# The full version, including alpha/beta/rc tags
version = holocron.__version__
release = holocron.__version__ + "-git"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxemoji.sphinxemoji",  # cf. https://sphinxemojicodes.readthedocs.io/en/stable/
    "sphinx_copybutton",
    "recommonmark",
    "sphinx_markdown_tables",
]

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
pygments_dark_style = "monokai"
highlight_language = "python3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_title = "Holocron"
REPO_URL = "https://github.com/frgfm/holocron"
# html_logo = "_static/images/logo.png"
html_favicon = "_static/images/favicon.ico"
language = "en"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": REPO_URL,
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": REPO_URL,
    "source_branch": "main",
    "source_directory": "docs/source/",
    "sidebar_hide_name": False,
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- A patch that prevents Sphinx from cross-referencing ivar tags -------
# See http://stackoverflow.com/a/41184353/3343043

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docfields import TypedField


def patched_make_field(self, types, domain, items, **kw):
    # `kw` catches `env=None` needed for newer sphinx while maintaining
    #  backwards compatibility when passed along further down!

    # type: (list, unicode, tuple) -> nodes.field
    def handle_item(fieldarg, content):
        par = nodes.paragraph()
        par += addnodes.literal_strong("", fieldarg)  # Patch: this line added
        # par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
        #                           addnodes.literal_strong))
        if fieldarg in types:
            par += nodes.Text(" (")
            # NOTE: using .pop() here to prevent a single type node to be
            # inserted twice into the doctree, which leads to
            # inconsistencies later when references are resolved
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = "".join(n.astext() for n in fieldtype)
                typename = typename.replace("int", "python:int")
                typename = typename.replace("long", "python:long")
                typename = typename.replace("float", "python:float")
                typename = typename.replace("type", "python:type")
                par.extend(self.make_xrefs(self.typerolename, domain, typename, addnodes.literal_emphasis, **kw))
            else:
                par += fieldtype
            par += nodes.Text(")")
        par += nodes.Text(" -- ")
        par += content
        return par

    fieldname = nodes.field_name("", self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item("", handle_item(fieldarg, content))
    fieldbody = nodes.field_body("", bodynode)
    return nodes.field("", fieldname, fieldbody)


TypedField.make_field = patched_make_field


def inject_checkpoint_metadata(app, what, name, obj, options, lines):
    """This hook is used to generate docs for the models weights.
    Objects like ConvNeXt_Atto_Checkpoint are enums with fields, where each field is a Weight object.
    Enums aren't easily documented in Python so the solution we're going for is to:
    - add an autoclass directive in the model's builder docstring, e.g.
    ```
    .. autoclass:: holocron.models.ConvNeXt_Atto_Checkpoint
        :members:
    ```
    (see resnet.py for an example)
    - then this hook is called automatically when building the docs, and it generates the text that gets
      used within the autoclass directive.
    """
    if obj.__name__.endswith(("_Checkpoint")):
        if len(obj) == 0:
            lines[:] = ["There are no available pre-trained checkpoints."]
            return

        lines[:] = [
            "The model builder above accepts the following values as the ``checkpoint`` parameter.",
            f"``{obj.__name__}.DEFAULT`` is equivalent to ``{obj.DEFAULT}``",
        ]

        if obj.__doc__ != "An enumeration.":
            # We only show the custom enum doc if it was overriden. The default one from Python is "An enumeration"
            lines.append("")
            lines.append(obj.__doc__)

        lines.append("")

        for field in obj:
            lines += [f"**{obj.__name__}.{field.name}**:", ""]
            if field == obj.DEFAULT:
                lines += [f"Also available as ``{obj.__name__}.DEFAULT``."]
            lines += [""]

            table = []
            # Performance metrics
            table.append(("dataset", field.value.evaluation.dataset.value))
            for metric, val in field.value.evaluation.results.items():
                table.append((metric.value, str(val)))

            # Loading Meta
            meta = field.value.meta
            table.extend((
                ("url", f"`link <{meta.url}>`__"),
                ("sha256", meta.sha256[:16]),
                ("size", f"{meta.size / 1024**2:.1f}MB"),
                ("num_params", f"{meta.num_params / 1000000.0:.1f}M"),
            ))
            # Wrap the text
            max_visible = 3
            v = meta.categories
            v_sample = ", ".join(v[:max_visible])
            v = f"{v_sample}, ... ({len(v) - max_visible} omitted)" if len(v) > max_visible else v_sample
            table.append(("categories", str(v)))

            # How to use it
            meta = field.value.pre_processing
            for k, v in meta.__dict__.items():
                table.append((k, str(v)))

            # How to reproduce the training
            meta = field.value.recipe
            commit_str = str(None) if meta.commit is None else f"`{meta.commit[:7]} <{REPO_URL}/tree/{meta.commit}>`__"
            table.append(("commit", commit_str))
            # table.append(("Training args", meta.args))

            column_widths = ["60", "60"]
            " ".join(column_widths)

            table = tabulate(table, tablefmt="rst")
            lines += [".. rst-class:: table-checkpoints"]  # Custom CSS class, see custom_theme.css
            lines += [".. table::", ""]
            lines += textwrap.indent(table, " " * 4).split("\n")
            lines.append("")


def generate_checkpoint_table(module, table_name, metrics):
    checkpoints_endswith = "_Checkpoint"
    checkpoint_enums = [getattr(module, name) for name in dir(module) if name.endswith(checkpoints_endswith)]
    # Unpack the enum
    [c for checkpoint_enum in checkpoint_enums for c in checkpoint_enum]

    metrics_keys, metrics_names = zip(*metrics)
    column_names = ["Checkpoint", *metrics_names, "Params", "Size (MB)"]  # Final column order
    column_names = [f"**{name}**" for name in column_names]  # Add bold

    content = []
    for ckpt_enum in checkpoint_enums:
        for ckpt in ckpt_enum:
            c = ckpt.value
            row = [
                f":class:`{ckpt_enum.__name__}.{ckpt.name} <{ckpt_enum.__name__}>`",
                *(f"{c.evaluation.results[metric]:.2%}" for metric in metrics_keys),
                f"{c.meta.num_params / 1e6:.1f}M",
                f"{round(c.meta.size / 1024**2, 1):.1f}",
            ]

            content.append(row)

    column_widths = ["100"] + ["20"] * (len(metrics_names) + 2)
    widths_table = " ".join(column_widths)

    table = tabulate(content, headers=column_names, tablefmt="rst")

    generated_dir = Path("generated")
    generated_dir.mkdir(exist_ok=True)
    with Path(generated_dir / f"{table_name}_table.rst").open("w+") as table_file:
        table_file.write(".. rst-class:: table-checkpoints\n")  # Custom CSS class, see custom_theme.css
        table_file.write(".. table::\n")
        table_file.write(f"    :widths: {widths_table} \n\n")
        table_file.write(f"{textwrap.indent(table, ' ' * 4)}\n\n")


generate_checkpoint_table(
    module=models,
    table_name="classification",
    metrics=[("top1-accuracy", "Acc@1"), ("top5-accuracy", "Acc@5")],
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Add googleanalytics id
# ref: https://github.com/orenhecht/googleanalytics/blob/master/sphinxcontrib/googleanalytics.py
def add_ga_javascript(app, pagename, templatename, context, doctree):
    metatags = context.get("metatags", "")
    metatags += """
    <!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={0}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{0}');
</script>
    """.format(app.config.googleanalytics_id)
    context["metatags"] = metatags


def setup(app):
    app.add_config_value("googleanalytics_id", "UA-148140560-2", "html")
    app.add_css_file("css/custom_theme.css")
    app.add_js_file("js/custom.js")
    app.connect("html-page-context", add_ga_javascript)
    app.connect("autodoc-process-docstring", inject_checkpoint_metadata)
