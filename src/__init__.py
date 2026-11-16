"""Allow this delivery's preprocessing and the data team's loader to coexist.

Only needed by this standalone delivery. When integrating into a team project,
copy src/preprocessing only; retain the team's existing src/__init__.py.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
