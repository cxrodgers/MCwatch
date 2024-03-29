"""Interacting with the mouse cloud database

Importing this module triggers the following
- Adds the mouse-cloud github repository to the path
  This means that the currently active branch will be in use.
- Sets DJANGO_SETTINGS_MODULE and calls django.setup()
  This will call code in settings.py and local_settings.py in the
  mouse-cloud project.

The mouse-cloud database can then be accessed via the django ORM.

Code for analyzing behavior, neural, and whisker data is contained in
those submodules. That code can still be loaded even if no django is
available. Import guards are used here to silently continue if the django
project or its dependencies aren't present.

If `runner.models` cannot be imported, debug like this:

import os, sys, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mouse2.settings")
sys.path.append(os.path.expanduser('~/dev/mouse-cloud'))
django.setup()
"""
from __future__ import absolute_import

import os
import sys
try:
    import django
    NO_DJANGO = False
except ImportError:
    NO_DJANGO = True
    pass

## Load the interface with the django database
def _setup_django():
    """Imports django settings from mouse2 project
    
    http://stackoverflow.com/questions/8047204/django-script-to-access-model-objects-without-using-manage-py-shell
    
    In the long run it may be better to rewrite this using sqlalchemy
    and direct URLs to the database. For one thing, we are dependent on
    the branch currently in place on the Dropbox.
    """
    # Test if this trick has already been used
    dsm_val = os.environ.get('DJANGO_SETTINGS_MODULE')    
    if dsm_val == 'mouse2.settings':
        # Seems to already have been done. Maybe we are running from manage.py
        #~ print "warning: DJANGO_SETTINGS_MODULE already set"
        return
    if dsm_val is not None:
        # Already set to some other module. This will not work
        raise ValueError("DJANGO_SETTINGS_MODULE already set to %s" % dsm_val)
    
    # Add to path
    django_project_path = os.path.expanduser('~/dev/mouse-cloud')
    if django_project_path not in sys.path:
        sys.path.append(django_project_path)

    # Set environment variable
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mouse2.settings")
    
    # Setup django
    django.setup()

try:
    if not NO_DJANGO:
        _setup_django()
except ImportError:
    # this happens if mouse-cloud doesn't exist at the expected path
    # it can also happen if a mouse-cloud requirement isn't installed
    #
    # and see what happens
    pass

# Now we can import the django modules
try:
    import runner
    import whisk_video
except ImportError:
    pass

# Import the sub-modules
from . import behavior
from . import neural
from . import whisker