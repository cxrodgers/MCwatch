"""Interacting with the mouse cloud database

Importing this module triggers the following
- Adds the mouse-cloud github repository to the path
  This means that the currently active branch will be in use.
- Sets DJANGO_SETTINGS_MODULE and calls django.setup()
  This will call code in settings.py and local_settings.py in the
  mouse-cloud project.

The mouse-cloud database can then be accessed via the django ORM.

Code for analyzing behavior, neural, and whisker data is contained in
those submodules. I woud like to allow importing those submodules without
triggering or requiring access to the django database, but I don't know
how because this __init__ will always be run.
"""

import os
import sys
import django

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
    django_project_path = os.path.expanduser('~/dev/mouse-cloud3')
    if django_project_path not in sys.path:
        sys.path.append(django_project_path)

    # Set environment variable
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mouse2.settings")
    
    # Setup django
    django.setup()

try:
    _setup_django()
except ImportError:
    # this happens if mouse-cloud doesn't exist at the expected pat
    pass

# Now we can import the django modules
try:
    import runner
    import whisk_video
except ImportError:
    pass

# Import the sub-modules
import behavior
import neural
import whisker