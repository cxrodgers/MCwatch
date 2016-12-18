"""Interacting with the mouse cloud database

Importing this module triggers the following
- Adds the mouse-cloud github repository to the path
- Sets django environment dir and calls django.setup()

The mouse-cloud database can then be accessed via the django ORM.

Code for analyzing behavior, neural, and whisker data is contained in
those submodules. I'll try to allow importing those submodules without
triggering or requiring access to the django database.
"""