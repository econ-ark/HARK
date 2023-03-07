# texmf-local contains customizations of LaTeX

If you use TeXLive on more than one system, or under more than one username,
it is useful to have a set of customizations that are shared across all your
identities and machines.

This can be accomplished in a number of ways.

The easiest is probably to install a file sync service like Dropbox that runs
automatically on all of the machines/users in question, and to have a master
version of texmf-local that lives in Dropbox.

Then you have two options:

1. On unix-based machines (Mac or Linux, say), you can replace the texmf-local
   file on your machine with a symbolic link to the filepath in Dropbox
1. Windows does not allow symbolic links. So on Windows machines you will need
   to modify the texmf.cnf configuration file so that it knows to look in the
   right place (your Dropbox path) for the texmf-local config files
