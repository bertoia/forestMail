# Welcome!  This RStudio Amazon AMI contains RStudio Server version
# 0.99.484, running R 3.2.2 on Ubuntu 14.04 LTS.
# AMI created by Louis Aslett (http://www.louisaslett.com/).  If you've
# any comments or suggestions please mail aslett@stats.ox.ac.uk

# NOTE: It is *highly* recommended that you immediately change the
# default password for logging into RStudio, which you can do by logging
# in via SSH (recommended) in the usual EC2 fashion.  Alternatively,
# since this AMI was created to make RStudio Server accessible to those
# who are less comfortable with Linux commands you can follow the
# instructions below to change it without touching Linux.

# There is now a mini package where functions to manipulate the server will be
# placed.  This includes a function to change the password.  First load the
# package:
library("RStudioAMI")

# Now you can change the password by just running the following function.  It
# will prompt you to provide the existing password (just rstudio) and then type
# in a new password
passwd()

# It is strongly recommended that you clear the console so that the password is
# not visible after running the function.  Either press Ctrl+L or go to
# Edit -> Clear Console within the RStudio interface.

# There is also a new function to assist with linking to a Dropbox account to
# make loading scripts/data on and off the server much easier.  Just run the
# following function once you are ready to link to your Dropbox and follow the
# instructions
linkDropbox()

# Once Dropbox is linked you will notice a new folder called Dropbox appear in
# the Files pane on the right ===>
# This will begin syncing immediately.  If you have a large Dropbox then it is
# strongly recommended that you selectively sync only what you need.  Use the
# excludeSyncDropbox and includeSyncDropbox functions for this.
