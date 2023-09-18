Installation
================

Start by downloading any release from the `VND download page <https://www.ks.uiuc.edu/Research/vnd/vnd-1.9.4/files/alpha/>`_. The instructions below are based on the `original VND installation guide <https://www.ks.uiuc.edu/Research/vnd/current/ig-alt.html>`_.

Use whatever vnd-[...].tar.gz version you downloaded instead of “vnd-1.9.4a53p7b.LINUXAMD64.opengl.tar.gz”.

You can use any location you have read/write access to. This guide uses $VSC_DATA (e.g. /data/leuven/338/vsc33895) as an example.

Type what is after $ in your terminal prompt::

   $ mkdir $VSC_DATA/vnd
   $ cd $VSC_DATA/vnd
   $ mkdir lib
   $ mkdir bin
   $ mkdir ../my-temp
   $ cd ../my-temp
   $ mkdir vnd-installer
   $ cd vnd-installer
   $ mv ~/Downloads/vnd-1.9.4a53p7b.LINUXAMD64.opengl.tar.gz $VSC_DATA/my-temp/vnd-installer
   $ cd $VSC_DATA/my-temp/vnd-installer
   $ gunzip vnd-1.9.4a53p7b.LINUXAMD64.opengl.tar.gz
   $ tar xvf vnd-1.9.4a53p7b.LINUXAMD64.opengl.tar
   $ cd vnd-1.9.4a53

Edit the following two lines (use the full path instead of $VSC_DATA) in ./configure in the terminal with ``$ vi configure`` or use any text editor::
      
   install_bin_dir="/data/leuven/338/vsc33895/vnd/bin";

   install_library_dir="/data/leuven/338/vsc33895/vnd/lib/$install_name";

Check you are still in $VSC_DATA/my-temp/vnd-installer/vnd-1.9.4a53 with ``$ pwd``, then continue the installation::

   $ ./configure
   $ cd src
   $ make install


Optionally, you can add the binary directory ($VSC_DATA/vnd/bin) to your PATH variable.
A bash user might instead add this line to ~/.bashrc::

   export PATH="${VSC_DATA}/vnd/bin:$PATH"

In a new terminal window, type ``$ which vnd`` to check where vnd is installed or type ``vnd`` ro run VND.

If the last step doesn't seem to work, a possible workaround installation is to explicitly point to the binary directory first::
   
   $ $VSC_DATA/vnd/bin/vnd