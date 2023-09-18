[Back to ToC](/docs/manual/README.md)

# ESAT server

You can access any ESAT server with a Virtual Machine (VM) running either Linux or Windows using oVirt. The VM will provide you with a graphical interface to work on the ESAT server. You can also access the server through the command line of your own computer. In either case, make sure you are connected to a KUL network or use the [VPN](https://admin.kuleuven.be/icts/services/extranet/ssl-vpn-PulseIvanti-client-en).

## Accesing an ESAT server through the command line

- Open a terminal window
- ```
  $ ssh -J [r-number]@ssh.esat.kuleuven.be [r-number]@[server].esat.kuleuven.be
  ```
  [List of accessible student servers](https://wiki.esat.kuleuven.be/it/StudentServers)

## Starting a VM 
  
  1. Go to https://vdi-portal.esat.kuleuven.be/ <br> (Or go to https://wiki.esat.kuleuven.be/it/FrontPage &ndash;you will need to log in with your ESAT credentials&ndash; and click *Virtual Desktop Portal*).
  2. Log in with your ESAT *Username* and *Password*
  3. Click *Take a Virtual Machine* <br> (Linux: stud-r8-? / Windows: stud-w10-?)
  4. Refreshing the page after a couple of seconds will allow you to open the VM either in the browser or with virt-viewer.

## Accessing a VM

### In the browser

1. Click the arrow next to *SPICE Console*. Select *VNC Console (Browser)* from the drop-down menu. This opens the VM in the same tab.


### With virt-viewer
1. If not already installed, get virt-viewer from https://virt-manager.org/download (Win x64 MSI (gpg) for Windows).
2. In oVirt, click *SPICE Console*. This will download a tiny file called *console.vv*. 
3. Opening *console.vv* will start a session of the VM in virt-viewer.
4. If you open a terminal on the VM, it will be connected to ESAT's login node. Make sure to switch to a suitable server when running any computationally intensive jobs. when you are already connected to any ESAT server, you can use a shorter command than described [above](#accesing-an-esat-server-through-the-command-line).
    ```
    $ ssh [server]
    ```
    [List of accessible student servers](https://wiki.esat.kuleuven.be/it/StudentServers)

## Closing a VM

1. Exiting the session without shutting it down will leave the VM running such that you can continue your work the next time you open the VM. However, inactive VMs are automatically shut down after 48 hours (you will receive a reminder after 24 hours via email).
2. You can manually shut down a session by selecting *Shutdown* from the drop-down menu (arrow next to *SPICE Console*).



