# GSoC 2016 Nmap, Machine Learning Project, IPv6 OS detection

This is the submission repository for my GSoC 2016 project. This is meant to be a public platform to display the work that has been done by me with the help of my mentors, Mathias Morbitzer and Alexandru Geana, and the nmap team.

## Its primarily divided into the following parts:
* Random forest code
* Multi-stage random forest code (these are the two models I tried on the data, finally went ahead with multi-stage random forest)
* Change in dataset representation (Unfortunately the IPv6 fingerprints in use contain private info hence I have just shared the methodolgy in use for the dataset representation conversion)
* Code for using the multi-stage random forest predictor in nmap. (This includes changing nmap's makefile and changing prediction code to use opencv models)
